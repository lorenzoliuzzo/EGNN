import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.actions import ElectroweakHiggsAction, HiggsAction, PseudofermionAction, WilsonAction
from src.dirac import g5_spin_last
from src.groups import get_gate
from src.lattice import create_lattice, find_rectangular_loops


class VacuumFinder(nn.Module):
    """SU(3) x SU(2) x U(1) + Higgs + pseudofermion vacuum via per-sector Adam."""

    def __init__(self, lattice_shape: tuple[int, ...],
                 groups: dict | None = None, betas: dict | None = None,
                 v: float = 1.0, lam: float = 0.5):
        super().__init__()
        groups = groups or {'su3': 3, 'su2': 2, 'u1': 1}
        betas = betas or {'su3': 6.0, 'su2': 4.0, 'u1': 5.0}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lattice_shape = lattice_shape
        self.groups = groups
        self.betas = betas

        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_dirs', edge_dirs)
        self.register_buffer('partner_map', partner_map)
        self.register_buffer('is_fwd', is_fwd)
        self.register_buffer('plaq_idx', find_rectangular_loops(lattice_shape, edge_index))

        num_edges = edge_index.size(1)
        num_nodes = int(np.prod(lattice_shape))

        # cold start on the group identity
        self.u_raw = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(num_edges, dim, dim, dtype=torch.complex64))
            for name, dim in groups.items()
        })
        self.phi = nn.Parameter(torch.randn(num_nodes, 2, dtype=torch.complex64))
        self.register_buffer('pf_phi', torch.randn(num_nodes, 3, 4, dtype=torch.complex64))

        self.wilson_calcs = nn.ModuleDict({
            name: WilsonAction(self.plaq_idx, group_dim=dim, beta=betas[name])
            for name, dim in groups.items()
        })
        self.higgs_calc = HiggsAction(v=v, lam=lam)
        self.fermion_calc = PseudofermionAction(y=1.0)
        self.to(self.device)

    def physical_gates(self) -> dict:
        return {
            name: get_gate(raw, self.partner_map, self.is_fwd, is_su=(name != 'u1'))
            for name, raw in self.u_raw.items()
        }

    def compute_actions(self, gates: dict) -> dict:
        return {
            'su3': self.wilson_calcs['su3'](gates['su3']),
            'su2': self.wilson_calcs['su2'](gates['su2']),
            'u1': self.wilson_calcs['u1'](gates['u1']),
            'higgs': self.higgs_calc(self.phi, self.edge_index, self.is_fwd,
                                     gates['su2'], gates['u1']),
            'fermion': self.fermion_calc(self.pf_phi, self.phi, self.edge_index,
                                         self.edge_dirs, gates['su3'], self.is_fwd),
        }

    def refresh_heat_bath(self, gates: dict) -> None:
        # phi = M^dag eta (M^dag = g5 M g5, eta ~ complex N(0,1)) so the
        # pseudofermion action carries the fermion determinant weight
        with torch.no_grad():
            dirac = self.fermion_calc.dirac
            g5_eta = g5_spin_last(dirac.g5, torch.randn_like(self.pf_phi))
            d_eta = dirac(g5_eta, self.phi, self.edge_index, self.edge_dirs,
                          gates['su3'], self.is_fwd)
            self.pf_phi.copy_(g5_spin_last(dirac.g5, d_eta))

    def find_vacuum(self, steps: int = 1000) -> dict:
        # separate optimizers so each sector cools at its own energy scale
        opt_su3 = optim.Adam([self.u_raw['su3']], lr=0.001)
        opt_ew = optim.Adam([self.u_raw['su2'], self.u_raw['u1']], lr=0.005)
        opt_higgs = optim.Adam([self.phi], lr=0.01)

        history = {k: [] for k in ['total', 'vev', 'gauge_su3', 'gauge_su2',
                                   'gauge_u1', 'higgs', 'fermion']}

        print("--- Cooling Vacuum (Multi-Optimizer Sector Cooling) ---")
        for step in range(steps):
            opt_su3.zero_grad()
            opt_ew.zero_grad()
            opt_higgs.zero_grad()

            gates = self.physical_gates()
            actions = self.compute_actions(gates)
            loss = sum(actions.values())
            loss.backward()

            # complex gradients accumulate as conjugated views; resolve them
            # before Adam touches its momentum buffers
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None and p.grad.is_complex():
                        p.grad = p.grad.resolve_conj()

            opt_su3.step()
            opt_ew.step()
            opt_higgs.step()

            if step % 10 == 0:
                self.refresh_heat_bath(gates)

            current_vev = torch.norm(self.phi, dim=-1).mean().item()
            history['total'].append(loss.item())
            history['vev'].append(current_vev)
            history['gauge_su3'].append(actions['su3'].item())
            history['gauge_su2'].append(actions['su2'].item())
            history['gauge_u1'].append(actions['u1'].item())
            history['higgs'].append(actions['higgs'].item())
            history['fermion'].append(actions['fermion'].item())

            if step % 100 == 0:
                print(f"Step {step:03d} | Total: {loss.item():.2f} | "
                      f"SU3: {actions['su3'].item():.2f} | "
                      f"Fermion: {actions['fermion'].item():.2f} | VEV: {current_vev:.3f}")

        return history


class QuantumVacuumFinder(VacuumFinder):
    """Langevin thermalization instead of deterministic cooling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # per-sector time steps: SU(3) is the stiffest manifold
        self.dt_map = {'su3': 0.0001, 'su2': 0.001, 'u1': 0.005, 'phi': 0.01}

    def langevin_step(self, parameters, dt: float) -> None:
        with torch.no_grad():
            for p in parameters:
                if p.grad is None:
                    continue
                grad = p.grad.resolve_conj()

                noise = torch.randn_like(p)
                if p.is_complex():
                    noise_imag = torch.randn_like(p)
                    noise = (noise + 1j * noise_imag) / np.sqrt(2)

                p.add_(-dt * grad + np.sqrt(2 * dt) * noise)

    def find_vacuum(self, steps: int = 1000) -> dict:
        history = {k: [] for k in ['total', 'vev', 'gauge_su3', 'gauge_su2',
                                   'gauge_u1', 'higgs', 'fermion']}

        for step in range(steps):
            gates = self.physical_gates()
            actions = self.compute_actions(gates)
            total_action = sum(actions.values())

            self.zero_grad()
            total_action.backward()

            for name, param in self.u_raw.items():
                self.langevin_step([param], dt=self.dt_map[name])
            self.langevin_step([self.phi], dt=self.dt_map['phi'])

            if step % 10 == 0:
                self.refresh_heat_bath(gates)

            current_vev = torch.norm(self.phi, dim=-1).mean().item()
            history['total'].append(total_action.item())
            history['vev'].append(current_vev)
            history['gauge_su3'].append(actions['su3'].item())
            history['gauge_su2'].append(actions['su2'].item())
            history['gauge_u1'].append(actions['u1'].item())
            history['higgs'].append(actions['higgs'].item())
            history['fermion'].append(actions['fermion'].item())

            if step % 100 == 0:
                gauge = actions['su3'] + actions['su2'] + actions['u1']
                print(f"Step {step:03d} | Total: {total_action.item():.2f} | "
                      f"Gauge: {gauge.item():.2f} | "
                      f"Fermion: {actions['fermion'].item():.2f} | VEV: {current_vev:.3f}")

        return history


class ElectroweakSimulator(nn.Module):
    """SU(2) x U(1) + Higgs doublet, cooled with a single Adam optimizer."""

    def __init__(self, lattice_shape: tuple[int, ...] = (8, 8, 8),
                 mu_sq: float = 1.0, lam: float = 0.5,
                 beta_su2: float = 4.0, beta_u1: float = 5.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        edge_idx, dirs, fwd, p_map = create_lattice(lattice_shape)
        self.register_buffer('edge_index', edge_idx)
        self.register_buffer('is_fwd', fwd)
        self.register_buffer('partner_map', p_map)
        plaq_idx = find_rectangular_loops(lattice_shape, edge_idx)

        num_nodes, num_edges = int(np.prod(lattice_shape)), edge_idx.size(1)
        self.phi = nn.Parameter(torch.randn(num_nodes, 2, dtype=torch.complex64) * 0.1)
        self.u_su2 = nn.Parameter(torch.zeros(num_edges, 2, 2, dtype=torch.complex64))
        self.u_u1 = nn.Parameter(torch.zeros(num_edges, 1, 1, dtype=torch.complex64))

        self.w_su2 = WilsonAction(plaq_idx, 2, beta_su2)
        self.w_u1 = WilsonAction(plaq_idx, 1, beta_u1)
        self.h_calc = ElectroweakHiggsAction(mu_sq=mu_sq, lam=lam)
        self.to(self.device)

    def cool_vacuum(self, steps: int = 500, lr: float = 0.01) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=lr)
        history = {'total': [], 'vev': [], 'gauge': [], 'higgs': []}

        for _ in tqdm(range(steps)):
            optimizer.zero_grad()

            g_su2 = get_gate(self.u_su2, self.partner_map, self.is_fwd, is_su=True)
            g_u1 = get_gate(self.u_u1, self.partner_map, self.is_fwd, is_su=False)

            s_gauge = self.w_su2(g_su2) + self.w_u1(g_u1)
            s_higgs = self.h_calc(self.phi, self.edge_index, self.is_fwd, g_su2, g_u1)
            loss = s_gauge + s_higgs
            loss.backward()

            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None and p.grad.is_complex():
                        p.grad = p.grad.resolve_conj()

            optimizer.step()

            vev = torch.sqrt(torch.mean(torch.sum(self.phi.abs() ** 2, dim=-1))).item()
            history['total'].append(loss.item())
            history['vev'].append(vev)
            history['gauge'].append(s_gauge.item())
            history['higgs'].append(s_higgs.item())

        return history
