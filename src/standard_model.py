import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing

from src.actions import HiggsAction, WilsonAction
from src.dirac import g5_spin_last, get_chiral_projectors, get_gamma_matrices
from src.groups import get_gate
from src.lattice import create_lattice, find_rectangular_loops

# The heterogeneous Standard Model: one message-passing operator per particle
# type, sharing a single spacetime lattice. Note the physics caveat: the
# chirality-split electroweak transport combined with a Wilson term is not an
# exactly gauge-invariant chiral lattice theory (Nielsen-Ninomiya); quantitative
# claims should rest on the vector-like sectors.


class QuarkGaugeConv(MessagePassing):
    """SU(3) x SU(2)_L x U(1)_Y transport for [N, Flavor, Color, Isospin, Spin]."""

    def __init__(self, r: float = 1.0, a: float = 1.0):
        super().__init__(aggr='add', node_dim=0)
        self.r, self.a = r, a

    def forward(self, x, edge_index, u_su3, u_su2, u_u1, edge_dirs, is_fwd,
                gammas, id_spin, P_L, P_R, **kwargs):
        return self.propagate(edge_index, x=x, u_su3=u_su3, u_su2=u_su2, u_u1=u_u1,
                              edge_dirs=edge_dirs, is_fwd=is_fwd, gammas=gammas,
                              id_spin=id_spin, P_L=P_L, P_R=P_R)

    def message(self, x_j, u_su3, u_su2, u_u1, edge_dirs, is_fwd, gammas,
                id_spin, P_L, P_R):
        x_L = torch.einsum('ab, efcib -> efcia', P_L, x_j)
        x_R = torch.einsum('ab, efcib -> efcia', P_R, x_j)

        # color transport applies to both chiralities
        x_L_color = torch.einsum('ecd, efdis -> efcis', u_su3, x_L)
        x_R_color = torch.einsum('ecd, efdis -> efcis', u_su3, x_R)

        # left-handed states feel SU(2) x U(1); right-handed only U(1)
        ew_link_L = u_u1 * u_su2
        x_L_full = torch.einsum('eij, efcjs -> efcis', ew_link_L, x_L_color)
        x_R_full = x_R_color * u_u1.view(-1, 1, 1, 1, 1)

        x_transported = x_L_full + x_R_full

        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1)
        P_edge = (self.r / self.a) * id_spin + (1.0 / self.a) * sign * gammas[edge_dirs]
        return torch.einsum('eab, efcib -> efcia', P_edge, x_transported)


class LeptonGaugeConv(MessagePassing):
    """SU(2)_L x U(1)_Y transport for colorless [N, Flavor, Isospin, Spin]."""

    def __init__(self, r: float = 1.0, a: float = 1.0):
        super().__init__(aggr='add', node_dim=0)
        self.r, self.a = r, a

    def forward(self, x, edge_index, u_su2, u_u1, edge_dirs, is_fwd,
                gammas, id_spin, P_L, P_R, **kwargs):
        return self.propagate(edge_index, x=x, u_su2=u_su2, u_u1=u_u1,
                              edge_dirs=edge_dirs, is_fwd=is_fwd, gammas=gammas,
                              id_spin=id_spin, P_L=P_L, P_R=P_R)

    def message(self, x_j, u_su2, u_u1, edge_dirs, is_fwd, gammas, id_spin, P_L, P_R):
        x_L = torch.einsum('ab, efib -> efia', P_L, x_j)
        x_R = torch.einsum('ab, efib -> efia', P_R, x_j)

        ew_link_L = u_u1 * u_su2
        x_L_full = torch.einsum('eij, efjs -> efis', ew_link_L, x_L)
        x_R_full = x_R * u_u1.view(-1, 1, 1, 1)

        x_transported = x_L_full + x_R_full

        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1)
        P_edge = (self.r / self.a) * id_spin + (1.0 / self.a) * sign * gammas[edge_dirs]
        return torch.einsum('eab, efib -> efia', P_edge, x_transported)


class HiggsGaugeConv(MessagePassing):
    """SU(2) x U(1) kinetic transport for the Higgs doublet [N, Isospin]."""

    def __init__(self, a: float = 1.0):
        super().__init__(aggr='add', node_dim=0)
        self.a = a

    def forward(self, x, edge_index, u_su2, u_u1, **kwargs):
        return self.propagate(edge_index, x=x, u_su2=u_su2, u_u1=u_u1)

    def message(self, x_j, u_su2, u_u1):
        u_total = u_u1 * u_su2
        return (1.0 / self.a) * torch.einsum('eij, ej -> ei', u_total, x_j)


class StandardModelGNN(nn.Module):
    def __init__(self, num_flavors: int = 3, a: float = 1.0, r: float = 1.0,
                 bare_mass: float = 0.01):
        super().__init__()
        self.a = a
        self.r = r
        self.bare_mass = bare_mass

        gammas, g5 = get_gamma_matrices()
        id_spin = torch.eye(4, dtype=torch.complex64)
        P_L, P_R = get_chiral_projectors(g5)

        self.register_buffer('gammas', gammas)
        self.register_buffer('g5', g5)
        self.register_buffer('id_spin', id_spin)
        self.register_buffer('P_L', P_L)
        self.register_buffer('P_R', P_R)

        # random Yukawa couplings; physically these map to measured masses
        self.y_quarks = nn.Parameter(torch.rand(num_flavors, dtype=torch.complex64))
        self.y_leptons = nn.Parameter(torch.rand(num_flavors, dtype=torch.complex64))

        self.gauge_interactions = HeteroConv({
            ('quark', 'gauge', 'quark'): QuarkGaugeConv(r, a),
            ('lepton', 'gauge', 'lepton'): LeptonGaugeConv(r, a),
            ('higgs', 'gauge', 'higgs'): HiggsGaugeConv(a),
        }, aggr='sum')

    def forward(self, x_dict: dict, edge_index_dict: dict, u_dict: dict,
                is_fwd: torch.Tensor, edge_dirs: torch.Tensor) -> dict:
        edge_types = list(edge_index_dict.keys())

        # HeteroConv unpacks each '<name>_dict' into the kwarg '<name>' of the
        # sub-convolution registered for that edge type
        transported_dict = self.gauge_interactions(
            x_dict,
            edge_index_dict,
            u_su3_dict={et: u_dict['su3'] for et in edge_types},
            u_su2_dict={et: u_dict['su2'] for et in edge_types},
            u_u1_dict={et: u_dict['u1'] for et in edge_types},
            edge_dirs_dict={et: edge_dirs for et in edge_types},
            is_fwd_dict={et: is_fwd for et in edge_types},
            gammas_dict={et: self.gammas for et in edge_types},
            id_spin_dict={et: self.id_spin for et in edge_types},
            P_L_dict={et: self.P_L for et in edge_types},
            P_R_dict={et: self.P_R for et in edge_types},
        )

        out_dict = {}
        phi = x_dict['higgs']
        phi_vev = torch.sqrt(torch.sum(phi.abs() ** 2, dim=-1))
        dim_space = len(torch.unique(edge_dirs))

        for p_type, y_couplings in zip(['quark', 'lepton'],
                                       [self.y_quarks, self.y_leptons]):
            psi = x_dict[p_type]
            transported_psi = transported_dict[p_type]

            # dynamical mass from the Higgs magnitude at each node
            dynamic_mass = self.bare_mass + torch.einsum('f, n -> nf', y_couplings, phi_vev)
            if p_type == 'quark':
                m_view = dynamic_mass.view(-1, len(y_couplings), 1, 1, 1)
            else:
                m_view = dynamic_mass.view(-1, len(y_couplings), 1, 1)

            local_term = (m_view + (self.r / self.a) * dim_space) * psi
            out_dict[p_type] = local_term - 0.5 * transported_psi

        # each direction contributes twice (+mu and -mu edges), so the local
        # coefficient of the covariant Laplacian is 2 * dims
        out_dict['higgs'] = (2 * dim_space / self.a) * phi - transported_dict['higgs']
        return out_dict


def dict_dot(dict_a: dict, dict_b: dict) -> torch.Tensor:
    return sum(torch.sum(dict_a[k].conj() * dict_b[k]).real for k in ['quark', 'lepton'])


def dict_add(dict_a: dict, dict_b: dict, alpha: float = 1.0) -> dict:
    return {k: dict_a[k] + alpha * dict_b[k] for k in ['quark', 'lepton']}


@torch.no_grad()
def hetero_conjugate_gradient(
    gnn_model: StandardModelGNN,
    pf_dict: dict,
    higgs_phi: torch.Tensor,
    edge_index_dict: dict,
    u_dict: dict,
    is_fwd: torch.Tensor,
    edge_dirs: torch.Tensor,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    g5 = gnn_model.g5

    def apply_m_dag_m(vec_dict: dict) -> dict:
        input_dict = {'quark': vec_dict['quark'], 'lepton': vec_dict['lepton'],
                      'higgs': higgs_phi}
        d_out = gnn_model(input_dict, edge_index_dict, u_dict, is_fwd, edge_dirs)
        g5_d = {
            'quark': g5_spin_last(g5, d_out['quark']),
            'lepton': g5_spin_last(g5, d_out['lepton']),
            'higgs': higgs_phi,
        }
        d_dag = gnn_model(g5_d, edge_index_dict, u_dict, is_fwd, edge_dirs)
        return {
            'quark': g5_spin_last(g5, d_dag['quark']),
            'lepton': g5_spin_last(g5, d_dag['lepton']),
        }

    x = {k: torch.zeros_like(v) for k, v in pf_dict.items()}
    r = {k: v.clone() for k, v in pf_dict.items()}
    p = {k: v.clone() for k, v in pf_dict.items()}
    rsold = dict_dot(r, r)

    for _ in range(max_iter):
        ap = apply_m_dag_m(p)
        alpha = rsold / (dict_dot(p, ap) + 1e-10)
        x = dict_add(x, p, alpha)
        r = dict_add(r, ap, -alpha)
        rsnew = dict_dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = dict_add(r, p, rsnew / rsold)
        rsold = rsnew

    return x


class HeteroPseudofermionAction(nn.Module):
    def __init__(self, gnn_model: StandardModelGNN):
        super().__init__()
        self.gnn = gnn_model

    def forward(self, pf_dict, higgs_phi, edge_index_dict, u_dict, is_fwd, edge_dirs):
        x_solved = hetero_conjugate_gradient(
            self.gnn, pf_dict, higgs_phi, edge_index_dict, u_dict, is_fwd, edge_dirs)
        actual_action = dict_dot(pf_dict, x_solved)

        # surrogate trick: -|Dx|^2 carries the gradients, the detach cancels
        # its value, so we get the action's value with the action's gradient
        input_dict = {'quark': x_solved['quark'], 'lepton': x_solved['lepton'],
                      'higgs': higgs_phi}
        d_x = self.gnn(input_dict, edge_index_dict, u_dict, is_fwd, edge_dirs)
        surrogate = -(torch.sum(d_x['quark'].abs() ** 2) + torch.sum(d_x['lepton'].abs() ** 2))
        return surrogate - surrogate.detach() + actual_action


class StandardModelUniverse(nn.Module):
    def __init__(self, gnn_model: StandardModelGNN, plaq_idx: torch.Tensor,
                 beta_su3: float = 5.5, beta_su2: float = 2.0, beta_u1: float = 1.0,
                 higgs_v: float = 1.0, higgs_lam: float = 0.1):
        super().__init__()
        self.gnn = gnn_model
        self.action_su3 = WilsonAction(plaq_idx, group_dim=3, beta=beta_su3)
        self.action_su2 = WilsonAction(plaq_idx, group_dim=2, beta=beta_su2)
        self.action_u1 = WilsonAction(plaq_idx, group_dim=1, beta=beta_u1)
        self.action_higgs = HiggsAction(v=higgs_v, lam=higgs_lam)
        self.action_fermions = HeteroPseudofermionAction(self.gnn)

    def forward(self, hetero_data, u_dict, pf_dict, is_fwd, edge_dirs):
        s_su3 = self.action_su3(u_dict['su3'])
        s_su2 = self.action_su2(u_dict['su2'])
        s_u1 = self.action_u1(u_dict['u1'])

        higgs_edge_index = hetero_data['higgs', 'gauge', 'higgs'].edge_index
        s_higgs = self.action_higgs(hetero_data['higgs'].x, higgs_edge_index,
                                    is_fwd, u_dict['su2'], u_dict['u1'])

        s_fermions = self.action_fermions(
            pf_dict, hetero_data['higgs'].x, hetero_data.edge_index_dict,
            u_dict, is_fwd, edge_dirs)

        total_action = s_su3 + s_su2 + s_u1 + s_higgs + s_fermions
        metrics = {
            'su3': s_su3.item(), 'su2': s_su2.item(), 'u1': s_u1.item(),
            'higgs': s_higgs.item(), 'fermion': s_fermions.item(),
        }
        return total_action, metrics


class HeteroQuantumVacuum(nn.Module):
    def __init__(self, lattice_shape: tuple[int, ...], num_flavors: int = 3,
                 a: float = 1.0,
                 groups: dict | None = None, betas: dict | None = None,
                 v: float = 1.0, lam: float = 0.5):
        super().__init__()
        groups = groups or {'su3': 3, 'su2': 2, 'u1': 1}
        betas = betas or {'su3': 6.0, 'su2': 4.0, 'u1': 5.0}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lattice_shape = lattice_shape
        self.num_flavors = num_flavors
        self.groups = groups

        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        plaq_idx = find_rectangular_loops(lattice_shape, edge_index)

        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_dirs', edge_dirs)
        self.register_buffer('is_fwd', is_fwd)
        self.register_buffer('partner_map', partner_map)

        self.data = HeteroData()
        self.data['quark', 'gauge', 'quark'].edge_index = edge_index
        self.data['lepton', 'gauge', 'lepton'].edge_index = edge_index
        self.data['higgs', 'gauge', 'higgs'].edge_index = edge_index
        self.data = self.data.to(self.device)

        num_edges = edge_index.size(1)
        num_nodes = int(np.prod(lattice_shape))

        # cold start on the group identity
        self.u_raw = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(num_edges, dim, dim, dtype=torch.complex64))
            for name, dim in groups.items()
        })
        self.higgs_phi = nn.Parameter(torch.ones(num_nodes, 2, dtype=torch.complex64) * 0.1)

        # pseudofermion heat-bath buffers
        self.register_buffer('pf_quark', torch.randn(num_nodes, num_flavors, 3, 2, 4,
                                                     dtype=torch.complex64))
        self.register_buffer('pf_lepton', torch.randn(num_nodes, num_flavors, 2, 4,
                                                      dtype=torch.complex64))

        gnn_model = StandardModelGNN(num_flavors=num_flavors, a=a)
        self.universe = StandardModelUniverse(
            gnn_model, plaq_idx,
            beta_su3=betas['su3'], beta_su2=betas['su2'], beta_u1=betas['u1'],
            higgs_v=v, higgs_lam=lam)

        # per-sector Langevin steps: SU(3) is the stiffest manifold
        self.dt_map = {'su3': 0.00005, 'su2': 0.0005, 'u1': 0.001, 'phi': 0.001}

        self.to(self.device)

    def physical_gates(self) -> dict:
        return {
            name: get_gate(raw, self.partner_map, self.is_fwd, is_su=(name != 'u1'))
            for name, raw in self.u_raw.items()
        }

    def langevin_step(self, parameters, dt: float) -> None:
        with torch.no_grad():
            for p in parameters:
                if p.grad is None:
                    continue

                # clip the force so a single stiff gradient cannot blow up the
                # Euler-Maruyama step
                force = p.grad.resolve_conj()
                force_norm = torch.norm(force)
                if force_norm > 10.0:
                    force = force * (10.0 / force_norm)

                noise = torch.randn_like(p)
                if p.is_complex():
                    noise_imag = torch.randn_like(p)
                    # complex noise variance splits across real and imaginary
                    noise = (noise + 1j * noise_imag) / np.sqrt(2)

                p.add_(-dt * force + np.sqrt(2 * dt) * noise)

    def refresh_heat_bath(self) -> None:
        # phi = M^dag eta (M^dag = g5 M g5, eta ~ complex N(0,1)) so that
        # S_pf = phi^dag (M^dag M)^-1 phi carries the fermion determinant
        # weight; a plain Gaussian phi would sample the wrong ensemble
        with torch.no_grad():
            gates = self.physical_gates()
            gnn = self.universe.gnn
            g5 = gnn.g5
            g5_eta = {
                'quark': g5_spin_last(g5, torch.randn_like(self.pf_quark)),
                'lepton': g5_spin_last(g5, torch.randn_like(self.pf_lepton)),
                'higgs': self.higgs_phi,
            }
            d_out = gnn(g5_eta, self.data.edge_index_dict, gates, self.is_fwd,
                        self.edge_dirs)
            self.pf_quark.copy_(g5_spin_last(g5, d_out['quark']))
            self.pf_lepton.copy_(g5_spin_last(g5, d_out['lepton']))

    def find_vacuum(self, steps: int = 1000, thermalize_steps: int = 200) -> dict:
        print("--- Heterogeneous Standard Model: Quantum Thermalization ---")
        history = {k: [] for k in ['total', 'vev', 'su3', 'su2', 'u1', 'higgs', 'fermion']}

        for step in range(steps):
            self.zero_grad()

            gates = self.physical_gates()
            self.data['higgs'].x = self.higgs_phi
            pf_dict = {'quark': self.pf_quark, 'lepton': self.pf_lepton}

            total_action, metrics = self.universe(
                self.data, gates, pf_dict, self.is_fwd, self.edge_dirs)
            total_action.backward()

            for name, param in self.u_raw.items():
                self.langevin_step([param], dt=self.dt_map[name])
            self.langevin_step([self.higgs_phi], dt=self.dt_map['phi'])

            if step % 10 == 0:
                self.refresh_heat_bath()

            current_vev = torch.norm(self.higgs_phi, dim=-1).mean().item()
            if step >= thermalize_steps:
                history['total'].append(total_action.item())
                history['vev'].append(current_vev)
                for k, val in metrics.items():
                    history[k].append(val)

            if step % 50 == 0:
                phase = "Thermalizing" if step < thermalize_steps else "Sampling    "
                print(f"[{phase}] Step {step:03d} | Action: {total_action.item():.2f} | "
                      f"SU3: {metrics['su3']:.2f} | Fermions: {metrics['fermion']:.2f} | "
                      f"VEV: {current_vev:.3f}")

        return history
