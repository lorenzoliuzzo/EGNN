import argparse
import ast
import os
from pathlib import Path

import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from torch.amp import GradScaler, autocast  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.actions import (  # noqa: E402
    covariant_kinetic_loss,
    higgs_potential_loss,
    pseudofermion_action,
    wilson_plaquette_loss,
)
from src.configs import SMConfig  # noqa: E402
from src.groups import SMGateFactory  # noqa: E402
from src.lattice import get_pbc_edge_index, get_plaquette_indices  # noqa: E402
from src.models import ALL_SPECIES, SM_HeteroGNN  # noqa: E402
from src.plotting import generate_sm_report  # noqa: E402


class SMLatticeDiskDataset(Dataset):
    """Lazy-loads pre-generated lattice configurations from disk."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.file_list = sorted(p.name for p in self.data_dir.glob("*.pt"))

        sample = torch.load(self.data_dir / self.file_list[0])
        self.species_dims = {k: v.size(-1) for k, v in sample['z_dict'].items()}

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.data_dir / self.file_list[idx])


def get_lattice_parameters(config: SMConfig) -> tuple[dict, dict, float]:
    """Convert physical GeV/fm units into dimensionless lattice units."""
    # 1 fm ~ 5.068 GeV^-1; a_inv is the lattice energy scale in GeV
    a_inv = 1.0 / (config.lattice.a_spacing * 5.0677)

    # kappa = 1 / (2 a m + 8) for 4D Wilson fermions
    def to_kappa(m_gev: float) -> float:
        am = m_gev / a_inv
        return 1.0 / (2.0 * am + 8.0)

    kappa_dict = {}
    generation_masses = {
        1: (config.fermions.m_u, config.fermions.m_d, config.fermions.m_e),
        2: (config.fermions.m_c, config.fermions.m_s, config.fermions.m_mu),
        3: (config.fermions.m_t, config.fermions.m_b, config.fermions.m_tau),
    }
    for g, (m_up, m_down, m_lepton) in generation_masses.items():
        kappa_dict[f'quark_left_g{g}'] = to_kappa(m_up)
        kappa_dict[f'quark_up_right_g{g}'] = to_kappa(m_up)
        kappa_dict[f'quark_down_right_g{g}'] = to_kappa(m_down)
        kappa_dict[f'lepton_left_g{g}'] = to_kappa(m_lepton)
        kappa_dict[f'lepton_right_g{g}'] = to_kappa(m_lepton)
        kappa_dict[f'lepton_nu_right_g{g}'] = to_kappa(1e-9)

    return kappa_dict, config.gauge.betas, config.higgs.lambda_coupling


def main(config: SMConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    lat = config.lattice
    tc = config.training
    print(f"Lattice L: {lat.L} | Dimensions: {lat.dims}")

    base_edge_index, base_edge_dirs = get_pbc_edge_index(lat.L, lat.dims, device)
    p1, p2, p3, p4 = get_plaquette_indices(lat.L, lat.dims, base_edge_index, device)

    static_edge_index_dict = {(s, 'transport', s): base_edge_index for s in ALL_SPECIES}
    static_edge_dirs_dict = {(s, 'transport', s): base_edge_dirs for s in ALL_SPECIES}

    kappa_dict, betas, lam = get_lattice_parameters(config)

    model = SM_HeteroGNN(lat.hidden_dim, kappa_dict, device).to(device)
    opt_nn = torch.optim.Adam(model.parameters(), lr=tc.lr_nn)
    scaler = GradScaler()

    # batch_size=1: a single lattice graph is already thousands of nodes, and
    # batching heterogeneous graphs would need index offsetting for no benefit
    dataset = SMLatticeDiskDataset("data/L8_4D")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    history = {k: [] for k in ['loss_total', 'loss_higgs', 'loss_kinetic_boson',
                               'loss_action_fermion', 'loss_gauge_su3',
                               'loss_gauge_su2', 'loss_flow', 'vev']}

    for epoch in range(tc.epochs):
        model.train()
        ep_metrics = {k: 0.0 for k in history.keys()}

        for batch in dataloader:
            opt_nn.zero_grad()

            z_dict = {k: v.squeeze(0).to(device, non_blocking=True)
                      for k, v in batch['z_dict'].items()}
            u_dict = {k: v.squeeze(0).to(device, non_blocking=True)
                      for k, v in batch['u_dict'].items()}

            with autocast("cuda"):
                out_dict, log_det = model(z_dict, static_edge_index_dict,
                                          static_edge_dirs_dict, u_dict)

                total_dof = (sum(dataset.species_dims.values())
                             * (lat.L ** lat.dims) * lat.hidden_dim)
                loss_flow = -1.0 * log_det / total_dof

                loss_gauge_su3 = betas['su3'] * wilson_plaquette_loss(u_dict['su3'], p1, p2, p3, p4)
                loss_gauge_su2 = betas['su2'] * wilson_plaquette_loss(u_dict['su2'], p1, p2, p3, p4)

                gates = SMGateFactory(u_dict).get_all_gates()

                loss_kinetic_boson = torch.tensor(0.0, device=device)
                loss_action_fermion = torch.tensor(0.0, device=device)

                for species in ALL_SPECIES:
                    if species not in out_dict:
                        continue
                    edge_type = (species, 'transport', species)
                    gate = gates[edge_type]

                    if species == 'higgs':
                        loss_kinetic_boson += covariant_kinetic_loss(
                            out_dict[species], base_edge_index, gate)
                    else:
                        # chi is the flow output: with the CG solution detached,
                        # d(chi^dag Y)/d chi* = Y is the exact gradient, so the
                        # fermion action backpropagates into the flow
                        loss_action_fermion += pseudofermion_action(
                            chi=out_dict[species],
                            edge_index=base_edge_index,
                            edge_dirs=base_edge_dirs,
                            u_gate=gate,
                            dirac_layer=model.physics_layers[species],
                        )

                loss_higgs, mean_mag_sq = higgs_potential_loss(
                    out_dict['higgs'], lat.v_target, lam)

                # the Wilson terms act on dataset links (constants w.r.t. the
                # model), so they are logged but kept out of the backprop loss
                loss = loss_flow + loss_kinetic_boson + loss_action_fermion + loss_higgs

                scaler.scale(loss).backward()
                scaler.step(opt_nn)
                scaler.update()

                ep_metrics['loss_total'] += loss.item()
                ep_metrics['loss_flow'] += loss_flow.item()
                ep_metrics['loss_gauge_su3'] += loss_gauge_su3.item()
                ep_metrics['loss_gauge_su2'] += loss_gauge_su2.item()
                ep_metrics['loss_kinetic_boson'] += loss_kinetic_boson.item()
                ep_metrics['loss_action_fermion'] += loss_action_fermion.item()
                ep_metrics['loss_higgs'] += loss_higgs.item()
                ep_metrics['vev'] += torch.sqrt(mean_mag_sq).item()

                del z_dict, u_dict, out_dict

        num_steps = len(dataloader)
        for k in history.keys():
            history[k].append(ep_metrics[k] / num_steps)

        print(f"Epoch {epoch:03d} | "
              f"Tot: {history['loss_total'][-1]:.3f} | "
              f"SU3: {history['loss_gauge_su3'][-1]:.3f} | "
              f"Fermion: {history['loss_action_fermion'][-1]:.3f} | "
              f"Higgs: {history['loss_higgs'][-1]:.3f} | "
              f"VEV: {history['vev'][-1]:.3f}")

        if (epoch + 1) % tc.check_interval == 0 or epoch == tc.epochs - 1:
            model.eval()
            with torch.no_grad():
                dummy_batch = next(iter(dataloader))
                z_final = {str(k): v.squeeze(0).to(device)
                           for k, v in dummy_batch['z_dict'].items()}
                u_final = {str(k): v.squeeze(0).to(device)
                           for k, v in dummy_batch['u_dict'].items()}
                e_idx = {ast.literal_eval(k) if isinstance(k, str) else k: v.squeeze(0).to(device)
                         for k, v in dummy_batch['edge_index_dict'].items()}
                e_dir = {ast.literal_eval(k) if isinstance(k, str) else k: v.squeeze(0).to(device)
                         for k, v in dummy_batch.get('edge_dirs_dict', {}).items()}

                final_out_dict, _ = model(z_final, e_idx, e_dir, u_final)
                generate_sm_report(history, final_out_dict, u_final,
                                   lat.L, lat.dims, epoch, folder="plots")
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(SMConfig.from_yaml(args.config))
