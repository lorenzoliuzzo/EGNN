import os
import ast
import argparse
import matplotlib.pyplot as plt

import math
import numpy as np
from scipy.optimize import fsolve

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torch.nn as nn
from torch.optim import Optimizer

from src.configs import SMConfig
from src.utils.lattice import get_pbc_edge_index, get_plaquette_indices
from src.physics.gauge_groups import get_sm_generators, exp_sm_algebra_to_group, SMGateFactory
from src.physics.action_losses import wilson_plaquette_loss, covariant_kinetic_loss, higgs_potential_loss, pseudofermion_action
from src.models import SM_HeteroGNN, ALL_SPECIES # Imported our new models
from src.checks import plot_full_sm_dashboard, track_confinement



# class SGLD(Optimizer):
#     """Stochastic Gradient Langevin Dynamics (SGLD) with scalar enforcement."""
#     def __init__(self, params, lr=1e-3, temperature=1.0):
#         defaults = dict(lr=float(lr), temperature=float(temperature))
#         super(SGLD, self).__init__(params, defaults)

#     def step(self, closure=None):
#         for group in self.param_groups:
#             lr = float(group['lr'])
#             temp = float(group['temperature'])
#             for p in group['params']:
#                 if p.grad is None: continue
#                 noise = torch.randn_like(p.data)
#                 sigma = torch.sqrt(torch.tensor(2 * lr * temp, device=p.device))
#                 p.data.add_(p.grad.data, alpha=-lr)
#                 p.data.add_(noise, alpha=sigma.item())



class SMLatticeDiskDataset(Dataset):
    """
    Lazy-loads pre-generated Standard Model configurations from disk.
    Ultra-low RAM footprint.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Get all .pt files in the directory
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.file_list.sort() # Ensure consistent ordering
        
        # Extract species dims dynamically from the first sample so we can calculate DoF later
        sample = torch.load(os.path.join(self.data_dir, self.file_list[0]))
        self.species_dims = {k: v.size(-1) for k, v in sample['z_dict'].items()}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Read the file from disk directly into CPU RAM
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        snapshot = torch.load(file_path)
        
        return snapshot
        


def get_lattice_parameters(config: SMConfig):
    """
    Converts physical GeV/fm units into dimensionless lattice units.
    """
    # 1. Spacing conversion: 1 fm ≈ 5.067 GeV^-1
    # a_inv is the 'energy scale' of the lattice in GeV
    a_inv = 1.0 / (config.lattice.a_spacing * 5.0677)
    
    # 2. Convert Fermion masses to Hopping Parameters (Kappa)
    # Formula: kappa = 1 / (2*am + 8) for 4D Wilson Fermions
    def to_kappa(m_gev):
        am = m_gev / a_inv
        return 1.0 / (2.0 * am + 8.0)

    kappa_dict = {
        # Generation 1
        'quark_left_g1': to_kappa(config.fermions.m_u), # Approximation
        'quark_up_right_g1': to_kappa(config.fermions.m_u),
        'quark_down_right_g1': to_kappa(config.fermions.m_d),
        'lepton_left_g1': to_kappa(config.fermions.m_e),
        'lepton_right_g1': to_kappa(config.fermions.m_e),
        'lepton_nu_right_g1': to_kappa(1e-9), # Neutrinos are nearly massless
        
        # ... Repeat for Generation 2 (m_c, m_s, m_mu) and 3 (m_t, m_b, m_tau)
    }
    
    # 3. Gauge Betas: beta = 2*N / g^2
    betas = config.gauge.betas
    
    # 4. Higgs Lambda: physical lambda is already dimensionless
    lam = config.higgs.lambda_coupling
    
    return kappa_dict, betas, lam


def main(config: SMConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    lat = config.lattice
    tc = config.training

    print(f"Lattice L: {lat.L} | Dimensions: {lat.dims}")

    # 1. Load Geometry ONCE
    # These stay on the GPU for the duration of the program
    base_edge_index, base_edge_dirs = get_pbc_edge_index(lat.L, lat.dims, device)
    p1, p2, p3, p4 = get_plaquette_indices(lat.L, lat.dims, base_edge_index, device)
    
    # Pre-calculate the edge_index_dict for the HeteroGNN
    # This prevents creating new dictionaries every step
    static_edge_index_dict = {
        (spec, 'transport', spec): base_edge_index for spec in ALL_SPECIES
    }
    static_edge_dirs_dict = {
        (spec, 'transport', spec): base_edge_dirs for spec in ALL_SPECIES
    }
    
    raw_edge_index = list(static_edge_index_dict.values())[0] 
    raw_edge_dirs = list(static_edge_dirs_dict.values())[0]

    kappa_dict, betas, lam = get_lattice_parameters(config)

    # --- 1. Initialize Model ---
    model = SM_HeteroGNN(lat.hidden_dim, kappa_dict, device).to(device)
    opt_nn = torch.optim.Adam(model.parameters(), lr=tc.lr_nn)
    scaler = GradScaler() # For Mixed Precision

    # --- 2. Initialize Lazy DataLoader ---
    # We use batch_size=1. A single lattice graph is already massive (thousands of nodes). 
    # Batching multiple heterogeneous graphs requires complex index offsetting, 
    # and batch_size=1 easily fits inside an 8GB GPU.
    dataset = SMLatticeDiskDataset("data/L8_4D")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    history = {k: [] for k in ['loss_total', 'loss_higgs', 'loss_kinetic_boson', 'loss_action_fermion', 'loss_gauge_su3', 'loss_gauge_su2', 'loss_flow', 'vev']}

    for epoch in range(tc.epochs):
        model.train()
        epoch_loss = 0.0
        ep_metrics = {k: 0.0 for k in history.keys()}

        for step, batch in enumerate(dataloader):
            opt_nn.zero_grad()

            # 2. Use Non-Blocking Transfers
            # Squeeze(0) handles the batch dimension from DataLoader
            z_dict = {k: v.squeeze(0).to(device, non_blocking=True) for k, v in batch['z_dict'].items()}
            u_dict = {k: v.squeeze(0).to(device, non_blocking=True) for k, v in batch['u_dict'].items()}
            
            for k, v in z_dict.items():
                print(f"Species {k} shape: {v.shape}")
            print(f"Edge Index shape: {list(static_edge_index_dict.values())[0].shape}")


            with autocast("cuda"):
                # B. Forward Pass (GNN builds physical fields from noise + gauge links)
                out_dict, log_det = model(z_dict, static_edge_index_dict, static_edge_dirs_dict, u_dict)

                # C. Calculate Losses
                total_dof = sum(dim for dim in dataset.species_dims.values()) * (lat.L**4) * lat.hidden_dim
                loss_flow = - 1.0 * log_det / total_dof 
                
                loss_gauge_su3 = betas['su3'] * wilson_plaquette_loss(u_dict['su3'], p1, p2, p3, p4)
                loss_gauge_su2 = betas['su2'] * wilson_plaquette_loss(u_dict['su2'], p1, p2, p3, p4)
                
                factory = SMGateFactory(u_dict)
                gates = factory.get_all_gates()
                
                loss_kinetic_boson = torch.tensor(0.0, device=device)
                loss_action_fermion = torch.tensor(0.0, device=device)
                
                # C.1 Physics Split: Evaluate Matter Fields Properly
                for species in ALL_SPECIES:
                    if species not in out_dict: continue
                    
                    edge_type = (species, 'transport', species)
                    gate = gates[edge_type]
                    
                    if species == 'higgs':
                        loss_kinetic_boson += covariant_kinetic_loss(out_dict[species], raw_edge_index, gate)
                    else:
                        dirac_layer = model.physics_layers[species]
                        # chi must be the flow output, not the raw noise: with the CG
                        # solution Y detached, d(chi^dag Y)/d chi* = Y is the exact
                        # gradient, so the fermion action backpropagates into the flow
                        loss_action_fermion += pseudofermion_action(
                            chi=out_dict[species],
                            edge_index=raw_edge_index, 
                            edge_dirs=raw_edge_dirs, 
                            u_gate=gate, 
                            dirac_layer=dirac_layer
                        )

                    torch.cuda.empty_cache()

                # C.2 Higgs Potential Loss
                loss_higgs, mean_mag_sq = higgs_potential_loss(out_dict['higgs'], lat.v_target, lam)
                
                # D. Total Physical Unsupervised Loss
                # The Wilson terms act on gauge links loaded from the dataset, which are
                # constants w.r.t. the model parameters: they are logged below but kept
                # out of the backprop loss.
                loss = loss_flow + loss_kinetic_boson + loss_action_fermion + loss_higgs
                
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # opt_nn.step()

                epoch_loss += loss.item()

                scaler.scale(loss).backward()
                scaler.step(opt_nn)
                scaler.update()

                # Critical: Clear refs immediately
                del z_dict, u_dict, out_dict

                ep_metrics['loss_total'] += loss.item()
                ep_metrics['loss_flow'] += loss_flow.item()
                ep_metrics['loss_gauge_su3'] += loss_gauge_su3.item()
                ep_metrics['loss_gauge_su2'] += loss_gauge_su2.item()
                ep_metrics['loss_kinetic_boson'] += loss_kinetic_boson.item()
                ep_metrics['loss_action_fermion'] += loss_action_fermion.item()
                ep_metrics['loss_higgs'] += loss_higgs.item()
                ep_metrics['vev'] += torch.sqrt(mean_mag_sq).item()
                
                # E. Memory Management (CRITICAL FOR 8GB GPU)
                # Explicitly delete massive intermediate tensors and clear cache
            
            
                # del z_dict, u_dict, edge_index_dict, edge_dirs_dict, out_dict, gates
                # torch.cuda.empty_cache()

        num_steps = len(dataloader)
        for k in history.keys():
            history[k].append(ep_metrics[k] / num_steps)
            
        print(f"Epoch {epoch:03d} | "
              f"Tot: {history['loss_total'][-1]:.3f} | "
              f"SU3: {history['loss_gauge_su3'][-1]:.3f} | "
              f"Fermion: {history['loss_action_fermion'][-1]:.3f} | "
              f"Higgs: {history['loss_higgs'][-1]:.3f} | "
              f"VEV: {history['vev'][-1]:.3f}")

        # Periodically generate plots
        if (epoch + 1) % tc.check_interval == 0 or epoch == tc.epochs - 1:
            model.eval()
            with torch.no_grad():
                dummy_batch = next(iter(dataloader))
                z_final = {str(k): v.squeeze(0).to(device) for k, v in dummy_batch['z_dict'].items()}
                u_final = {str(k): v.squeeze(0).to(device) for k, v in dummy_batch['u_dict'].items()}
                e_idx = {ast.literal_eval(k) if isinstance(k, str) else k: v.squeeze(0).to(device) for k, v in dummy_batch['edge_index_dict'].items()}
                e_dir = {ast.literal_eval(k) if isinstance(k, str) else k: v.squeeze(0).to(device) for k, v in dummy_batch.get('edge_dirs_dict', {}).items()}
                
                final_out_dict, _ = model(z_final, e_idx, e_dir, u_final)
                generate_sm_report(history, final_out_dict, u_final, lat.L, lat.dims, epoch, folder="plots")
                model.train()
            


def plot_sm_training_dynamics(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Loss Decomposition
    axes[0].plot(history['loss_total'], color='black', alpha=0.3, label='Total')
    axes[0].plot(history['loss_kinetic_boson'], label='Boson Kinetic')
    axes[0].plot(history['loss_action_fermion'], label='Fermion Action (PF)')
    axes[0].plot(history['loss_higgs'], label='Higgs Potential')
    axes[0].plot(history['loss_gauge_su3'], label='Wilson SU(3)')
    axes[0].plot(history['loss_flow'], label='Flow Entropy', linestyle='--')
    axes[0].set_yscale('log')
    axes[0].set_title("Action Minimization (Energy Landscape)")
    axes[0].legend()
    axes[0].grid(True, which="both", ls="-", alpha=0.2)

    # Right: VEV Evolution
    axes[1].plot(history['vev'], color='#1F77B4', lw=2)
    axes[1].axhline(y=history['vev'][-1], color='r', linestyle='--', alpha=0.5)
    axes[1].set_title(f"Higgs VEV Evolution (Final: {history['vev'][-1]:.4f})")
    axes[1].set_ylabel("$\\langle \\Phi \\rangle$ Magnitude")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_dynamics.png"))
    plt.close()

def plot_matter_spatial_distribution(out_dict, L, dims, save_path):
    # Updated to look for generation 1 names
    species_to_plot = ['quark_left_g1', 'lepton_left_g1', 'higgs']
    fig, axes = plt.subplots(1, len(species_to_plot), figsize=(18, 5))
    
    for i, name in enumerate(species_to_plot):
        if name not in out_dict: continue
        
        phi = out_dict[name].detach().cpu()
        mag = torch.norm(phi, dim=-1).mean(dim=1) 
        
        # 4D Slicing Fix: Reshape to (L, L, L, L) and take a 2D cross-section at Z=L/2, T=L/2
        mag_nd = mag.view(*([L] * dims))
        if dims == 4:
            mag_map = mag_nd[:, :, L//2, L//2].numpy()
        else:
            mag_map = mag_nd.numpy() # Fallback for 2D
        
        im = axes[i].imshow(mag_map, cmap='magma', origin='lower')
        axes[i].set_title(f"Density: {name.replace('_', ' ').title()}")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
    plt.suptitle("Lattice Matter Field Configurations (2D Slice of 4D Grid)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "matter_spatial.png"))
    plt.close()

def plot_gauge_topology(u_dict, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'su3': '#D62728', 'su2': '#1F77B4', 'u1': '#2CA02C'}
    
    for i, (group, u) in enumerate(u_dict.items()):
        u_cpu = u.detach().cpu()
        n_dim = u_cpu.shape[-1]
        traces = torch.real(torch.einsum('...ii->...', u_cpu)).numpy() / n_dim
        
        axes[i].hist(traces, bins=60, color=colors[group], alpha=0.7, edgecolor='black')
        axes[i].set_title(f"{group.upper()} Link Trace Distribution")
        axes[i].set_xlabel("Re[Tr(U)] / N")
        axes[i].set_xlim([-1.1, 1.1])
        axes[i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "gauge_topology.png"))
    plt.close()

def plot_higgs_phase_map(out_dict, L, dims, save_path):
    if 'higgs' not in out_dict: return
    
    phi = out_dict['higgs'].detach().cpu()
    phase = torch.angle(phi[:, 0, 0])
    
    # 4D Slicing Fix
    phase_nd = phase.view(*([L] * dims))
    if dims == 4:
        phase_map = phase_nd[:, :, L//2, L//2].numpy()
    else:
        phase_map = phase_nd.numpy()
        
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(phase_map, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title("Higgs Field Phase Map (Symmetry Angle)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels(['$-\\pi$', '0', '$\\pi$'])
    
    plt.savefig(os.path.join(save_path, "higgs_phase.png"))
    plt.close()

def generate_sm_report(history, out_dict, u_dict, L, dims, epoch, folder="plots"):
    if not os.path.exists(folder): os.makedirs(folder)
    
    plot_sm_training_dynamics(history, folder)
    plot_matter_spatial_distribution(out_dict, L, dims, folder)
    plot_gauge_topology(u_dict, folder)
    plot_higgs_phase_map(out_dict, L, dims, folder)
    
    print(f"--- Dashboard Updated at Epoch {epoch} ---")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    master_config = SMConfig.from_yaml(args.config)
    main(master_config)