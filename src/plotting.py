from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _smooth(data: list[float], window: int = 10) -> np.ndarray | list[float]:
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='same')


def plot_sm_action_landscape(history: dict, save_path: str, smooth: bool = False) -> None:
    """Dual-axis view of the heterogeneous SM sector actions and the Higgs VEV."""
    epochs = np.arange(len(history['total']))
    sm = _smooth if smooth else (lambda d: d)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_yscale('log')
    ax1.set_xlabel("Langevin Sampling Steps", fontsize=12)
    ax1.set_ylabel("Action Energy ($S$)", fontsize=12)

    ax1.plot(epochs, history['total'], 'k-', alpha=0.3, linewidth=1)
    ax1.plot(epochs, sm(history['total']), 'k-', linewidth=2, label='Total Action ($S_{tot}$)')
    ax1.plot(epochs, sm(history['su3']), 'r--', label='SU(3) Strong', alpha=0.8)
    ax1.plot(epochs, sm(history['su2']), 'g--', label='SU(2) Weak', alpha=0.8)
    ax1.plot(epochs, sm(history['u1']), 'b--', label='U(1) Hypercharge', alpha=0.8)
    ax1.plot(epochs, sm(history['higgs']), 'm-.', label='Higgs Sector', alpha=0.8)
    ax1.plot(epochs, sm(history['fermion']), 'c:', label='Fermion Action', linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Higgs VEV $\langle \phi \rangle$", color='darkorange', fontsize=12)
    ax2.plot(epochs, history['vev'], color='orange', alpha=0.2)
    ax2.plot(epochs, sm(history['vev']), color='darkorange', linewidth=2.5,
             label='VEV (Order Parameter)')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    plt.title("Heterogeneous Standard Model: Vacuum Equilibrium & Phase Transition")
    ax1.grid(True, which="both", ls="-", alpha=0.15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_qcd_action_landscape(history: dict, save_path: str) -> None:
    """Per-sector action decomposition for the standalone vacuum finders."""
    epochs = range(len(history['total']))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['total'], 'k-', linewidth=2, label='Total Action ($S_{tot}$)')
    plt.plot(epochs, history['gauge_su3'], 'r--', alpha=0.8, label='SU(3) Strong')
    plt.plot(epochs, history['gauge_su2'], 'g--', alpha=0.8, label='SU(2) Weak')
    plt.plot(epochs, history['gauge_u1'], 'b--', alpha=0.8, label='U(1) Hypercharge')
    plt.plot(epochs, history['higgs'], 'm-.', alpha=0.8, label=r'Higgs ($V(\phi)$)')
    plt.plot(epochs, history['fermion'], 'c:', linewidth=2,
             label=r'Fermion ($\overline{\psi} D \psi$)')

    plt.title("Vacuum Loss Landscape (Action Contributions over Time)")
    plt.xlabel("Cooling Steps")
    plt.ylabel("Action Energy")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cooling_landscape(history: dict, save_path: str) -> None:
    """Action minimization and VEV condensation for the electroweak simulator."""
    epochs = range(len(history['total']))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Cooling Steps (Optimizer Epochs)')
    ax1.set_ylabel('Euclidean Action $S$', color='tab:red')
    ax1.plot(epochs, history['total'], 'k-', label='Total Action', linewidth=2)
    ax1.plot(epochs, history['gauge'], 'r--', label='Gauge Action $S_W$', alpha=0.7)
    ax1.plot(epochs, history['higgs'], 'm-.', label='Higgs Action $S_H$', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Higgs VEV $\langle |\phi| \rangle$', color='tab:blue')
    ax2.plot(epochs, history['vev'], color='tab:blue', linewidth=2, label='Measured VEV')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Vacuum Search: Action Minimization and VEV Condensation")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sm_training_dynamics(history: dict, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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

    axes[1].plot(history['vev'], color='#1F77B4', lw=2)
    axes[1].axhline(y=history['vev'][-1], color='r', linestyle='--', alpha=0.5)
    axes[1].set_title(f"Higgs VEV Evolution (Final: {history['vev'][-1]:.4f})")
    axes[1].set_ylabel(r"$\langle \Phi \rangle$ Magnitude")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "training_dynamics.png")
    plt.close(fig)


def plot_matter_spatial_distribution(out_dict: dict, L: int, dims: int, save_dir: str) -> None:
    species_to_plot = ['quark_left_g1', 'lepton_left_g1', 'higgs']
    fig, axes = plt.subplots(1, len(species_to_plot), figsize=(18, 5))

    for i, name in enumerate(species_to_plot):
        if name not in out_dict:
            continue
        phi = out_dict[name].detach().cpu()
        mag = torch.norm(phi, dim=-1).mean(dim=1)

        mag_nd = mag.view(*([L] * dims))
        # 2D cross-section of the 4D grid at z = t = L/2
        mag_map = mag_nd[:, :, L // 2, L // 2].numpy() if dims == 4 else mag_nd.numpy()

        im = axes[i].imshow(mag_map, cmap='magma', origin='lower')
        axes[i].set_title(f"Density: {name.replace('_', ' ').title()}")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle("Lattice Matter Field Configurations (2D Slice)", fontsize=16)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "matter_spatial.png")
    plt.close(fig)


def plot_gauge_topology(u_dict: dict, save_dir: str) -> None:
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
    plt.savefig(Path(save_dir) / "gauge_topology.png")
    plt.close(fig)


def plot_higgs_phase_map(out_dict: dict, L: int, dims: int, save_dir: str) -> None:
    if 'higgs' not in out_dict:
        return

    phi = out_dict['higgs'].detach().cpu()
    phase = torch.angle(phi[:, 0, 0])

    phase_nd = phase.view(*([L] * dims))
    phase_map = phase_nd[:, :, L // 2, L // 2].numpy() if dims == 4 else phase_nd.numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(phase_map, cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
    ax.set_title("Higgs Field Phase Map (Symmetry Angle)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

    plt.savefig(Path(save_dir) / "higgs_phase.png")
    plt.close(fig)


def generate_sm_report(history: dict, out_dict: dict, u_dict: dict,
                       L: int, dims: int, epoch: int, folder: str = "plots") -> None:
    Path(folder).mkdir(exist_ok=True)
    plot_sm_training_dynamics(history, folder)
    plot_matter_spatial_distribution(out_dict, L, dims, folder)
    plot_gauge_topology(u_dict, folder)
    plot_higgs_phase_map(out_dict, L, dims, folder)
    print(f"--- Dashboard Updated at Epoch {epoch} ---")
