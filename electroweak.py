import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import MessagePassing

from tqdm import tqdm
from torch_geometric.utils import to_networkx
import torch_geometric
import networkx as nx

# Visualizing a small slice (e.g., a 4x4 lattice)

def create_lattice(shape):
    dims = len(shape)
    num_nodes = np.prod(shape)
    grid = np.arange(num_nodes).reshape(shape)
    
    srcs, dsts, dirs, fwd = [], [], [], []

    # 1. Build the raw lists
    for d in range(dims):
        for is_fwd in [True, False]:
            shift = -1 if is_fwd else 1
            neighbor_grid = np.roll(grid, shift, axis=d)
            
            s_list = grid.flatten()
            d_list = neighbor_grid.flatten()
            
            for i in range(num_nodes):
                srcs.append(s_list[i])
                dsts.append(d_list[i])
                dirs.append(d)
                fwd.append(is_fwd)

    # 2. Convert to tensors
    srcs_t = torch.tensor(srcs, dtype=torch.long)
    dsts_t = torch.tensor(dsts, dtype=torch.long)
    dirs_t = torch.tensor(dirs, dtype=torch.long)
    fwd_t = torch.tensor(fwd, dtype=torch.bool)
    
    # 3. THE OPTIMIZATION: Sort edges by Direction, then Forward/Backward
    # This guarantees contiguous memory chunks for the GPU
    sort_keys = dirs_t * 2 + (~fwd_t).long()
    # stable=True keeps nodes in order inside each (dir, fwd) block; the
    # e = d*2V + node indexing in find_rectangular_loops depends on it
    sorted_idx = torch.argsort(sort_keys, stable=True)
    
    srcs_sorted = srcs_t[sorted_idx]
    dsts_sorted = dsts_t[sorted_idx]
    edge_dirs = dirs_t[sorted_idx]
    is_fwd = fwd_t[sorted_idx]
    
    edge_index = torch.stack([srcs_sorted, dsts_sorted], dim=0)
    
    # 4. Rebuild the Partner Map using the sorted edges
    # We use a dictionary to find the reverse edge (dst -> src)
    edge_dict = {(s.item(), d.item()): i for i, (s, d) in enumerate(zip(srcs_sorted, dsts_sorted))}
    partner_map = torch.tensor([edge_dict[(d.item(), s.item())] for s, d in zip(srcs_sorted, dsts_sorted)], dtype=torch.long)
    
    return edge_index, edge_dirs, is_fwd, partner_map


def find_rectangular_loops(lattice_shape, edge_index, R=1, T=1):
    """
    Vectorized finder for R x T rectangular loops in an N-D periodic lattice.
    Returns: A tensor of shape [Num_Loops, 2*R + 2*T] containing the ordered edge indices.
    """
    dims = len(lattice_shape)
    V = np.prod(lattice_shape)
    device = edge_index.device # Detect where the lattice lives
    all_loops = []
    
    # Base starting nodes [0, 1, ..., V-1] on the correct device
    base_nodes = torch.arange(V, dtype=torch.long, device=device)
    
    for mu in range(dims):
        for nu in range(mu + 1, dims):
            orientations = [(mu, nu, R, T)]
            if R != T:
                orientations.append((mu, nu, T, R))
                
            for dir1, dir2, len1, len2 in orientations:
                current_nodes = base_nodes.clone()
                loop_edges = []
                
                # Walk the rectangle
                for d, length in [(dir1, len1), (dir2, len2)]:
                    for _ in range(length):
                        e = (d * 2 * V) + current_nodes
                        loop_edges.append(e)
                        current_nodes = edge_index[1, e]
                        
                # Walk back (-dir)
                for d, length in [(dir1, len1), (dir2, len2)]:
                    for _ in range(length):
                        e = (d * 2 * V) + V + current_nodes
                        loop_edges.append(e)
                        current_nodes = edge_index[1, e]
                    
                plane_loops = torch.stack(loop_edges, dim=1)
                all_loops.append(plane_loops)

    return torch.cat(all_loops, dim=0)


def get_gate(raw_u, partner_map, is_fwd, is_su=True):
    """
    Projects raw tensors into Unitary (U(N)) or Special Unitary (SU(N)) groups.
    """
    # 1. Ensure the generator is anti-Hermitian (H = -H^dagger)
    h = (raw_u - raw_u.adjoint()).resolve_conj()
    
    # 2. If SU(N), subtract the trace to ensure det(exp(H)) = 1
    if is_su:
        dim = h.shape[-1]
        # Calculate trace for each edge matrix
        trace = torch.einsum('eii->e', h)
        # Identity matrix for the specific dimension
        eye = torch.eye(dim, dtype=torch.complex64, device=h.device)
        # Reshape trace to [Edges, 1, 1] for broadcasting
        h = h - (trace.view(-1, 1, 1) / dim) * eye
    
    # 3. Exponential map: H (Lie Algebra) -> U (Lie Group)
    u_all = torch.matrix_exp(h)
    
    # 4. Enforce U(-mu) = U(mu)^dagger for lattice consistency
    u_bwd_calculated = u_all[partner_map].adjoint().resolve_conj()
    u_consistent = torch.where(is_fwd.view(-1, 1, 1), u_all, u_bwd_calculated)
    
    return u_consistent


class WilsonAction(nn.Module):
    def __init__(self, plaq_idx, group_dim, beta):
        super().__init__()
        self.register_buffer('plaq_idx', plaq_idx)
        self.group_dim = group_dim
        self.beta = beta

    def forward(self, u_gates):
        """
        u_gates: [Total_Edges, dim, dim]
        """
        # Get all edges for all plaquettes at once
        # Shape: [Num_Plaquettes, 4, dim, dim]
        u_p = u_gates[self.plaq_idx]
        
        # Compose in reverse path order: U_e psi transports column vectors, so the
        # closed-loop transport is U_e4 U_e3 U_e2 U_e1 and its trace telescopes
        # under a gauge transformation U'_e = G_dst U_e G_src^dag
        loop = u_p[:, 0]
        for i in range(1, 4):
            loop = torch.matmul(u_p[:, i], loop)
        tr = torch.einsum('pii -> p', loop)
        
        # 4. Final Scalar calculation
        # S = beta * sum(1 - Re(Tr(U))/Nc)
        nc = float(self.group_dim)
        action_per_plaq = 1.0 - (tr.real / nc)
        
        return self.beta * torch.sum(action_per_plaq)


class ElectroweakHiggsAction(nn.Module):
    """Calculates the Higgs Potential and Kinematics."""
    def __init__(self, mu_sq=1.0, lam=0.5):
        super().__init__()
        self.mu_sq = mu_sq
        self.lam = lam

    def forward(self, phi, edge_index, is_fwd, u_su2, u_u1):
        # 1. Symmetry Breaking Potential: V = -mu^2|phi|^2 + lambda|phi|^4
        phi_sq = torch.sum(phi.abs()**2, dim=-1)
        v_pot = torch.sum(-self.mu_sq * phi_sq + self.lam * (phi_sq**2))
        
        # 2. Covariant Kinetic Energy
        src, dst = edge_index[0, is_fwd], edge_index[1, is_fwd]
        u_total = u_su2[is_fwd] * u_u1[is_fwd]
        phi_transported = torch.einsum('eij, ej -> ei', u_total, phi[src])
        
        diff = phi[dst] - phi_transported
        kinetic = torch.sum(diff.abs()**2)
        
        return kinetic + v_pot
    

class ElectroweakSimulator(nn.Module):
    def __init__(self, lattice_shape=(8, 8, 8), mu_sq=1.0, lam=0.5, beta_su2=4.0, beta_u1=5.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Geometry
        edge_idx, dirs, fwd, p_map = create_lattice(lattice_shape)
        self.register_buffer('edge_index', edge_idx)
        self.register_buffer('is_fwd', fwd)
        self.register_buffer('partner_map', p_map)
        plaq_idx = find_rectangular_loops(lattice_shape, edge_idx)
        
        # Fields
        num_nodes, num_edges = np.prod(lattice_shape), edge_idx.size(1)
        self.phi = nn.Parameter(torch.randn(num_nodes, 2, dtype=torch.complex64) * 0.1)
        self.u_su2 = nn.Parameter(torch.zeros(num_edges, 2, 2, dtype=torch.complex64))
        self.u_u1 = nn.Parameter(torch.zeros(num_edges, 1, 1, dtype=torch.complex64))
        
        # Physics Calculators
        self.w_su2 = WilsonAction(plaq_idx, 2, beta_su2)
        self.w_u1 = WilsonAction(plaq_idx, 1, beta_u1)
        self.h_calc = ElectroweakHiggsAction(mu_sq=mu_sq, lam=lam)
        self.to(self.device)

    def cool_vacuum(self, steps=500, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        history = {'total': [], 'vev': [], 'gauge': [], 'higgs': []}
        
        for step in tqdm(range(steps)):
            optimizer.zero_grad()
            
            # Project to Manifolds
            g_su2 = get_gate(self.u_su2, self.partner_map, self.is_fwd, is_su=True)
            g_u1 = get_gate(self.u_u1, self.partner_map, self.is_fwd, is_su=False)
            
            # Calculate Actions
            s_gauge = self.w_su2(g_su2) + self.w_u1(g_u1)
            s_higgs = self.h_calc(self.phi, self.edge_index, self.is_fwd, g_su2, g_u1)
            loss = s_gauge + s_higgs
            
            loss.backward()
            
            # Safe Complex Gradients
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is not None and p.grad.is_complex():
                        p.grad = p.grad.resolve_conj()
            
            optimizer.step()
            
            # Track Observables
            vev = torch.sqrt(torch.mean(torch.sum(self.phi.abs()**2, dim=-1))).item()
            history['total'].append(loss.item())
            history['vev'].append(vev)
            history['gauge'].append(s_gauge.item())
            history['higgs'].append(s_higgs.item())
            
        return history
    


def measure_electroweak_masses(phi_vac, u_su2_vac, u_u1_vac, edge_index, is_fwd, higgs_calc, g_su2=1.0, g_u1=0.5, eps=1e-3):
    print("\n--- Measuring Electroweak Boson Masses ---")
    device = phi_vac.device
    num_edges = u_su2_vac.shape[0]

    # 1. Calculate Theoretical Weinberg Angle
    # In the Standard Model, tan(theta_W) = g' / g (here g_u1 / g_su2)
    theta_w_theory = np.arctan(g_u1 / g_su2)
    print(f"Theoretical Weinberg Angle (theta_W): {np.degrees(theta_w_theory):.2f}°")

    # 2. Define the Basis Generators
    tau_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    tau_3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    I_su2 = torch.eye(2, dtype=torch.complex64, device=device)

    # Physical Excitations (Corrected for Unitary Gauge)
    # W Boson: Fluctuations in the 'off-diagonal' directions
    gen_W = tau_1 
    
    # Photon (A): The symmetry that remains unbroken. 
    # Must result in 0 mass if phi is aligned to [0, v]
    gen_A = tau_3 + I_su2 

    # Z Boson: The broken neutral combination
    # Orthogonal to the photon in the g/g' space
    gen_Z = (g_su2**2 * tau_3 - g_u1**2 * I_su2) / (g_su2**2 + g_u1**2)**0.5


    u_su2_pure = I_su2.expand(num_edges, -1, -1)
    u_u1_pure  = torch.ones((num_edges, 1, 1), dtype=torch.complex64, device=device)

    masses = {}

    for name, generator in [("W Boson", gen_W), ("Z Boson", gen_Z), ("Photon", gen_A)]:
        # 1. Calculate Base Energy (using identity links for everyone)
        with torch.no_grad():
            s_base = higgs_calc(phi_vac, edge_index, is_fwd, u_su2_pure, u_u1_pure).item()

        # 2. Create the Perturbation Matrix (2x2)
        u_eps_mat = torch.matrix_exp(1j * eps * generator)
        
        # 3. CRITICAL: Expand the 2x2 matrix to [num_edges, 2, 2] 
        # so it matches the expected input shape for the GNN layers
        u_eps_all = u_eps_mat.expand(num_edges, -1, -1)
        
        # 4. Calculate Perturbed Energy
        with torch.no_grad():
            # Pass the expanded u_eps_all instead of the raw u_eps matrix
            s_new = higgs_calc(phi_vac, edge_index, is_fwd, u_eps_all, u_u1_pure).item()
                
        # 5. Extract the mass
        # The kinetic action only sums forward edges, so normalize per forward edge
        num_fwd_edges = int(is_fwd.sum().item())
        delta_s_per_edge = (s_new - s_base) / num_fwd_edges
        mass = np.sqrt(max(0, 2 * delta_s_per_edge) / (eps**2))
        masses[name] = mass
        
        print(f"{name:8} Mass: {mass:.4f}")

    # 5. Calculate derived physical parameters
    mw = masses["W Boson"]
    mz = masses["Z Boson"]
    ma = masses["Photon"]
    
    # Rho Parameter: Should be exactly 1 at tree-level for a single Higgs doublet
    # rho = M_W^2 / (M_Z^2 * cos^2(theta_W))
    if mz > 1e-6: # Prevent division by zero if Z is massless (unbroken phase)
        rho_measured = (mw**2) / ((mz**2) * (np.cos(theta_w_theory)**2))
    else:
        rho_measured = 0.0
        
    # Measured Weinberg Angle from the mass ratio: sin^2(theta_w) = 1 - (Mw/Mz)^2
    if mz > 1e-6 and mw <= mz:
        sin2_theta_w_measured = 1.0 - (mw**2 / mz**2)
        theta_w_measured = np.arcsin(np.sqrt(sin2_theta_w_measured))
    else:
        theta_w_measured = 0.0

    print("\n--- Physical Parameter Estimates ---")
    print(f"Measured Rho (ρ) parameter:     {rho_measured:.6f} (Theory: 1.0)")
    print(f"Measured Weinberg Angle:        {np.degrees(theta_w_measured):.2f}° (Theory: {np.degrees(theta_w_theory):.2f}°)")
    print(f"Mass Ratio (Mw/Mz):             {mw/mz if mz > 1e-6 else 0:.4f} (Theory: {np.cos(theta_w_theory):.4f})")
    
    # Physics sanity check
    if abs(rho_measured - 1.0) < 0.05 and ma < 1e-2:
        print("✅ SUCCESS: Custodial symmetry is preserved (ρ ≈ 1) and Photon is massless.")
    else:
        print("⚠️ WARNING: Significant deviation in the ρ parameter or Photon mass.")

    return {
        "masses": masses,
        "theta_w_theory": theta_w_theory,
        "theta_w_measured": theta_w_measured,
        "rho": rho_measured
    }


def plot_action_landscape(history, smooth=False):
    """
    Plots a dual-axis representation of the Standard Model vacuum energy
    and the Higgs VEV, with optional smoothing for Langevin noise.
    """
    epochs = np.arange(len(history['total']))
    
    # Helper for smoothing Langevin noise
    def get_smooth(data, window=10):
        if not smooth or len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='same')

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- AXIS 1: Action/Energy (Log Scale) ---
    ax1.set_yscale('log')
    ax1.set_xlabel("Langevin Sampling Steps", fontsize=12)
    ax1.set_ylabel("Action Energy ($S$)", fontsize=12)
    
    # Plotting sector energy contributions
    ax1.plot(epochs, history['total'], 'k-', alpha=0.3, linewidth=1) # Raw total
    ax1.plot(epochs, get_smooth(history['total']), 'k-', linewidth=2, label='Total Action ($S_{tot}$)')
    
    ax1.plot(epochs, get_smooth(history['su3']), 'r--', label='SU(3) Strong', alpha=0.8)
    ax1.plot(epochs, get_smooth(history['su2']), 'g--', label='SU(2) Weak', alpha=0.8)
    ax1.plot(epochs, get_smooth(history['u1']), 'b--', label='U(1) Hypercharge', alpha=0.8)
    ax1.plot(epochs, get_smooth(history['higgs']), 'm-.', label='Higgs Sector', alpha=0.8)
    ax1.plot(epochs, get_smooth(history['fermion']), 'c:', label='Fermion Action', linewidth=2)

    # --- AXIS 2: Higgs VEV (Linear Scale) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("Higgs VEV $\\langle \phi \\rangle$", color='darkorange', fontsize=12)
    ax2.plot(epochs, history['vev'], color='orange', alpha=0.2) # Raw VEV
    ax2.plot(epochs, get_smooth(history['vev']), color='darkorange', linewidth=2.5, label='VEV (Order Parameter)')
    ax2.tick_params(axis='y', labelcolor='darkorange')

    # Formatting
    plt.title("Heterogeneous Standard Model: Vacuum Equilibrium & Phase Transition", fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.15)
    
    # Merging legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig("plots/sm_action_landscape.png", dpi=150)


def plot_cooling_landscape(history):
    """Visualizes the energy minimization during the GNN training loop."""
    epochs = range(len(history['total']))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Cooling Steps (Optimizer Epochs)')
    ax1.set_ylabel('Euclidean Action $S$', color=color)
    ax1.plot(epochs, history['total'], 'k-', label='Total Action', linewidth=2)
    ax1.plot(epochs, history['gauge'], 'r--', label='Gauge Action $S_W$', alpha=0.7)
    ax1.plot(epochs, history['higgs'], 'm-.', label='Higgs Action $S_H$', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Higgs VEV $\\langle |\\phi| \\rangle$', color=color)
    ax2.plot(epochs, history['vev'], color=color, linewidth=2, label='Measured VEV')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("GNN Vacuum Search: Action Minimization and VEV Condensation")
    fig.tight_layout()
    plt.savefig("action_landscape.png", dpi=300)
    plt.show()

def plot_phase_transition():
    """Sweeps the mass parameter to locate the second-order transition."""
    mu_sq_vals = np.linspace(-1.5, 3.0, 20)
    vevs = []
    
    print("Executing Phase Transition Sweep...")
    for mu in mu_sq_vals:
        sim = ElectroweakSimulator(lattice_shape=(8, 8), mu_sq=mu)
        hist = sim.cool_vacuum(steps=300, lr=0.02)
        # Record the settled VEV at the end of cooling
        vevs.append(hist['vev'][-1])
        print(f"mu^2: {mu: .2f} | Final VEV: {vevs[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(mu_sq_vals, vevs, 'bo-', linewidth=2, markersize=6)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Critical Point $\\mu^2 = 0$')
    
    # Theoretical curve for broken phase: v = sqrt(mu^2 / (2 * lambda))
    theoretical_v = [np.sqrt(max(0, m) / (2 * 0.5)) for m in mu_sq_vals]
    plt.plot(mu_sq_vals, theoretical_v, 'r--', label='Theoretical Prediction')
    
    plt.title('Electroweak Phase Transition via GNN Optimization')
    plt.xlabel('Mass Parameter $\\mu^2$')
    plt.ylabel('Vacuum Expectation Value $\\langle |\\phi| \\rangle$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("phase_transition.png", dpi=300)
    plt.show()


@torch.no_grad()
def align_to_unitary_gauge(phi, edge_index, u_su2, u_u1):
    """
    Rotates the local Higgs field at every node to point to [0, v].
    This 'straightens' the vacuum so we can measure masses using fixed generators.
    """
    num_nodes = phi.shape[0]
    device = phi.device
    
    # 1. Calculate the rotation G for every node
    # We want G @ phi = [0, |phi|]
    g_mats = torch.zeros((num_nodes, 2, 2), dtype=torch.complex64, device=device)
    phi_aligned = torch.zeros_like(phi)
    
    for n in range(num_nodes):
        p = phi[n]
        norm = torch.norm(p)
        if norm < 1e-8:
            g_mats[n] = torch.eye(2, dtype=torch.complex64, device=device)
            continue
            
        # SU(2) rotation matrix to align p with [0, norm]
        # For p = [a, b], G = [[b, -a], [conj(a), conj(b)]] / norm gives G @ p = [0, norm]
        a, b = p[0], p[1]
        g = torch.tensor([[b, -a],
                          [a.conj(), b.conj()]], dtype=torch.complex64, device=device) / norm
        g_mats[n] = g
        phi_aligned[n] = torch.tensor([0, norm], dtype=torch.complex64, device=device)

    # 2. Transform the Gauge Links to maintain invariance
    # U'_{ij} = G_i * U_{ij} * G_j^H
    # This ensures the physics (the action) doesn't change, only the 'view'
    g_src = g_mats[edge_index[0]] # [Edges, 2, 2]
    g_tgt = g_mats[edge_index[1]] # [Edges, 2, 2]
    
    u_su2_aligned = torch.einsum('eij, ejk, ekl -> eil', g_tgt, u_su2, g_src.adjoint())
    
    # U(1) is a central charge; it doesn't change under SU(2) rotations
    return phi_aligned, u_su2_aligned, u_u1


if __name__ == "__main__":
    # --- BLOCK A: CONFIGURATION ---
    L_SHAPE = (8, 8, 8)
    B_SU2 = 4.0
    B_U1 = 5.0
    
    sim = ElectroweakSimulator(
        lattice_shape=L_SHAPE, 
        mu_sq=2.0, 
        lam=0.5, 
        beta_su2=B_SU2, 
        beta_u1=B_U1
    )

    # --- BLOCK B: VACUUM MINIMIZATION ---
    print("Beginning Vacuum Cooling...")
    history = sim.cool_vacuum(steps=600, lr=0.01)
    
    # Optional: Plot the cooling landscape here
    # plot_action_landscape(history, smooth=True, thermalization_step=100)


    # --- BLOCK C: PHYSICS DIAGNOSTICS ---
    # Extract final states
    phi_final = sim.phi.detach()
    
    with torch.no_grad():
        u_su2_final = get_gate(
            sim.u_su2, sim.partner_map, sim.is_fwd, is_su=True
        ).detach()
        
        u_u1_final = get_gate(
            sim.u_u1, sim.partner_map, sim.is_fwd, is_su=False
        ).detach()


    phi_aligned, u_su2_aligned, u_u1_aligned = align_to_unitary_gauge(
        sim.phi.detach(), sim.edge_index, u_su2_final, u_u1_final
    )

    g = np.sqrt(4.0 / B_SU2)
    gp = np.sqrt(1.0 / B_U1)

    # 2. Measure with the aligned fields
    stats = measure_electroweak_masses(
        phi_aligned, 
        u_su2_aligned, 
        u_u1_aligned, 
        sim.edge_index, 
        sim.is_fwd, 
        sim.h_calc, 
        g_su2=g, g_u1=gp
    )

    # --- BLOCK D: REPORTING ---
    print(f"\nFinal VEV: {history['vev'][-1]:.4f}")
    print(f"Final Rho: {stats['rho']:.6f}")