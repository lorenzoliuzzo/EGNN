import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.data import HeteroData


def create_lattice(shape, a=1):
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
    sorted_idx = torch.argsort(sort_keys)
    
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


def get_gammas():
    zeros = torch.zeros(2, 2, dtype=torch.complex64)
    eye = torch.eye(2, dtype=torch.complex64)
    px = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    py = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    pz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    g1 = torch.cat([torch.cat([zeros, 1j*px], dim=1), torch.cat([-1j*px, zeros], dim=1)], dim=0)
    g2 = torch.cat([torch.cat([zeros, 1j*py], dim=1), torch.cat([-1j*py, zeros], dim=1)], dim=0)
    g3 = torch.cat([torch.cat([zeros, 1j*pz], dim=1), torch.cat([-1j*pz, zeros], dim=1)], dim=0)
    g4 = torch.cat([torch.cat([zeros, eye], dim=1), torch.cat([eye, zeros], dim=1)], dim=0)
    g5 = g1 @ g2 @ g3 @ g4

    return torch.stack([g1, g2, g3, g4], dim=0), g5


def get_chiral_projectors(id_spin, g5):
    P_L = 0.5 * (id_spin - g5) 
    P_R = 0.5 * (id_spin + g5) 
    return P_L, P_R


class HiggsPhysicsSimulation(nn.Module):
    def __init__(self, num_nodes, num_edges, gauge_dim=2):
        super().__init__()
        # 1. Higgs Field: A complex doublet (SU(2)) or complex scalar (U(1))
        # Shape: [Nodes, 2] (treating complex as two real channels or using torch.complex)
        self.phi = nn.Parameter(torch.randn(num_nodes, gauge_dim, dtype=torch.complex64) * 0.1)
        
        # 2. Gauge Links: Raw weights that we project to U(1) or SU(2)
        self.raw_u = nn.Parameter(torch.randn(num_edges, gauge_dim, gauge_dim, dtype=torch.complex64) * 0.1)

    def compute_action(self, edge_index, partner_map, is_fwd, mu_sq, lam, beta):
        # A. Project to Gauge Group (e.g., U(n))
        u = get_gate(self.raw_u, partner_map, is_fwd, is_su=False)
        
        # B. Kinetic Term (D_mu phi)^2
        # Measures how much the Higgs changes across edges
        row, col = edge_index
        phi_j = self.phi[col]
        # Transport phi_j to node i: U_ij * phi_j
        phi_transported = torch.einsum('eij, ej -> ei', u, phi_j)
        kinetic_term = torch.sum(torch.abs(self.phi[row] - phi_transported)**2)
        
        # C. Potential Term: V(phi) = -mu^2 |phi|^2 + lambda |phi|^4
        phi_sq = torch.sum(torch.abs(self.phi)**2, dim=-1)
        potential_term = torch.sum(-mu_sq * phi_sq + lam * (phi_sq**2))
        
        # D. Gauge Action (Wilson Loops/Plaquettes) - Optional for pure SSB 
        # but keeps the gauge field from being totally random.
        # Action = beta * (1 - Re Tr(P))
        
        return kinetic_term + potential_term
    

def run_higgs_simulation(lattice_shape, mu_range):
    # Setup Lattice
    edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
    num_nodes = np.prod(lattice_shape)
    num_edges = edge_index.shape[1]
    
    vevs = []
    masses = []

    for mu_sq in mu_range:
        model = HiggsPhysicsSimulation(num_nodes, num_edges)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Energy Minimization (Finding the Vacuum)
        for _ in range(500):
            optimizer.zero_grad()
            loss = model.compute_action(edge_index, partner_map, is_fwd, 
                                        mu_sq=mu_sq, lam=0.5, beta=2.0)
            loss.backward()
            optimizer.step()
        
        # Measure Results
        with torch.no_grad():
            # 1. Vacuum Expectation Value (VEV)
            current_vev = torch.sqrt(torch.mean(torch.sum(model.phi.abs()**2, dim=-1))).item()
            vevs.append(current_vev)
            
            # 2. Measure "Boson Mass" 
            # (The energy penalty for gauge transport in the vacuum)
            u = get_gate(model.raw_u, partner_map, is_fwd, is_su=False)
            phi_j = model.phi[edge_index[1]]
            phi_transported = torch.einsum('eij, ej -> ei', u, phi_j)
            link_variance = torch.mean(torch.abs(model.phi[edge_index[0]] - phi_transported)**2)
            masses.append(link_variance.item())

    return vevs, masses


mu_vals = np.linspace(-2.0, 4.0, 20)
vevs, masses = run_higgs_simulation((8, 8), mu_vals)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mu_vals, vevs, 'o-')
plt.title("Higgs VEV")
plt.xlabel("mu^2")

plt.subplot(1, 2, 2)
plt.plot(mu_vals, masses, 's-', color='orange')
plt.title("Gauge Boson Mass (Link Energy)")
plt.xlabel("mu^2")
plt.savefig("plots/higgs.png")