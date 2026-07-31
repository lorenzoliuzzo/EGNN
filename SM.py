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


class QuarkGaugeConv(MessagePassing):
    """
    Tensor Product Propagator for Quarks: SU(3) x SU(2)_L x U(1)_Y
    State: [Edges, Flavors, Color, Isospin, Spin]
    """
    def __init__(self, r=1.0, a=1.0):
        super().__init__(aggr='add', node_dim=0)
        self.r, self.a = r, a

    def forward(self, x, edge_index, u_su3, u_su2, u_u1, edge_dirs, is_fwd, gammas, id_spin, P_L, P_R, **kwargs):
        return self.propagate(edge_index, x=x, u_su3=u_su3, u_su2=u_su2, u_u1=u_u1, 
                              edge_dirs=edge_dirs, is_fwd=is_fwd, gammas=gammas, id_spin=id_spin, P_L=P_L, P_R=P_R)

    def message(self, x_j, u_su3, u_su2, u_u1, edge_dirs, is_fwd, gammas, id_spin, P_L, P_R):
        # 1. Chiral Projection (Spin is dim 's' or 'b')
        x_L = torch.einsum('ab, efcib -> efcia', P_L, x_j)
        x_R = torch.einsum('ab, efcib -> efcia', P_R, x_j)

        # 2. Strong Force (Color Transport) applies to both chiralities
        x_L_color = torch.einsum('ecd, efdis -> efcis', u_su3, x_L)
        x_R_color = torch.einsum('ecd, efdis -> efcis', u_su3, x_R)

        # 3. Electroweak Transport
        # Left-Handed feels SU(2) * U(1)
        ew_link_L = u_u1 * u_su2
        x_L_full = torch.einsum('eij, efcjs -> efcis', ew_link_L, x_L_color)
        
        # Right-Handed feels ONLY U(1) (SU(2) Singlet)
        u_u1_view = u_u1.view(-1, 1, 1, 1, 1)
        x_R_full = x_R_color * u_u1_view

        # 4. Recombine and apply Wilson Doubler Penalty
        x_transported = x_L_full + x_R_full
        
        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1)
        P_edge = (self.r / self.a) * id_spin + (1.0 / self.a) * sign * gammas[edge_dirs]
        
        return torch.einsum('eab, efcib -> efcia', P_edge, x_transported)


class LeptonGaugeConv(MessagePassing):
    """
    Tensor Product Propagator for Leptons: SU(2)_L x U(1)_Y (Colorless)
    State: [Edges, Flavors, Isospin, Spin]
    """
    def __init__(self, r=1.0, a=1.0):
        super().__init__(aggr='add', node_dim=0)
        self.r, self.a = r, a

    def forward(self, x, edge_index, u_su2, u_u1, edge_dirs, is_fwd, gammas, id_spin, P_L, P_R, **kwargs):
        return self.propagate(edge_index, x=x, u_su2=u_su2, u_u1=u_u1, 
                              edge_dirs=edge_dirs, is_fwd=is_fwd, gammas=gammas, id_spin=id_spin, P_L=P_L, P_R=P_R)

    def message(self, x_j, u_su2, u_u1, edge_dirs, is_fwd, gammas, id_spin, P_L, P_R):
        x_L = torch.einsum('ab, efib -> efia', P_L, x_j)
        x_R = torch.einsum('ab, efib -> efia', P_R, x_j)

        ew_link_L = u_u1 * u_su2
        x_L_full = torch.einsum('eij, efjs -> efis', ew_link_L, x_L)
        
        u_u1_view = u_u1.view(-1, 1, 1, 1)
        x_R_full = x_R * u_u1_view

        x_transported = x_L_full + x_R_full
        
        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1)
        P_edge = (self.r / self.a) * id_spin + (1.0 / self.a) * sign * gammas[edge_dirs]
        
        return torch.einsum('eab, efib -> efia', P_edge, x_transported)


class HiggsGaugeConv(MessagePassing):
    """
    Kinetic Propagator for the Higgs Boson: SU(2) x U(1)
    State: [Edges, Isospin]
    """
    def __init__(self, a=1.0):
        super().__init__(aggr='add', node_dim=0)
        self.a = a

    def forward(self, x, edge_index, u_su2, u_u1, **kwargs):
        return self.propagate(edge_index, x=x, u_su2=u_su2, u_u1=u_u1)

    def message(self, x_j, u_su2, u_u1):
        u_total = u_u1 * u_su2
        # (1/a) scaling for the finite difference derivative
        return (1.0 / self.a) * torch.einsum('eij, ej -> ei', u_total, x_j)


class StandardModelGNN(nn.Module):
    def __init__(self, num_flavors=3, a=1.0, r=1.0, bare_mass=0.01):
        super().__init__()
        self.a = a
        self.r = r
        self.bare_mass = bare_mass
        
        # 1. Physics Infrastructure
        gammas, g5 = get_gammas()
        id_spin = torch.eye(4, dtype=torch.complex64)
        P_L, P_R = get_chiral_projectors(id_spin, g5)

        self.register_buffer('gammas', gammas)
        self.register_buffer('g5', g5)
        self.register_buffer('id_spin', id_spin)
        self.register_buffer('P_L', P_L)
        self.register_buffer('P_R', P_R)
        
        # 2. Flavor Dynamics
        # Random initialization; in reality, these map to measured particle masses
        self.y_quarks = nn.Parameter(torch.rand(num_flavors, dtype=torch.complex64))
        self.y_leptons = nn.Parameter(torch.rand(num_flavors, dtype=torch.complex64))

        # 3. The Heterogeneous Graph Schema
        self.gauge_interactions = HeteroConv({
            ('quark', 'gauge', 'quark'): QuarkGaugeConv(r, a),
            ('lepton', 'gauge', 'lepton'): LeptonGaugeConv(r, a),
            ('higgs', 'gauge', 'higgs'): HiggsGaugeConv(a)
        }, aggr='sum')


    def forward(self, x_dict, edge_index_dict, u_dict, is_fwd, edge_dirs):
        """
        Executes one full application of the Standard Model Dirac & Kinetic operators.
        """
        # A. Unpack Gauge Links for the HeteroConv
        # We pass kwargs specific to each particle's requirement
        edge_types = list(edge_index_dict.keys())
        
        # Gauge Links
        u_su3_dict = {et: u_dict['su3'] for et in edge_types}
        u_su2_dict = {et: u_dict['su2'] for et in edge_types}
        u_u1_dict  = {et: u_dict['u1']  for et in edge_types}
        
        # Geometry & Symmetry Tensors
        dirs_dict = {et: edge_dirs for et in edge_types}
        fwd_dict  = {et: is_fwd for et in edge_types}
        g_dict    = {et: self.gammas for et in edge_types}
        id_spin_dict = {et: self.id_spin for et in edge_types}
        pl_dict   = {et: self.P_L for et in edge_types}
        pr_dict   = {et: self.P_R for et in edge_types}

        # 2. CALL HETEROCONV
        # When you pass 'u_su3_dict', HeteroConv passes 'u_su3_dict[edge_type]' 
        # to the sub-convolution as the keyword argument 'u_su3'.
        transported_dict = self.gauge_interactions(
            x_dict, 
            edge_index_dict, 
            u_su3_dict=u_su3_dict, 
            u_su2_dict=u_su2_dict, 
            u_u1_dict=u_u1_dict,
            edge_dirs_dict=dirs_dict,
            is_fwd_dict=fwd_dict,
            gammas_dict=g_dict,
            id_spin_dict=id_spin_dict,
            P_L_dict=pl_dict,
            P_R_dict=pr_dict
        )
        
        # hetero_kwargs = {
        #     'quark': {
        #         'u_su3': u_dict['su3'], 'u_su2': u_dict['su2'], 'u_u1': u_dict['u1'],
        #         'edge_dirs': edge_dirs, 'is_fwd': is_fwd, 'gammas': self.gammas
        #     },
        #     'lepton': {
        #         'u_su2': u_dict['su2'], 'u_u1': u_dict['u1'],
        #         'edge_dirs': edge_dirs, 'is_fwd': is_fwd, 'gammas': self.gammas
        #     },
        #     'higgs': {
        #         'u_su2': u_dict['su2'], 'u_u1': u_dict['u1']
        #     }
        # }

        # # B. Propagate fields across the spacetime graph
        # # transported_dict = self.gauge_interactions(x_dict, edge_index_dict, hetero_kwargs)
        # transported_dict = self.gauge_interactions(
        #     x_dict, 
        #     edge_index_dict, 
        #     u_su3=u_dict['su3'], 
        #     u_su2=u_dict['su2'], 
        #     u_u1=u_dict['u1'],
        #     edge_dirs=edge_dirs,
        #     is_fwd=is_fwd,
        #     gammas=self.gammas
        # )


        out_dict = {}
        phi = x_dict['higgs']
        phi_vev = torch.sqrt(torch.sum(phi.abs()**2, dim=-1)) # Shape: [Nodes]
        dim_space = len(torch.unique(edge_dirs))

        # C. Apply Local Terms (Masses & Wilson Restoring Force)
        for p_type, y_couplings in zip(['quark', 'lepton'], [self.y_quarks, self.y_leptons]):
            psi = x_dict[p_type]
            transported_psi = transported_dict[p_type]
            
            # Dynamic Mass generation via Higgs Mechanism
            dynamic_mass = self.bare_mass + torch.einsum('f, n -> nf', y_couplings, phi_vev)
            
            # Broadcast mass and apply Wilson penalty (r / a * dims)
            if p_type == 'quark':
                m_view = dynamic_mass.view(-1, len(y_couplings), 1, 1, 1)
            else:
                m_view = dynamic_mass.view(-1, len(y_couplings), 1, 1)

            local_term = (m_view + (self.r / self.a) * dim_space) * psi
            out_dict[p_type] = local_term - 0.5 * transported_psi

        # D. Higgs local term (Standard discrete derivative formulation)
        # D_mu phi = (phi(x) - U phi(x-mu))/a
        out_dict['higgs'] = (dim_space / self.a) * phi - transported_dict['higgs']

        return out_dict


# --- Helper functions for Dictionary Linear Algebra ---
def dict_dot(dict_a, dict_b):
    """Computes the real inner product across all fermion fields."""
    return sum(torch.sum(dict_a[k].conj() * dict_b[k]).real for k in ['quark', 'lepton'])

def dict_add(dict_a, dict_b, alpha=1.0):
    """Adds two dictionaries of tensors: A + alpha * B"""
    return {k: dict_a[k] + alpha * dict_b[k] for k in ['quark', 'lepton']}


@torch.no_grad()
def hetero_conjugate_gradient(gnn_model, pf_dict, higgs_phi, edge_index_dict, u_dict, is_fwd, edge_dirs, max_iter=200, tol=1e-6):
    """
    pf_dict: Dictionary containing the pseudofermion noise {'quark': tensor, 'lepton': tensor}
    higgs_phi: The actual Higgs field [Nodes, 2], treated as a fixed background field here.
    """
    # Initialize solution x as zero tensors matching the shapes of pf_dict
    x = {k: torch.zeros_like(v) for k, v in pf_dict.items()}
    
    def apply_m_dag_m(vec_dict):
        # 1. Package the current fermion guess with the fixed Higgs field
        input_dict = {'quark': vec_dict['quark'], 'lepton': vec_dict['lepton'], 'higgs': higgs_phi}
        
        # 2. Forward pass (D)
        d_out = gnn_model(input_dict, edge_index_dict, u_dict, is_fwd, edge_dirs)
        
        # 3. Multiply by Gamma_5 for the adjoint trick (Gamma_5 is hermitian and unitary)
        # Spin is the last dimension 's'
        g5 = gnn_model.gammas[0] @ gnn_model.gammas[1] @ gnn_model.gammas[2] @ gnn_model.gammas[3]
        g5_d = {
            'quark': torch.einsum('ss, ...s -> ...s', g5, d_out['quark']),
            'lepton': torch.einsum('ss, ...s -> ...s', g5, d_out['lepton']),
            'higgs': higgs_phi # Keep Higgs fixed
        }
        
        # 4. Second Forward pass (D)
        d_dag_partial = gnn_model(g5_d, edge_index_dict, u_dict, is_fwd, edge_dirs)
        
        # 5. Final Gamma_5 multiplication
        return {
            'quark': torch.einsum('ss, ...s -> ...s', g5, d_dag_partial['quark']),
            'lepton': torch.einsum('ss, ...s -> ...s', g5, d_dag_partial['lepton'])
        }

    # Initial residual: r = pf_dict - A*x (which is just pf_dict since x=0)
    r = {k: v.clone() for k, v in pf_dict.items()}
    p = {k: v.clone() for k, v in pf_dict.items()}
    rsold = dict_dot(r, r)

    for i in range(max_iter):
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
    def __init__(self, gnn_model):
        super().__init__()
        # We pass the full GNN into the action so it can use it for the Dirac operator
        self.gnn = gnn_model

    def forward(self, pf_dict, higgs_phi, edge_index_dict, u_dict, is_fwd, edge_dirs):
        # 1. Solve (D^dag D) x = pf_dict using Hetero CG
        x_solved = hetero_conjugate_gradient(
            self.gnn, pf_dict, higgs_phi, edge_index_dict, u_dict, is_fwd, edge_dirs
        )
        
        # 2. Actual Action: S = pf_dict^dag * x_solved
        actual_action = dict_dot(pf_dict, x_solved)
        
        # 3. Surrogate Loss for AutoGrad
        # Pass the solved fields back through the GNN to build the computational graph
        input_dict = {'quark': x_solved['quark'], 'lepton': x_solved['lepton'], 'higgs': higgs_phi}
        d_x = self.gnn(input_dict, edge_index_dict, u_dict, is_fwd, edge_dirs)
        
        # The surrogate loss only cares about the fermions, not the Higgs kinetic output
        dx_sq_quark = torch.sum(d_x['quark'].abs()**2)
        dx_sq_lepton = torch.sum(d_x['lepton'].abs()**2)
        surrogate_loss = -(dx_sq_quark + dx_sq_lepton)
                
        # Detach trick: Value of actual_action, Gradients of surrogate_loss
        return surrogate_loss - surrogate_loss.detach() + actual_action


class StandardModelUniverse(nn.Module):
    def __init__(self, gnn_model, plaq_idx, beta_su3=5.5, beta_su2=2.0, beta_u1=1.0, higgs_v=1.0, higgs_lam=0.1):
        super().__init__()
        self.gnn = gnn_model
        
        # Gauge Actions (The Direct Sum part of the SM)
        self.action_su3 = WilsonAction(plaq_idx, group_dim=3, beta=beta_su3)
        self.action_su2 = WilsonAction(plaq_idx, group_dim=2, beta=beta_su2)
        self.action_u1  = WilsonAction(plaq_idx, group_dim=1, beta=beta_u1)
        
        # Bosonic Matter
        self.action_higgs = HiggsAction(v=higgs_v, lam=higgs_lam)
        
        # Fermionic Matter
        self.action_fermions = HeteroPseudofermionAction(self.gnn)

    def forward(self, hetero_data, u_dict, pf_dict, is_fwd, edge_dirs):
        """
        Calculates the total Action (Energy) of the specific lattice configuration.
        """
        # 1. Pure Gauge Energy
        s_su3 = self.action_su3(u_dict['su3'])
        s_su2 = self.action_su2(u_dict['su2'])
        s_u1  = self.action_u1(u_dict['u1'])
        
        # 2. Higgs Energy (Potential + Kinetic)
        # We extract the SU(2) / U(1) sub-graph edges for the Higgs transport
        higgs_edge_index = hetero_data['higgs', 'gauge', 'higgs'].edge_index
        s_higgs = self.action_higgs(
            hetero_data['higgs'].x, 
            higgs_edge_index, 
            is_fwd, 
            u_dict['su2'], 
            u_dict['u1']
        )
        
        # 3. Fermion Energy (Quarks & Leptons via Pseudofermions)
        s_fermions = self.action_fermions(
            pf_dict, 
            hetero_data['higgs'].x, 
            hetero_data.edge_index_dict, 
            u_dict, 
            is_fwd, 
            edge_dirs
        )
        
        # Total Action of the Universe
        total_action = s_su3 + s_su2 + s_u1 + s_higgs + s_fermions
        metrics = {
            'su3': s_su3.item(),
            'su2': s_su2.item(),
            'u1': s_u1.item(),
            'higgs': s_higgs.item(),
            'fermion': s_fermions.item()
        }
        
        return total_action, metrics
        

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
        
        # Matrix multiply around the loop: U1 * U2 * U3 * U4
        # We can use a loop over the 4 edges of the plaquette
        loop = u_p[:, 0]
        for i in range(1, 4):
            loop = torch.matmul(loop, u_p[:, i])
        tr = torch.einsum('pii -> p', loop)
        
        # 4. Final Scalar calculation
        # S = beta * sum(1 - Re(Tr(U))/Nc)
        nc = float(self.group_dim)
        action_per_plaq = 1.0 - (tr.real / nc)
        
        return self.beta * torch.sum(action_per_plaq)


class HiggsAction(nn.Module):
    def __init__(self, v=1.0, lam=0.1):
        super().__init__()
        self.v = v
        self.lam = lam

    def forward(self, phi, edge_index, is_fwd, u_su2, u_u1):
        # 1. Potential
        phi_sq = torch.sum(phi.abs()**2, dim=-1)
        v_pot = self.lam * torch.sum((phi_sq - self.v**2)**2)
        
        # 2. Kinetic Interaction with BOTH fields
        src, dst = edge_index[0, is_fwd], edge_index[1, is_fwd]
        
        u_total = u_su2[is_fwd] * u_u1[is_fwd]
        phi_j_transported = torch.einsum('eij, ej -> ei', u_total, phi[src])
        diff = phi[dst] - phi_j_transported
        kinetic = torch.sum(diff.abs()**2)

        return kinetic + v_pot


class HeteroQuantumVacuum(nn.Module):
    def __init__(self, lattice_shape, num_flavors=3, a=1.0, 
                 groups={'su3': 3, 'su2': 2, 'u1': 1}, 
                 betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0}, 
                 v=1.0, lam=0.5):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lattice_shape = lattice_shape
        self.num_flavors = num_flavors
        self.groups = groups
        
        # 1. Lattice Infrastructure
        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        plaq_idx = find_rectangular_loops(lattice_shape, edge_index)
        
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_dirs', edge_dirs)
        self.register_buffer('is_fwd', is_fwd)
        self.register_buffer('partner_map', partner_map)
        
        # 2. Build the Static HeteroData Routing Skeleton
        self.data = HeteroData()
        # All interactions traverse the same spacetime edges
        self.data['quark', 'gauge', 'quark'].edge_index = edge_index
        self.data['lepton', 'gauge', 'lepton'].edge_index = edge_index
        self.data['higgs', 'gauge', 'higgs'].edge_index = edge_index
        self.data = self.data.to(self.device)

        # 3. Learnable Physical Fields (The Vacuum State)
        num_edges = edge_index.size(1)
        num_nodes = np.prod(lattice_shape)
        
        # Gauge Bosons (Cold Start)
        self.u_raw = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(num_edges, dim, dim, dtype=torch.complex64))
            for name, dim in groups.items()
        })
        
        # The Higgs Field
        # self.higgs_phi = nn.Parameter(torch.randn(num_nodes, 2, dtype=torch.complex64))
        self.higgs_phi = nn.Parameter(torch.ones(num_nodes, 2, dtype=torch.complex64) * 0.1)
        # 4. Pseudofermion Buffers (The Heat Bath Particles)
        # Quarks: [Nodes, Flavors, Color, Isospin, Spin]
        self.register_buffer('pf_quark', torch.randn(num_nodes, num_flavors, 3, 2, 4, dtype=torch.complex64))
        # Leptons: [Nodes, Flavors, Isospin, Spin]
        self.register_buffer('pf_lepton', torch.randn(num_nodes, num_flavors, 2, 4, dtype=torch.complex64))

        # 5. Instantiate the Physics Engines
        gnn_model = StandardModelGNN(num_flavors=num_flavors, a=a)
        self.universe = StandardModelUniverse(
            gnn_model, plaq_idx, 
            beta_su3=betas['su3'], beta_su2=betas['su2'], beta_u1=betas['u1'], 
            higgs_v=v, higgs_lam=lam
        )
        
        # 6. Multi-Scale Langevin Time Steps
        self.dt_map = {
            'su3': 0.00005,  # Stiffest/Hardest: Tiny steps to preserve SU(3) manifold
            'su2': 0.0005,   # Moderate: Standard electroweak scale
            'u1':  0.001,   # Fast: Abelian field, easier to sample
            'phi': 0.001     # Higgs: Large steps to explore the VEV landscape
        }
        
        self.to(self.device)

    def langevin_step(self, parameters, dt):
        """Standard Langevin integration: Drift (Gradient) + Diffusion (Quantum Noise)"""
        with torch.no_grad():
            for p in parameters:
                if p.grad is None: continue
                
                force = p.grad.resolve_conj()
                noise = torch.randn_like(p)
                grad = p.grad.resolve_conj()
                grad_norm = torch.norm(grad)
                if grad_norm > 10.0:
                    grad = grad * (10.0 / grad_norm)

                if p.is_complex():
                    noise_imag = torch.randn_like(p)
                    # Complex noise variance is split across real and imaginary
                    noise = (noise + 1j * noise_imag) / np.sqrt(2)
                
                drift = -dt * force
                diffusion = np.sqrt(2 * dt) * noise
                p.add_(drift + diffusion)

    def refresh_heat_bath(self):
        """Resamples the pseudofermions to maintain detailed balance."""
        # phi = M^dag eta (with M^dag = g5 M g5 and eta ~ complex N(0,1)) so that
        # S_pf = phi^dag (M^dag M)^-1 phi carries the fermion determinant weight;
        # a plain Gaussian phi would sample the wrong ensemble.
        with torch.no_grad():
            gates = {
                name: get_gate(raw, self.partner_map, self.is_fwd, is_su=(name != 'u1'))
                for name, raw in self.u_raw.items()
            }
            gnn = self.universe.gnn
            g5 = gnn.g5
            g5_eta = {
                'quark': torch.einsum('ss, ...s -> ...s', g5, torch.randn_like(self.pf_quark)),
                'lepton': torch.einsum('ss, ...s -> ...s', g5, torch.randn_like(self.pf_lepton)),
                'higgs': self.higgs_phi,
            }
            d_out = gnn(g5_eta, self.data.edge_index_dict, gates, self.is_fwd, self.edge_dirs)
            self.pf_quark.copy_(torch.einsum('ss, ...s -> ...s', g5, d_out['quark']))
            self.pf_lepton.copy_(torch.einsum('ss, ...s -> ...s', g5, d_out['lepton']))

    def find_vacuum(self, steps=1000, thermalize_steps=200):
        print("--- Heterogeneous Standard Model: Quantum Thermalization ---")
        history = {k: [] for k in ['total', 'vev', 'su3', 'su2', 'u1', 'higgs', 'fermion']}
        
        for step in range(steps):
            self.zero_grad()
            
            # 1. Project Raw Parameters to Physical Manifolds
            physical_gates = {
                name: get_gate(raw, self.partner_map, self.is_fwd, is_su=(name != 'u1'))
                for name, raw in self.u_raw.items()
            }
            
            # 2. Inject parameters into HeteroData
            self.data['higgs'].x = self.higgs_phi
            pf_dict = {'quark': self.pf_quark, 'lepton': self.pf_lepton}

            # 3. Calculate Action (Forward Pass)
            total_action, metrics = self.universe(
                self.data, physical_gates, pf_dict, self.is_fwd, self.edge_dirs
            )
            
            # 4. Calculate Physical Forces (Backward Pass)
            total_action.backward()

            # 5. Multi-Scale Langevin Update
            for name, param in self.u_raw.items():
                self.langevin_step([param], dt=self.dt_map[name])
            self.langevin_step([self.higgs_phi], dt=self.dt_map['phi'])

            # 6. Fermion Heat Bath Resampling
            if step % 10 == 0:
                self.refresh_heat_bath()

            # 7. Logging & Thermalization Logic
            current_vev = torch.norm(self.higgs_phi, dim=-1).mean().item()
            
            if step >= thermalize_steps:
                history['total'].append(total_action.item())
                history['vev'].append(current_vev)
                for k, v in metrics.items():
                    history[k].append(v)

            if step % 50 == 0:
                phase = "Thermalizing" if step < thermalize_steps else "Sampling    "
                print(f"[{phase}] Step {step:03d} | Action: {total_action.item():.2f} | "
                      f"SU3: {metrics['su3']:.2f} | Fermions: {metrics['fermion']:.2f} | "
                      f"VEV: {current_vev:.3f}")

        return history
        

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


def plot_action_landscape2(history):
    """
    Plots the separate energy contributions to the vacuum.
    """
    epochs = range(len(history['total']))
    
    plt.figure(figsize=(10, 6))
    
    # 1. Total Action (The overarching landscape)
    plt.plot(epochs, history['total'], 'k-', linewidth=2, label='Total Action ($S_{tot}$)')
    
    # 2. Individual Sector Contributions
    # Note: You may need to scale these if one dominates the others by orders of magnitude
    plt.plot(epochs, history['su3'], 'r--', alpha=0.8, label='SU(3) Strong Action')
    plt.plot(epochs, history['su2'], 'g--', alpha=0.8, label='SU(2) Weak Action')
    plt.plot(epochs, history['u1'], 'b--', alpha=0.8, label='U(1) Hypercharge Action')
    plt.plot(epochs, history['higgs'], 'm-.', alpha=0.8, label='Higgs Potential ($V(\phi)$)')
    plt.plot(epochs, history['fermion'], 'c:', linewidth=2, label='Fermion Action ($\overline{\psi} D \psi$)')
    
    plt.title("Vacuum Loss Landscape (Action Contributions over Time)")
    plt.xlabel("Cooling Steps (Epochs)")
    plt.ylabel("Action Energy")
    plt.yscale('log') # Log scale is crucial to see vastly different energy levels
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("plots/actions.png")


if __name__ == "__main__":
    # L_SHAPE = (16, 16, 16, 4)
    # L_SHAPE = (16, 16, 4)
    L_SHAPE = (16, 16)

    # model = VacuumFinder(
    #     lattice_shape=L_SHAPE,
    #     groups={'su3': 3, 'su2': 2, 'u1': 1},
    #     betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0} 
    # )
    
    model = HeteroQuantumVacuum(lattice_shape=L_SHAPE)
    history = model.find_vacuum()
    plot_action_landscape(history)



class GaugeEquivariantConv(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add', node_dim=0)
        self.lin = nn.Linear(channels, channels, bias=False, dtype=torch.complex64)

    def forward(self, x, edge_index, u_gate):
        # x: [Nodes, Channels, Group_Dim]
        # u_gate: [Edges, Group_Dim, Group_Dim]
        out = self.propagate(edge_index, x=x, u_gate=u_gate)
        
        # Mix channels: transpose to [Nodes, Group_Dim, Channels] 
        # so Linear only touches the Channel dimension.
        out = out.transpose(1, 2)
        out = self.lin(out)
        return out.transpose(1, 2)

    def message(self, x_j, u_gate):
        # Parallel Transport: U_ij * psi_j
        # u_gate: [E, G, G], x_j: [E, C, G]
        return torch.einsum('egh, ech -> ecg', u_gate, x_j)


class StrongGaugeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, u_su3):
        # x shape: [Nodes, Flavors, Color(3), Isospin(2), Spin(4)]
        # u_su3 shape: [Edges, 3, 3]
        return self.propagate(edge_index, x=x, u_su3=u_su3)

    def message(self, x_j, u_su3):
        # Parallel transport ONLY the Color dimension (index c)
        # f=flavor, c=color_out, d=color_in, i=isospin, s=spin
        return torch.einsum('ecd, efdis -> efcis', u_su3, x_j)


class ElectroweakGaugeConv(MessagePassing):
    def __init__(self, P_L, P_R):
        super().__init__(aggr='add')
        self.P_L = P_L
        self.P_R = P_R

    def forward(self, x, edge_index, u_su2, u_u1):
        # Works for Quarks: [N, Flavors, Color, Isospin, Spin]
        # Works for Leptons: [N, Flavors, Isospin, Spin]
        return self.propagate(edge_index, x=x, u_su2=u_su2, u_u1=u_u1)

    def message(self, x_j, u_su2, u_u1):
        # 1. Project into Left and Right handed components
        # The spin dimension is always the last one
        x_L = torch.einsum('ab, ...b -> ...a', self.P_L, x_j)
        x_R = torch.einsum('ab, ...b -> ...a', self.P_R, x_j)
        
        # 2. Left-Handed Interaction: Feels both SU(2) and U(1)
        # u_su2 rotates the Isospin dimension (second to last)
        # u_u1 is a scalar phase [E, 1, 1], so it just multiplies
        u_total_L = u_u1 * u_su2
        rotated_L = torch.einsum('eij, ...js -> ...is', u_total_L, x_L)
        
        # 3. Right-Handed Interaction: Feels ONLY U(1) hypercharge
        # It is an SU(2) singlet, so the isospin state is unchanged
        rotated_R = u_u1 * x_R
        
        # 4. Recombine
        return rotated_L + rotated_R


class WilsonDiracOperator(nn.Module):
    def __init__(self, y=1.0, bare_mass=0.01, r=1.0):
        super().__init__()
        self.y = y                 # Yukawa coupling constant
        self.bare_mass = bare_mass # Tiny stabilizer for the CG solver
        self.r = r
        
        # 1. Precompute Gamma matrices
        g1, g2, g3, g4 = self.get_gamma_matrices()
        gammas = torch.stack([g1, g2, g3, g4], dim=0)
        self.register_buffer('gammas', gammas)
        
        g5 = g1 @ g2 @ g3 @ g4
        self.register_buffer('g5', g5)
        self.register_buffer('id_spin', torch.eye(4, dtype=torch.complex64))


    def get_gamma_matrices(self):
        # Standard Euclidean Gamma matrices (Chiral representation)
        zeros = torch.zeros(2, 2, dtype=torch.complex64)
        eye = torch.eye(2, dtype=torch.complex64)
        px = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        py = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        pz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        # Gamma 1, 2, 3 (Spatial)
        g1 = torch.cat([torch.cat([zeros, 1j*px], dim=1), torch.cat([-1j*px, zeros], dim=1)], dim=0)
        g2 = torch.cat([torch.cat([zeros, 1j*py], dim=1), torch.cat([-1j*py, zeros], dim=1)], dim=0)
        g3 = torch.cat([torch.cat([zeros, 1j*pz], dim=1), torch.cat([-1j*pz, zeros], dim=1)], dim=0)
        # Gamma 4 (Temporal)
        g4 = torch.cat([torch.cat([zeros, eye], dim=1), torch.cat([eye, zeros], dim=1)], dim=0)
        
        return [g1, g2, g3, g4]

    def get_chiral_projectors(g5):
        """Creates Left and Right handed chiral projectors."""
        eye = torch.eye(4, dtype=torch.complex64, device=g5.device)
        P_L = 0.5 * (eye - g5)
        P_R = 0.5 * (eye + g5)
        return P_L, P_R


    # ADDED phi to the forward pass arguments
    def forward(self, psi, phi, edge_index, edge_dirs, u_su3, is_fwd):
        dim_space = len(torch.unique(edge_dirs))
        
        # --- NEW: DYNAMICAL YUKAWA MASS ---
        # phi is [Nodes, 2] (Complex SU(2) doublet)
        # Calculate the gauge-invariant local magnitude |phi| at every node
        phi_sq = torch.sum(phi.abs()**2, dim=-1)
        higgs_vev_local = torch.sqrt(phi_sq).view(-1, 1, 1) # Shape: [Nodes, 1, 1]
        
        # The mass is now a dynamic field dependent on the Higgs!
        dynamic_mass = self.bare_mass + (self.y * higgs_vev_local)
        out = (dynamic_mass + self.r * dim_space) * psi
        # ----------------------------------
        
        src, dst = edge_index[0], edge_index[1]
        
        # Parallel Transport (Color Rotation)
        rotated_psi = torch.einsum('ecd, eds -> ecs', u_su3, psi[src])
        
        # Vectorized Spin Gather
        g_edge = self.gammas[edge_dirs]
        
        # Spin Projection
        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1).to(psi.device)
        P_edge = self.r * self.id_spin + sign * g_edge
        
        # Multiply projection against the rotated spinor
        term = torch.einsum('eab, ecb -> eca', P_edge, rotated_psi)
        
        # Scatter Add
        out.index_add_(0, dst, -0.5 * term)
        
        return out


@torch.no_grad() 
def conjugate_gradient(dirac_op, psi, phi, edge_index, edge_dirs, u_su3, is_fwd, max_iter=200, tol=1e-6):
    x = torch.zeros_like(psi)
    
    def apply_m_dag_m(vec):
        # Pass phi into the Dirac operator
        d_vec = dirac_op(vec, phi, edge_index, edge_dirs, u_su3, is_fwd)
        g5_d_vec = torch.einsum('ss, ecs -> ecs', dirac_op.g5, d_vec)
        d_dag_partial = dirac_op(g5_d_vec, phi, edge_index, edge_dirs, u_su3, is_fwd)
        return torch.einsum('ss, ecs -> ecs', dirac_op.g5, d_dag_partial)

    r = psi - apply_m_dag_m(x)
    p = r.clone()
    rsold = torch.sum(r.conj() * r).real

    for i in range(max_iter):
        ap = apply_m_dag_m(p)
        alpha = rsold / (torch.sum(p.conj() * ap).real + 1e-10)
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = torch.sum(r.conj() * r).real
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        
    return x


class PseudofermionAction(nn.Module):
    def __init__(self, y=1.0):
        super().__init__()
        self.dirac = WilsonDiracOperator(y=y)

    def forward(self, pf_phi, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd):
        x = conjugate_gradient(self.dirac, pf_phi, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd)
        actual_action = torch.sum(pf_phi.conj() * x).real
        dx = self.dirac(x, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd)
        surrogate_loss = -torch.sum(dx.abs()**2)
                
        return surrogate_loss - surrogate_loss.detach() + actual_action


