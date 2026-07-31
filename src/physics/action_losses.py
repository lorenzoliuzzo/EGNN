# src/physics/action_losses.py

import torch
import torch.nn as nn

from .gauge_groups import build_sm_gate


# =============================================================================
# 2. GAUGE ACTION (WILSON)
# =============================================================================

def wilson_plaquette_loss(u_gate, p1, p2, p3, p4):
    """
    Computes the Yang-Mills gauge field energy (Wilson Action). 
    Assumes p1, p2 are forward links and p3, p4 are backward links needing adjoints.
    """
    N = u_gate.size(-1) 
    u1, u2 = u_gate[p1], u_gate[p2]
    u3, u4 = u_gate[p3], u_gate[p4]

    # Closed loop n00 -> n10 -> n11 -> n01 -> n00 composed in reverse path order
    # (transport acts on column vectors), so the trace telescopes under
    # U'_e = G_dst U_e G_src^dag
    u_p = u4.mH @ u3.mH @ u2 @ u1
    
    # Re(Tr(U_p))
    tr_u_p = torch.real(torch.einsum('...ii->...', u_p))

    # S_W = sum(1 - 1/N * Re(Tr(U_p)))
    return torch.mean(1.0 - (tr_u_p / N))


# =============================================================================
# 3. MATTER ACTIONS (KINETIC)
# =============================================================================

def covariant_kinetic_loss(phi, edge_index, u_gate):
    """
    Computes the magnitude of the discrete covariant derivative for a generalized field.
    |D_mu phi|^2 = |phi_i - U_ij * phi_j|^2
    """
    phi_j = phi[edge_index[0]] # Source nodes
    phi_i = phi[edge_index[1]] # Target nodes
    
    # Parallel transport phi_j across the edge using the gauge link U_ij
    transported_j = torch.einsum('enm,ecm->ecn', u_gate, phi_j)
    diff_mag_sq = torch.sum(torch.abs(phi_i - transported_j)**2, dim=-1)
    return torch.mean(diff_mag_sq)


def sm_kinetic_loss(phi_matter, edge_index, u_dict, hypercharge=1/6):
    """
    Computes the covariant derivative for a fundamental SM multiplet (e.g., Left-Handed Quarks).
    Dynamically builds the 6x6 Kronecker gate before transport.
    """
    u_total = build_sm_gate(u_dict['su3'], u_dict['su2'], u_dict['u1'], hypercharge)
    return covariant_kinetic_loss(phi_matter, edge_index, u_total)


def higgs_kinetic_loss(phi_higgs, edge_index, u_dict):
    """
    Computes the kinetic loss for the Higgs doublet (SU(3) Singlet, Y = +1/2).
    """
    # Build the U_SU2 ⊗ U_U1 gate
    u_ew = build_electroweak_gate(u_dict['su2'], u_dict['u1'], hypercharge=0.5)
    
    return covariant_kinetic_loss(phi_higgs, edge_index, u_ew)


# =============================================================================
# 4. HIGGS POTENTIAL
# =============================================================================

def higgs_potential_loss(phi_higgs, v_target: float, lambda_coupling: float):
    """
    Computes the physical Mexican Hat potential for the Electroweak Higgs field.
    V(phi) = lambda * (|phi|^2 - v^2/2)^2
    
    Returns the potential loss and the current measured VEV.
    """
    # Magnitude squared at each node: |Phi|^2
    mag_sq = torch.sum(torch.abs(phi_higgs)**2, dim=-1) 
    
    # Physical convention: minimum is at v^2 / 2
    local_potential = lambda_coupling * (mag_sq - (v_target**2) / 2.0)**2 
    
    return torch.mean(local_potential), torch.mean(mag_sq)
    
    

def sm_yukawa_loss(psi_L, psi_R, phi_higgs, yukawa_matrix):
    """
    Computes the gauge-invariant Yukawa interaction: y * (L_bar * Phi * R + h.c.)
    
    psi_L: Left-handed SU(2) Doublet [nodes, channels, 2]
    psi_R: Right-handed SU(2) Singlet [nodes, channels, 1]
    phi_higgs: SU(2) Doublet [nodes, channels, 2]
    """
    # 1. Gauge-invariant inner product of SU(2) doublets: (L_bar * Phi)
    # Einsum contraction over the SU(2) index 'i' creates a gauge singlet.
    l_bar_phi = torch.einsum('nci,nci->nc', torch.conj(psi_L), phi_higgs)
    
    # 2. Couple to the Right-handed singlet
    # psi_R is shape [nodes, channels, 1], we squeeze it to [nodes, channels]
    interaction = yukawa_matrix * l_bar_phi * psi_R.squeeze(-1)
    
    # 3. Add Hermitian conjugate to ensure the Lagrangian is real
    lagrangian_term = interaction + torch.conj(interaction)
    
    # We want to minimize the action, so we return the negative real part
    return -torch.mean(torch.real(lagrangian_term))

    

# =============================================================================
# 1. GAMMA 5 DEFINITION
# =============================================================================

def get_gamma_5(device):
    """
    Computes the Gamma_5 matrix for the chiral representation.
    gamma_5 = gamma_1 * gamma_2 * gamma_3 * gamma_4
    In the standard chiral basis, it is simply block diagonal: [I, -I]
    """
    I = torch.eye(2, dtype=torch.complex64, device=device)
    gamma_5 = torch.block_diag(I, -I)
    return gamma_5

# =============================================================================
# 2. THE DIRAC SYSTEM APPLICATOR
# =============================================================================

def apply_D_dag_D(y, edge_index, u_gate, edge_dirs, dirac_layer):
    """
    Computes (D^dag D) * y using the gamma_5 hermiticity trick.
    D^dag = gamma_5 * D * gamma_5
    """
    gamma_5 = get_gamma_5(y.device)
    
    # 1. Apply D:  y1 = D * y
    y1 = dirac_layer(y, edge_index, edge_dirs, u_gate)
    
    # 2. Apply gamma_5: y2 = gamma_5 * y1
    # Contracting the spin index (dimension -2)
    y2 = torch.einsum('ab, ...bc -> ...ac', gamma_5, y1)
    
    # 3. Apply D again: y3 = D * y2
    y3 = dirac_layer(y2, edge_index, edge_dirs, u_gate)
    
    # 4. Apply gamma_5 again: out = gamma_5 * y3
    out = torch.einsum('ab, ...bc -> ...ac', gamma_5, y3)
    
    return out

# =============================================================================
# 3. CONJUGATE GRADIENT SOLVER
# =============================================================================

def solve_conjugate_gradient(dirac_layer, chi, edge_index, edge_dirs, u_gate, tol=1e-6, max_iter=500):
    """
    Solves (D^dag D) Y = chi for Y using the Conjugate Gradient method.
    """
    # Helper for the global complex dot product: Re(a^dagger * b)
    def dot_product(v1, v2):
        return torch.sum(torch.real(torch.conj(v1) * v2))

    # Initialize Y with zeros (or a warm start if you have one)
    Y = torch.zeros_like(chi)
    
    # Initial residual: r = chi - A*Y = chi (since Y is 0)
    r = chi.clone()
    p = r.clone()
    
    rs_old = dot_product(r, r)
    
    for i in range(max_iter):
        # A * p = (D^dag D) * p
        Ap = apply_D_dag_D(p, edge_index, u_gate, edge_dirs, dirac_layer)
        
        # Step size alpha
        p_Ap = dot_product(p, Ap)
        alpha = rs_old / (p_Ap + 1e-12) # Add epsilon to prevent division by zero
        
        # Update solution and residual
        Y = Y + alpha * p
        r = r - alpha * Ap
        
        rs_new = dot_product(r, r)
        
        # Check convergence
        if torch.sqrt(rs_new) < tol:
            # print(f"CG converged in {i} iterations.")
            break
            
        # Update search direction
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    # If it hits max_iter without breaking, it hasn't fully converged, 
    # but we return the best approximation.
    return Y
    

def pseudofermion_action(chi, edge_index, u_gate, edge_dirs, dirac_layer):
    """
    Computes S_pf = chi^dag * (D^dag D)^{-1} * chi
    """
    # 1. We must solve the system (D^dag D) * Y = chi for Y.
    # In practice, Lattice QCD uses Conjugate Gradient (CG) solvers here.
    # For a naive differentiable PyTorch approximation, we can use an iterative solver 
    # or rely on the GNN to learn an inverse mapping, but standard CG is best.
    with torch.no_grad():
        Y = solve_conjugate_gradient(dirac_layer, chi, edge_index, edge_dirs, u_gate)
    
    # 2. Action is chi^dagger * Y
    action = torch.einsum('ncsi, ncsi -> n', torch.conj(chi), Y)
    
    return torch.mean(torch.real(action))
    
# import torch

# def wilson_plaquette_loss(u_gate, p1, p2, p3, p4):
#     """
#     Computes the Yang-Mills gauge field energy. 
#     Minimizing this ensures the gauge links remain physically well-behaved.
#     """
#     N = u_gate.shape[-1] 
#     u1, u2 = u_gate[p1], u_gate[p2]
#     u3, u4 = u_gate[p3], u_gate[p4]
#     # u_p = u1 @ u2 @ u3 @ u4

#     # Path: Forward -> Forward -> Backward -> Backward
#     u_p = u1 @ u2 @ u3.mH @ u4.mH
#     tr_u_p = torch.real(torch.einsum('...ii->...', u_p))

#     # tr_u_p = torch.real(torch.diagonal(u_p, dim1=-2, dim2=-1).sum(-1))
#     return torch.mean(1.0 - (tr_u_p / N))


# def mixed_electroweak_kinetic_loss(phi, edge_index, u_su2, u_u1, hypercharge=1.0):
#     """
#     Computes the kinetic loss with SU(2) x U(1) mixing[cite: 2, 3].
#     The parallel transport now depends on both gauge fields simultaneously.
#     """
#     phi_j = phi[edge_index[0]] 
#     phi_i = phi[edge_index[1]] 
    
#     # Combined parallel transport: U_SU2 * (U_U1 ^ hypercharge)
#     # U_U1 is a complex scalar; we raise it to the power of the hypercharge[cite: 1].
#     u_combined = u_su2 * (u_u1 ** hypercharge)
    
#     # Transport phi_j using the combined Electroweak connection
#     transported_j = torch.einsum('enm,ecm->ecn', u_combined, phi_j)
    
#     # The squared magnitude of the difference[cite: 1]
#     diff_mag_sq = torch.sum(torch.abs(phi_i - transported_j)**2, dim=-1)
    
#     return torch.mean(diff_mag_sq)


# def covariant_kinetic_loss(phi, edge_index, u_gate):
#     """
#     Computes the magnitude of the discrete covariant derivative.
#     Minimizing |phi_i - U_ij * phi_j|^2 bounds the loss at 0 and 
#     prevents artificial inflation of the field magnitude.
#     """
#     phi_j = phi[edge_index[0]] 
#     phi_i = phi[edge_index[1]] 
    
#     # Parallel transport phi_j across the edge using the gauge link U_ij
#     transported_j = torch.einsum('enm,ecm->ecn', u_gate, phi_j)
    
#     # Calculate the squared magnitude of the difference
#     diff = phi_i - transported_j
#     diff_mag_sq = torch.sum(torch.abs(diff)**2, dim=-1)
    
#     return torch.mean(diff_mag_sq)
    

# def covariant_kinetic_loss_unified(phi, edge_index, u_total):
#     """
#     Computes the covariant derivative for the full 6D SM multiplet.
#     phi: [nodes, channels, 6]
#     u_total: [edges, 6, 6] unified Kronecker gauge link
#     """
#     phi_j = phi[edge_index[0]] 
#     phi_i = phi[edge_index[1]] 
    
#     # Transport using the unified 6x6 SM Gate
#     transported_j = torch.einsum('enm,ecm->ecn', u_total, phi_j)
    
#     # Squared magnitude of the unified covariant difference
#     diff = phi_i - transported_j
#     diff_mag_sq = torch.sum(torch.abs(diff)**2, dim=-1)
    
#     return torch.mean(diff_mag_sq)


# def higgs_hat_loss(out_su2, v_target):
#     # Magnitude squared at each node
#     mag_sq = torch.sum(torch.abs(out_su2)**2, dim=-1) 
    
#     # The Mexican Hat potential applied LOCALLY at every node
#     local_potential = (mag_sq - v_target**2)**2 
    
#     # Average the potential energy across the lattice
#     return torch.mean(local_potential), torch.mean(mag_sq)

# # def higgs_hat_loss(out_su2, v_target):
# #     mag_sq = torch.sum(torch.abs(out_su2)**2, dim=-1) 
# #     mean_mag_sq = torch.mean(mag_sq) 
# #     return (mean_mag_sq - v_target**2)**2, mean_mag_sq


# def dirac_kinetic_loss(psi, edge_index, u_total, gamma_mu):
#     """
#     Computes the kinetic energy for spin-1/2 fermion fields using the Dirac equation.
#     """
#     psi_j = psi[edge_index[0]]
#     psi_i = psi[edge_index[1]]
    
#     # Transport spinor across the gauge field
#     transported_j = torch.einsum('enm,ecm->ecn', u_total, psi_j)
    
#     # The derivative includes the gamma matrices mapping spin components
#     # (Requires gamma_mu to be defined and mapped to edge directions)
#     dirac_diff = psi_i - torch.einsum('ab,ecb->eca', gamma_mu, transported_j)
    
#     return torch.mean(torch.sum(torch.abs(dirac_diff)**2, dim=-1))


