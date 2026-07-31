# src/physics/gauge_groups.py

import numpy as np
import torch


def su2_generators(device):
    """Returns the 3 Pauli matrices for SU(2)."""
    su2 = torch.zeros((3, 2, 2), dtype=torch.complex64, device=device)
    su2[0] = torch.tensor([[0, 1], [1, 0]])
    su2[1] = torch.tensor([[0, -1j], [1j, 0]])
    su2[2] = torch.tensor([[1, 0], [0, -1]])
    return su2


def su3_generators(device):
    """Returns the 8 Gell-Mann matrices for SU(3)."""
    su3 = torch.zeros((8, 3, 3), dtype=torch.complex64, device=device)
    su3[0] = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    su3[1] = torch.tensor([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    su3[2] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    su3[3] = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    su3[4] = torch.tensor([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    su3[5] = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    su3[6] = torch.tensor([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    su3[7] = (1/np.sqrt(3)) * torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
    return su3

def get_gamma_matrices(device):
    """Euclidean Gamma matrices in chiral representation."""
    I = torch.eye(2, dtype=torch.complex64, device=device)
    tau1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    tau2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    tau3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    
    # Gamma_1,2,3: off-diagonal chiral form so gamma_mu^2 = +I (Euclidean Clifford
    # algebra) and g1@g2@g3@g4 = block_diag(I, -I), matching get_gamma_5
    Z = torch.zeros_like(I)
    g1 = torch.cat([torch.cat([Z, -1j*tau1], dim=1), torch.cat([1j*tau1, Z], dim=1)], dim=0)
    g2 = torch.cat([torch.cat([Z, -1j*tau2], dim=1), torch.cat([1j*tau2, Z], dim=1)], dim=0)
    g3 = torch.cat([torch.cat([Z, -1j*tau3], dim=1), torch.cat([1j*tau3, Z], dim=1)], dim=0)
    # Gamma_4 (Euclidean time)
    g4 = torch.cat([torch.cat([torch.zeros_like(I), I], dim=1),
                    torch.cat([I, torch.zeros_like(I)], dim=1)], dim=0)
    
    return [g1, g2, g3, g4]


def get_sm_generators(device):
    return {
        'su3': su3_generators(device),
        'su2': su2_generators(device),
        'u1':  1j
    }

def exp_sm_algebra_to_group(a_dict, gen_dict, hypercharge=1.0):
    """
    Transforms learned gauge potentials into physical gauge link matrices $U = exp(i * \alpha * T)$
    """
    # Contraction over algebra index 'a' to produce [edges, dim, dim] matrix group elements
    u_su3 = torch.linalg.matrix_exp(1j * torch.einsum('ea,abc->ebc', a_dict['su3'].to(torch.complex64), gen_dict['su3']))
    u_su2 = torch.linalg.matrix_exp(1j * torch.einsum('ea,abc->ebc', a_dict['su2'].to(torch.complex64), gen_dict['su2']))

    # U(1) is a 1x1 complex phase rotation
    u_u1 = torch.exp(1j * a_dict['u1'].view(-1, 1, 1).to(torch.complex64) * hypercharge)

    return (u_su3, u_su2, u_u1)



def kronecker_prod(A, B):
    """Computes batched Kronecker product."""
    b = A.size(0)
    out = torch.einsum('bij,bkl->bikjl', A, B)
    return out.reshape(b, A.size(1) * B.size(1), A.size(2) * B.size(2))


def build_sm_gate(u_su3, u_su2, u_u1, hypercharge: float):
    """
    Builds the exact 6x6 SM representation: U_SU3 ⊗ U_SU2 ⊗ (U_U1)^Y
    """
    # 1. Kron SU(3) and SU(2) -> [edges, 6, 6]
    u_32 = kronecker_prod(u_su3, u_su2)
    
    # 2. Apply U(1) phase. Because U(1) is 1x1, exponentiation is element-wise safe.
    # U_U1 shape: [edges, 1, 1]. Broadcasting handles the multiplication.
    u_u1_y = u_u1 ** hypercharge 
    
    return u_32 * u_u1_y


def build_electroweak_gate(u_su2, u_u1, hypercharge: float = 0.5):
    """
    Builds the 2x2 Electroweak representation for the Higgs: U_SU2 ⊗ (U_U1)^Y
    (SU(3) is ignored because the Higgs is a color singlet).
    """
    u_u1_y = u_u1 ** hypercharge
    return u_su2 * u_u1_y



class SMGateFactory:
    """Generates specific gauge-equivariant gates for SM particle species."""
    def __init__(self, u_dict):
        self.u_dict = u_dict
        self.device = u_dict['su3'].device
        self.edges = u_dict['su3'].size(0)

    def _get_eye(self, dim):
        return torch.eye(dim, device=self.device).expand(self.edges, dim, dim).to(torch.complex64)

    def build_gate(self, su3_dim, su2_dim, hypercharge):
        gate = self.u_dict['su3'] if su3_dim == 3 else self._get_eye(1)
        gate = kronecker_prod(gate, self.u_dict['su2']) if su2_dim == 2 else kronecker_prod(gate, self._get_eye(1))
        return gate * (self.u_dict['u1'] ** hypercharge)

    def get_all_gates(self):
        gates = {}
        for g in [1, 2, 3]:
            # Each generation uses identical gauge logic
            gates.update({
                (f'quark_left_g{g}', 'transport', f'quark_left_g{g}'): self.build_gate(3, 2,  1/6),
                (f'quark_up_right_g{g}', 'transport', f'quark_up_right_g{g}'): self.build_gate(3, 1,  2/3),
                (f'quark_down_right_g{g}', 'transport', f'quark_down_right_g{g}'): self.build_gate(3, 1, -1/3),
                (f'lepton_left_g{g}', 'transport', f'lepton_left_g{g}'): self.build_gate(1, 2, -1/2),
                (f'lepton_right_g{g}', 'transport', f'lepton_right_g{g}'): self.build_gate(1, 1, -1.0),
                (f'lepton_nu_right_g{g}', 'transport', f'lepton_nu_right_g{g}'): self.build_gate(1, 1, 0.0), # Sterile!
            })
        
        # Add the Higgs (Singlet under SU3, Doublet under SU2, Y=1/2)
        gates[('higgs', 'transport', 'higgs')] = self.build_gate(1, 2, 0.5)
        return gates
