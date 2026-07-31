import numpy as np
import torch


def get_gate(
    raw_u: torch.Tensor,
    partner_map: torch.Tensor,
    is_fwd: torch.Tensor,
    is_su: bool = True,
) -> torch.Tensor:
    # anti-Hermitian generator -> exp lands in U(N)
    h = (raw_u - raw_u.adjoint()).resolve_conj()

    if is_su:
        # subtract the trace so det(exp(H)) = 1
        dim = h.shape[-1]
        trace = torch.einsum('eii->e', h)
        eye = torch.eye(dim, dtype=torch.complex64, device=h.device)
        h = h - (trace.view(-1, 1, 1) / dim) * eye

    u_all = torch.matrix_exp(h)

    # U(-mu) = U(mu)^dag keeps every closed loop consistent
    u_bwd = u_all[partner_map].adjoint().resolve_conj()
    return torch.where(is_fwd.view(-1, 1, 1), u_all, u_bwd)


def su2_generators(device: torch.device | str) -> torch.Tensor:
    su2 = torch.zeros((3, 2, 2), dtype=torch.complex64, device=device)
    su2[0] = torch.tensor([[0, 1], [1, 0]])
    su2[1] = torch.tensor([[0, -1j], [1j, 0]])
    su2[2] = torch.tensor([[1, 0], [0, -1]])
    return su2


def su3_generators(device: torch.device | str) -> torch.Tensor:
    su3 = torch.zeros((8, 3, 3), dtype=torch.complex64, device=device)
    su3[0] = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    su3[1] = torch.tensor([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    su3[2] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    su3[3] = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    su3[4] = torch.tensor([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
    su3[5] = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    su3[6] = torch.tensor([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    su3[7] = (1 / np.sqrt(3)) * torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -2]])
    return su3


def get_sm_generators(device: torch.device | str) -> dict:
    return {
        'su3': su3_generators(device),
        'su2': su2_generators(device),
        'u1': 1j,
    }


def exp_sm_algebra_to_group(
    a_dict: dict, gen_dict: dict, hypercharge: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_su3 = torch.linalg.matrix_exp(
        1j * torch.einsum('ea,abc->ebc', a_dict['su3'].to(torch.complex64), gen_dict['su3']))
    u_su2 = torch.linalg.matrix_exp(
        1j * torch.einsum('ea,abc->ebc', a_dict['su2'].to(torch.complex64), gen_dict['su2']))
    u_u1 = torch.exp(1j * a_dict['u1'].view(-1, 1, 1).to(torch.complex64) * hypercharge)
    return u_su3, u_su2, u_u1


def kronecker_prod(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    b = A.size(0)
    out = torch.einsum('bij,bkl->bikjl', A, B)
    return out.reshape(b, A.size(1) * B.size(1), A.size(2) * B.size(2))


def build_sm_gate(
    u_su3: torch.Tensor, u_su2: torch.Tensor, u_u1: torch.Tensor, hypercharge: float,
) -> torch.Tensor:
    u_32 = kronecker_prod(u_su3, u_su2)
    return u_32 * (u_u1 ** hypercharge)


def build_electroweak_gate(
    u_su2: torch.Tensor, u_u1: torch.Tensor, hypercharge: float = 0.5,
) -> torch.Tensor:
    return u_su2 * (u_u1 ** hypercharge)


class SMGateFactory:
    """Builds the per-species U_SU3 (x) U_SU2 (x) U_U1^Y transport gates."""

    def __init__(self, u_dict: dict):
        self.u_dict = u_dict
        self.device = u_dict['su3'].device
        self.edges = u_dict['su3'].size(0)

    def _get_eye(self, dim: int) -> torch.Tensor:
        return torch.eye(dim, device=self.device).expand(self.edges, dim, dim).to(torch.complex64)

    def build_gate(self, su3_dim: int, su2_dim: int, hypercharge: float) -> torch.Tensor:
        gate = self.u_dict['su3'] if su3_dim == 3 else self._get_eye(1)
        gate = (kronecker_prod(gate, self.u_dict['su2']) if su2_dim == 2
                else kronecker_prod(gate, self._get_eye(1)))
        return gate * (self.u_dict['u1'] ** hypercharge)

    def get_all_gates(self) -> dict:
        gates = {}
        for g in [1, 2, 3]:
            gates.update({
                (f'quark_left_g{g}', 'transport', f'quark_left_g{g}'): self.build_gate(3, 2, 1 / 6),
                (f'quark_up_right_g{g}', 'transport', f'quark_up_right_g{g}'): self.build_gate(3, 1, 2 / 3),
                (f'quark_down_right_g{g}', 'transport', f'quark_down_right_g{g}'): self.build_gate(3, 1, -1 / 3),
                (f'lepton_left_g{g}', 'transport', f'lepton_left_g{g}'): self.build_gate(1, 2, -1 / 2),
                (f'lepton_right_g{g}', 'transport', f'lepton_right_g{g}'): self.build_gate(1, 1, -1.0),
                # right-handed neutrinos are sterile: no gauge charge at all
                (f'lepton_nu_right_g{g}', 'transport', f'lepton_nu_right_g{g}'): self.build_gate(1, 1, 0.0),
            })
        gates[('higgs', 'transport', 'higgs')] = self.build_gate(1, 2, 0.5)
        return gates
