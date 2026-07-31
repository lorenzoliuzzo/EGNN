from collections.abc import Callable

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


def get_gamma_matrices(
    device: torch.device | str = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euclidean chiral gammas: gamma_mu^2 = +I and g5 = g1 g2 g3 g4 = diag(I, -I)."""
    eye2 = torch.eye(2, dtype=torch.complex64, device=device)
    Z = torch.zeros_like(eye2)
    tau1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    tau2 = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    tau3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

    def offdiag(tau: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.cat([Z, -1j * tau], dim=1),
                          torch.cat([1j * tau, Z], dim=1)], dim=0)

    g4 = torch.cat([torch.cat([Z, eye2], dim=1), torch.cat([eye2, Z], dim=1)], dim=0)
    gammas = torch.stack([offdiag(tau1), offdiag(tau2), offdiag(tau3), g4], dim=0)
    g5 = gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
    return gammas, g5


def get_chiral_projectors(g5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    id_spin = torch.eye(4, dtype=torch.complex64, device=g5.device)
    return 0.5 * (id_spin - g5), 0.5 * (id_spin + g5)


def cg_solve(
    apply_A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> torch.Tensor:
    """Conjugate gradient for a Hermitian positive-definite A, x0 = 0."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.sum(r.conj() * r).real

    for _ in range(max_iter):
        ap = apply_A(p)
        alpha = rs_old / (torch.sum(p.conj() * ap).real + 1e-12)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.sum(r.conj() * r).real
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


# ---------------------------------------------------------------------------
# Standalone Wilson-Dirac operator on [Nodes, Color, Spin] fields with a
# dynamical Yukawa mass from the Higgs magnitude. Used by the vacuum finders.
# ---------------------------------------------------------------------------

class WilsonDiracOperator(nn.Module):
    def __init__(self, y: float = 1.0, bare_mass: float = 0.01, r: float = 1.0):
        super().__init__()
        self.y = y
        self.bare_mass = bare_mass
        self.r = r

        gammas, g5 = get_gamma_matrices()
        self.register_buffer('gammas', gammas)
        self.register_buffer('g5', g5)
        self.register_buffer('id_spin', torch.eye(4, dtype=torch.complex64))

    def forward(
        self,
        psi: torch.Tensor,
        phi: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        u_su3: torch.Tensor,
        is_fwd: torch.Tensor,
    ) -> torch.Tensor:
        dim_space = len(torch.unique(edge_dirs))

        # the mass is a dynamical field: bare mass + Yukawa * |phi| at each node
        phi_sq = torch.sum(phi.abs() ** 2, dim=-1)
        higgs_vev_local = torch.sqrt(phi_sq).view(-1, 1, 1)
        dynamic_mass = self.bare_mass + (self.y * higgs_vev_local)
        out = (dynamic_mass + self.r * dim_space) * psi

        src, dst = edge_index[0], edge_index[1]
        rotated_psi = torch.einsum('ecd, eds -> ecs', u_su3, psi[src])

        # Wilson projector (r -/+ gamma_mu) on forward/backward links
        g_edge = self.gammas[edge_dirs]
        sign = torch.where(is_fwd, -1.0, 1.0).view(-1, 1, 1).to(psi.device)
        P_edge = self.r * self.id_spin + sign * g_edge

        term = torch.einsum('eab, ecb -> eca', P_edge, rotated_psi)
        out.index_add_(0, dst, -0.5 * term)
        return out


def g5_spin_last(g5: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multiply g5 into a field whose LAST index is spin."""
    return torch.einsum('ab, ...b -> ...a', g5, x)


@torch.no_grad()
def conjugate_gradient(
    dirac_op: WilsonDiracOperator,
    psi: torch.Tensor,
    phi: torch.Tensor,
    edge_index: torch.Tensor,
    edge_dirs: torch.Tensor,
    u_su3: torch.Tensor,
    is_fwd: torch.Tensor,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    def apply_m_dag_m(vec: torch.Tensor) -> torch.Tensor:
        # D^dag = g5 D g5 for the vector-like operator
        d_vec = dirac_op(vec, phi, edge_index, edge_dirs, u_su3, is_fwd)
        g5_d = g5_spin_last(dirac_op.g5, d_vec)
        d_dag = dirac_op(g5_d, phi, edge_index, edge_dirs, u_su3, is_fwd)
        return g5_spin_last(dirac_op.g5, d_dag)

    return cg_solve(apply_m_dag_m, psi, tol=tol, max_iter=max_iter)


# ---------------------------------------------------------------------------
# Message-passing Wilson-Dirac operator on [Nodes, Channels, Spin, Color]
# fields for the flow pipeline (interleaved edge layout: even edges forward).
# ---------------------------------------------------------------------------

class WilsonDiracConv(MessagePassing):
    def __init__(self, kappa: float, device: torch.device | str):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.kappa = kappa
        gammas, _ = get_gamma_matrices(device)
        self.gammas = gammas
        self.I4 = torch.eye(4, dtype=torch.complex64, device=device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        u_gate: torch.Tensor,
    ) -> torch.Tensor:
        # get_pbc_edge_index interleaves links: even indices are +mu, odd are -mu
        is_fwd = torch.arange(edge_index.size(1), device=edge_index.device) % 2 == 0
        hopping = self.propagate(edge_index, x=x, edge_dirs=edge_dirs,
                                 u_gate=u_gate, is_fwd=is_fwd)
        return x - self.kappa * hopping

    def message(
        self,
        x_j: torch.Tensor,
        edge_dirs: torch.Tensor,
        u_gate: torch.Tensor,
        is_fwd: torch.Tensor,
    ) -> torch.Tensor:
        out_messages = torch.zeros_like(x_j)

        for mu in range(4):
            mask = (edge_dirs == mu)
            if not mask.any():
                continue

            # Wilson projector: (I - gamma_mu) on +mu links, (I + gamma_mu) on -mu
            sign = torch.where(is_fwd[mask], -1.0, 1.0).view(-1, 1, 1).to(self.I4.dtype)
            spin_proj = self.I4 + sign * self.gammas[mu]

            x_spin = torch.einsum('eab, ecbf -> ecaf', spin_proj, x_j[mask])
            out_messages[mask] = torch.einsum('eij, ecsj -> ecsi', u_gate[mask], x_spin)

        return out_messages


def apply_D_dag_D(
    y: torch.Tensor,
    edge_index: torch.Tensor,
    u_gate: torch.Tensor,
    edge_dirs: torch.Tensor,
    dirac_layer: WilsonDiracConv,
) -> torch.Tensor:
    _, g5 = get_gamma_matrices(y.device)

    def g5_spin(v: torch.Tensor) -> torch.Tensor:
        # spin is the second-to-last index of [N, C, S, Color] fields
        return torch.einsum('ab, ...bc -> ...ac', g5, v)

    y1 = dirac_layer(y, edge_index, edge_dirs, u_gate)
    y2 = g5_spin(y1)
    y3 = dirac_layer(y2, edge_index, edge_dirs, u_gate)
    return g5_spin(y3)


def solve_conjugate_gradient(
    dirac_layer: WilsonDiracConv,
    chi: torch.Tensor,
    edge_index: torch.Tensor,
    edge_dirs: torch.Tensor,
    u_gate: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> torch.Tensor:
    return cg_solve(
        lambda p: apply_D_dag_D(p, edge_index, u_gate, edge_dirs, dirac_layer),
        chi, tol=tol, max_iter=max_iter)
