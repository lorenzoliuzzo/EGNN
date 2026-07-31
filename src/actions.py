import torch
import torch.nn as nn

from src.dirac import (
    WilsonDiracConv,
    WilsonDiracOperator,
    conjugate_gradient,
    g5_spin_last,
    solve_conjugate_gradient,
)


class WilsonAction(nn.Module):
    """S_W = beta * sum_p (1 - Re Tr(U_p) / N) over closed loops of edge gates."""

    def __init__(self, plaq_idx: torch.Tensor, group_dim: int, beta: float):
        super().__init__()
        self.register_buffer('plaq_idx', plaq_idx)
        self.group_dim = group_dim
        self.beta = beta

    def forward(self, u_gates: torch.Tensor) -> torch.Tensor:
        u_p = u_gates[self.plaq_idx]

        # Compose in reverse path order: U_e psi transports column vectors, so
        # the closed-loop transport is U_ek ... U_e1 and its trace telescopes
        # under a gauge transformation U'_e = G_dst U_e G_src^dag
        loop = u_p[:, 0]
        for i in range(1, u_p.shape[1]):
            loop = torch.matmul(u_p[:, i], loop)
        tr = torch.einsum('pii -> p', loop)

        action_per_plaq = 1.0 - (tr.real / float(self.group_dim))
        return self.beta * torch.sum(action_per_plaq)


def wilson_plaquette_loss(
    u_gate: torch.Tensor,
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor,
) -> torch.Tensor:
    N = u_gate.size(-1)
    u1, u2 = u_gate[p1], u_gate[p2]
    u3, u4 = u_gate[p3], u_gate[p4]

    # closed loop n00 -> n10 -> n11 -> n01 -> n00 in reverse path order
    # (p3/p4 are stored as forward edges, hence the adjoints)
    u_p = u4.mH @ u3.mH @ u2 @ u1
    tr_u_p = torch.real(torch.einsum('...ii->...', u_p))
    return torch.mean(1.0 - (tr_u_p / N))


class HiggsAction(nn.Module):
    """V = lam * (|phi|^2 - v^2)^2 plus the covariant kinetic term."""

    def __init__(self, v: float = 1.0, lam: float = 0.1):
        super().__init__()
        self.v = v
        self.lam = lam

    def forward(
        self,
        phi: torch.Tensor,
        edge_index: torch.Tensor,
        is_fwd: torch.Tensor,
        u_su2: torch.Tensor,
        u_u1: torch.Tensor,
    ) -> torch.Tensor:
        phi_sq = torch.sum(phi.abs() ** 2, dim=-1)
        v_pot = self.lam * torch.sum((phi_sq - self.v ** 2) ** 2)
        return _higgs_kinetic(phi, edge_index, is_fwd, u_su2, u_u1) + v_pot


class ElectroweakHiggsAction(nn.Module):
    """V = -mu^2 |phi|^2 + lam |phi|^4 plus the covariant kinetic term."""

    def __init__(self, mu_sq: float = 1.0, lam: float = 0.5):
        super().__init__()
        self.mu_sq = mu_sq
        self.lam = lam

    def forward(
        self,
        phi: torch.Tensor,
        edge_index: torch.Tensor,
        is_fwd: torch.Tensor,
        u_su2: torch.Tensor,
        u_u1: torch.Tensor,
    ) -> torch.Tensor:
        phi_sq = torch.sum(phi.abs() ** 2, dim=-1)
        v_pot = torch.sum(-self.mu_sq * phi_sq + self.lam * (phi_sq ** 2))
        return _higgs_kinetic(phi, edge_index, is_fwd, u_su2, u_u1) + v_pot


def _higgs_kinetic(
    phi: torch.Tensor,
    edge_index: torch.Tensor,
    is_fwd: torch.Tensor,
    u_su2: torch.Tensor,
    u_u1: torch.Tensor,
) -> torch.Tensor:
    # each physical link counted once: forward edges only
    src, dst = edge_index[0, is_fwd], edge_index[1, is_fwd]
    u_total = u_su2[is_fwd] * u_u1[is_fwd]
    phi_transported = torch.einsum('eij, ej -> ei', u_total, phi[src])
    diff = phi[dst] - phi_transported
    return torch.sum(diff.abs() ** 2)


def covariant_kinetic_loss(
    phi: torch.Tensor, edge_index: torch.Tensor, u_gate: torch.Tensor,
) -> torch.Tensor:
    phi_j = phi[edge_index[0]]
    phi_i = phi[edge_index[1]]
    transported_j = torch.einsum('enm,ecm->ecn', u_gate, phi_j)
    diff_mag_sq = torch.sum(torch.abs(phi_i - transported_j) ** 2, dim=-1)
    return torch.mean(diff_mag_sq)


def higgs_potential_loss(
    phi_higgs: torch.Tensor, v_target: float, lambda_coupling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mag_sq = torch.sum(torch.abs(phi_higgs) ** 2, dim=-1)
    # physical convention: the minimum sits at |phi|^2 = v^2 / 2
    local_potential = lambda_coupling * (mag_sq - (v_target ** 2) / 2.0) ** 2
    return torch.mean(local_potential), torch.mean(mag_sq)


def sm_yukawa_loss(
    psi_L: torch.Tensor,
    psi_R: torch.Tensor,
    phi_higgs: torch.Tensor,
    yukawa_matrix: torch.Tensor,
) -> torch.Tensor:
    # contracting the SU(2) index makes (L_bar Phi) a gauge singlet
    l_bar_phi = torch.einsum('nci,nci->nc', torch.conj(psi_L), phi_higgs)
    interaction = yukawa_matrix * l_bar_phi * psi_R.squeeze(-1)
    lagrangian_term = interaction + torch.conj(interaction)
    return -torch.mean(torch.real(lagrangian_term))


class PseudofermionAction(nn.Module):
    """S_pf = pf^dag (D^dag D)^-1 pf for the standalone Wilson-Dirac operator."""

    def __init__(self, y: float = 1.0):
        super().__init__()
        self.dirac = WilsonDiracOperator(y=y)

    def forward(
        self,
        pf_phi: torch.Tensor,
        higgs_phi: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        u_su3: torch.Tensor,
        is_fwd: torch.Tensor,
    ) -> torch.Tensor:
        x = conjugate_gradient(self.dirac, pf_phi, higgs_phi, edge_index,
                               edge_dirs, u_su3, is_fwd)
        actual_action = torch.sum(pf_phi.conj() * x).real

        # surrogate trick: -|Dx|^2 has the gradient of the true action while x
        # stays a detached CG solution; the detach cancels its value
        dx = self.dirac(x, higgs_phi, edge_index, edge_dirs, u_su3, is_fwd)
        surrogate_loss = -torch.sum(dx.abs() ** 2)
        return surrogate_loss - surrogate_loss.detach() + actual_action


def pseudofermion_action(
    chi: torch.Tensor,
    edge_index: torch.Tensor,
    u_gate: torch.Tensor,
    edge_dirs: torch.Tensor,
    dirac_layer: WilsonDiracConv,
) -> torch.Tensor:
    with torch.no_grad():
        Y = solve_conjugate_gradient(dirac_layer, chi, edge_index, edge_dirs, u_gate)

    # with Y detached, d(chi^dag Y)/d chi* = Y = (D^dag D)^-1 chi is the exact
    # gradient, so the action backpropagates correctly through chi
    action = torch.einsum('ncsi, ncsi -> n', torch.conj(chi), Y)
    return torch.mean(torch.real(action))


__all__ = [
    'WilsonAction', 'wilson_plaquette_loss', 'HiggsAction',
    'ElectroweakHiggsAction', 'covariant_kinetic_loss', 'higgs_potential_loss',
    'sm_yukawa_loss', 'PseudofermionAction', 'pseudofermion_action',
    'g5_spin_last',
]
