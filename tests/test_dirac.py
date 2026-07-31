import pytest
import torch

from src.actions import pseudofermion_action
from src.dirac import WilsonDiracConv, apply_D_dag_D, get_gamma_matrices, solve_conjugate_gradient
from src.lattice import get_pbc_edge_index

L, D = 4, 2
N = L ** D


def g5_spin(x: torch.Tensor) -> torch.Tensor:
    _, g5 = get_gamma_matrices()
    return torch.einsum('ab,ncbf->ncaf', g5, x)


def rand_field(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, 2, 4, 3, dtype=torch.complex64)


@pytest.fixture(scope="module")
def operator() -> tuple:
    torch.manual_seed(0)
    ei, ed = get_pbc_edge_index(L, D, 'cpu')
    E = ei.size(1)
    h = torch.randn(E // 2, 3, 3, dtype=torch.complex64)
    u_fwd = torch.matrix_exp(1j * (h + h.mH))
    u = torch.zeros(E, 3, 3, dtype=torch.complex64)
    u[0::2] = u_fwd
    u[1::2] = u_fwd.mH
    return WilsonDiracConv(0.12, 'cpu'), ei, ed, u


def test_wilson_dirac_gamma5_hermitian(operator: tuple) -> None:
    layer, ei, ed, u = operator
    x, y = rand_field(1), rand_field(2)
    lhs = torch.sum(torch.conj(y) * layer(x, ei, ed, u))
    rhs = torch.sum(torch.conj(g5_spin(layer(g5_spin(y), ei, ed, u))) * x)
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_ddagd_hermitian_and_positive(operator: tuple) -> None:
    layer, ei, ed, u = operator
    x, y = rand_field(3), rand_field(4)
    Ax = apply_D_dag_D(x, ei, u, ed, layer)
    Ay = apply_D_dag_D(y, ei, u, ed, layer)
    assert torch.allclose(torch.sum(torch.conj(y) * Ax),
                          torch.sum(torch.conj(Ay) * x), rtol=1e-4)
    assert torch.sum(torch.conj(x) * Ax).real > 0


def test_cg_solves_the_system(operator: tuple) -> None:
    layer, ei, ed, u = operator
    chi = rand_field(5)
    Y = solve_conjugate_gradient(layer, chi, ei, ed, u, tol=1e-6, max_iter=1000)
    residual = apply_D_dag_D(Y, ei, u, ed, layer) - chi
    assert residual.abs().max() / chi.abs().max() < 1e-4


def test_pseudofermion_action_identity_and_gradient(operator: tuple) -> None:
    # For chi = D^dag eta:  S = chi^dag (D^dag D)^-1 chi = |eta|^2 exactly
    layer, ei, ed, u = operator
    eta = rand_field(6)
    chi = g5_spin(layer(g5_spin(eta), ei, ed, u)).detach().requires_grad_(True)
    S = pseudofermion_action(chi, ei, u, ed, layer)
    expected = torch.mean(torch.sum(torch.abs(eta) ** 2, dim=(1, 2, 3)))
    assert torch.allclose(S, expected, rtol=1e-3)
    S.backward()
    assert chi.grad is not None and chi.grad.abs().max() > 0
