import itertools

import torch

from src.dirac import get_chiral_projectors, get_gamma_matrices

I4 = torch.eye(4, dtype=torch.complex64)
ZERO = torch.zeros(4, 4, dtype=torch.complex64)


def test_gammas_satisfy_clifford_algebra() -> None:
    gammas, _ = get_gamma_matrices()
    for mu, nu in itertools.product(range(4), repeat=2):
        anti = gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]
        expected = 2 * I4 if mu == nu else ZERO
        assert torch.allclose(anti, expected, atol=1e-6), (mu, nu)


def test_gamma5_consistent_with_gammas() -> None:
    gammas, g5 = get_gamma_matrices()
    assert torch.allclose(g5, gammas[0] @ gammas[1] @ gammas[2] @ gammas[3], atol=1e-6)
    assert torch.allclose(g5 @ g5, I4, atol=1e-6)
    for mu in range(4):
        assert torch.allclose(g5 @ gammas[mu] @ g5, -gammas[mu], atol=1e-6)


def test_gamma5_is_diagonal() -> None:
    # g5_spin_last relies on matmul, but the chiral representation keeps g5
    # diagonal; pin it so einsum-diagonal shortcuts stay valid if reintroduced
    _, g5 = get_gamma_matrices()
    off = g5 - torch.diag(torch.diagonal(g5))
    assert off.abs().max() < 1e-6


def test_chiral_projectors() -> None:
    _, g5 = get_gamma_matrices()
    P_L, P_R = get_chiral_projectors(g5)
    assert torch.allclose(P_L + P_R, I4, atol=1e-6)
    assert torch.allclose(P_L @ P_L, P_L, atol=1e-6)
    assert torch.allclose(P_R @ P_R, P_R, atol=1e-6)
    assert torch.allclose(P_L @ P_R, ZERO, atol=1e-6)
