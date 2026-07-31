import itertools

import torch

import SM
from src.physics.action_losses import get_gamma_5
from src.physics.gauge_groups import get_gamma_matrices

I4 = torch.eye(4, dtype=torch.complex64)
ZERO = torch.zeros(4, 4, dtype=torch.complex64)


def assert_clifford(gammas):
    for mu, nu in itertools.product(range(4), repeat=2):
        anti = gammas[mu] @ gammas[nu] + gammas[nu] @ gammas[mu]
        expected = 2 * I4 if mu == nu else ZERO
        assert torch.allclose(anti, expected, atol=1e-6), (mu, nu)


def test_src_gammas_satisfy_clifford_algebra():
    assert_clifford(get_gamma_matrices('cpu'))


def test_src_gamma5_consistent_with_gammas():
    gs = get_gamma_matrices('cpu')
    g5 = get_gamma_5('cpu')
    assert torch.allclose(g5, gs[0] @ gs[1] @ gs[2] @ gs[3], atol=1e-6)
    assert torch.allclose(g5 @ g5, I4, atol=1e-6)
    for g in gs:
        assert torch.allclose(g5 @ g @ g5, -g, atol=1e-6)


def test_sm_gammas_satisfy_clifford_algebra():
    gammas, g5 = SM.get_gammas()
    assert_clifford(list(gammas))
    assert torch.allclose(g5, gammas[0] @ gammas[1] @ gammas[2] @ gammas[3],
                          atol=1e-6)


def test_sm_gamma5_is_diagonal():
    # SM.py and test.py multiply by g5 via the diagonal einsum 'ss,...s->...s',
    # which is only valid while g5 is diagonal in this representation
    _, g5 = SM.get_gammas()
    off = g5 - torch.diag(torch.diagonal(g5))
    assert off.abs().max() < 1e-6
