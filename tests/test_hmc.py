import pytest
import torch
from helpers import is_unitary
from scipy.special import iv

from src.hmc import StandardModelHMC, leapfrog, sample_link_momentum
from src.measure import jackknife_mean


def _pure_gauge(group: str, dim: int, beta: float, shape: tuple[int, ...],
                eps: float, n_leapfrog: int) -> StandardModelHMC:
    return StandardModelHMC(lattice_shape=shape, groups={group: dim},
                            betas={group: beta}, eps=eps, n_leapfrog=n_leapfrog)


def test_momentum_matches_kinetic_normalization() -> None:
    # density exp(-Tr(pi^2)/2) must match T = Tr(pi^2)/2: the expected kinetic
    # energy per link is (algebra dimension)/2
    torch.manual_seed(0)
    u = torch.eye(2, dtype=torch.complex64).expand(20000, 2, 2)
    pi_su2 = sample_link_momentum(u, is_su=True)
    t_su2 = 0.5 * (pi_su2.abs() ** 2).sum(dim=(1, 2)).mean().item()
    assert abs(t_su2 - 3 / 2) < 0.05  # dim su(2) = 3

    u1 = torch.ones(20000, 1, 1, dtype=torch.complex64)
    pi_u1 = sample_link_momentum(u1, is_su=False)
    t_u1 = 0.5 * (pi_u1.abs() ** 2).sum(dim=(1, 2)).mean().item()
    assert abs(t_u1 - 1 / 2) < 0.02  # dim u(1) = 1

    # traceless and hermitian
    assert torch.einsum('eii->e', pi_su2).abs().max() < 1e-5
    assert torch.allclose(pi_su2, pi_su2.mH, atol=1e-6)


def _single_trajectory_dh(eps: float, n_leapfrog: int) -> float:
    torch.manual_seed(3)
    smc = _pure_gauge('su2', 2, 2.0, (4, 4), eps, n_leapfrog)
    # warm the state away from the cold start so forces are nonzero
    for _ in range(3):
        smc.trajectory()
    torch.manual_seed(7)
    _, dh = smc.trajectory()
    return dh


def test_leapfrog_energy_conservation_scaling() -> None:
    # leapfrog is second order: halving eps (at fixed trajectory length)
    # should shrink |dH| by ~4x; allow 2x for float32 noise
    dh_coarse = _single_trajectory_dh(eps=0.20, n_leapfrog=5)
    dh_fine = _single_trajectory_dh(eps=0.10, n_leapfrog=10)
    assert abs(dh_fine) < max(0.6 * abs(dh_coarse), 5e-3)


def test_leapfrog_is_reversible() -> None:
    torch.manual_seed(0)
    smc = _pure_gauge('su2', 2, 2.0, (4, 4), eps=0.1, n_leapfrog=8)
    for _ in range(2):
        smc.trajectory()

    links0 = {k: v.clone() for k, v in smc.hmc.links.items()}
    momenta = {k: sample_link_momentum(v, smc.hmc.is_su[k])
               for k, v in links0.items()}

    fwd_links, _, fwd_mom, _, _ = leapfrog(
        links0, {}, momenta, {}, 0.1, 8, smc.hmc._forces)
    neg_mom = {k: -v for k, v in fwd_mom.items()}
    back_links, _, _, _, _ = leapfrog(
        fwd_links, {}, neg_mom, {}, 0.1, 8, smc.hmc._forces)

    for k in links0:
        assert torch.allclose(back_links[k], links0[k], atol=1e-4), k


def test_links_stay_on_the_group() -> None:
    torch.manual_seed(0)
    smc = StandardModelHMC(lattice_shape=(4, 4), groups={'su2': 2, 'u1': 1},
                           betas={'su2': 2.0, 'u1': 1.0}, eps=0.1, n_leapfrog=5)
    for _ in range(5):
        smc.trajectory()
    u2 = smc.hmc.links['su2']
    assert is_unitary(u2)
    ones = torch.ones(u2.size(0), dtype=torch.complex64)
    assert torch.allclose(torch.linalg.det(u2), ones, atol=1e-4)
    u1 = smc.hmc.links['u1']
    assert torch.allclose(u1.abs(), torch.ones_like(u1.abs()), atol=1e-4)


def test_u1_2d_plaquette_matches_exact_solution() -> None:
    # 2D U(1) lattice gauge theory is exactly solvable: <Re U_p> = I1(b)/I0(b)
    # up to torus corrections of order (I1/I0)^V, negligible here
    torch.manual_seed(0)
    beta = 1.0
    smc = _pure_gauge('u1', 1, beta, (6, 6), eps=0.3, n_leapfrog=10)
    history = smc.run(n_traj=220, warmup=40, log_every=0)

    mean, err = jackknife_mean(history['plaquette']['u1'])
    exact = iv(1, beta) / iv(0, beta)
    assert history['acceptance_rate'] > 0.6
    assert abs(mean - exact) < 0.03, (mean, exact, err)


def test_full_system_hmc_smoke() -> None:
    # gauge + Higgs + pseudofermions: trajectories run, some accept, and the
    # links stay on their manifolds
    torch.manual_seed(0)
    smc = StandardModelHMC(lattice_shape=(4, 4),
                           groups={'su3': 3, 'su2': 2, 'u1': 1},
                           betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0},
                           include_higgs=True, include_fermions=True,
                           eps=0.02, n_leapfrog=3)
    accepts = [smc.trajectory()[0] for _ in range(3)]
    assert any(accepts)
    assert is_unitary(smc.hmc.links['su3'])
    assert 'phi' in smc.hmc.scalars


@pytest.mark.parametrize("beta", [0.5])
def test_su2_strong_coupling_plaquette(beta: float) -> None:
    # leading strong-coupling expansion for SU(2): <P> = beta/4 + O(beta^3)
    torch.manual_seed(1)
    smc = _pure_gauge('su2', 2, beta, (4, 4), eps=0.25, n_leapfrog=8)
    history = smc.run(n_traj=150, warmup=50, log_every=0)
    mean, _ = jackknife_mean(history['plaquette']['su2'])
    assert abs(mean - beta / 4) < 0.05, mean
