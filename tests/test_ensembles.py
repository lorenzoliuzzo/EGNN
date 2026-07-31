import numpy as np
import torch

from src.actions import HiggsAction
from src.dirac import WilsonDiracOperator
from src.hmc import StandardModelHMC
from src.lattice import create_lattice
from src.measure import (
    ensemble_cornell_potential,
    ensemble_electroweak_masses,
    ensemble_pion_mass,
    fit_string_tension,
    jackknife_mean,
    jackknife_transformed,
    pion_correlator,
)


def test_jackknife_transformed_identity_matches_mean() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(2.0, 0.3, size=50)
    m0, e0 = jackknife_mean(v)
    m1, e1 = jackknife_transformed(v, lambda x: x)
    assert abs(m0 - m1) < 1e-12
    assert abs(e0 - e1) < 1e-12
    # a linear map scales both center and error
    m2, e2 = jackknife_transformed(v, lambda x: 3.0 * x)
    assert abs(m2 - 3 * m0) < 1e-9
    assert abs(e2 - 3 * e0) < 1e-9


def test_cornell_potential_vanishes_on_cold_ensemble() -> None:
    shape = (6, 6)
    edge_index, *_ = create_lattice(shape)
    u_id = torch.eye(3, dtype=torch.complex64).expand(edge_index.size(1), 3, 3)
    potential = ensemble_cornell_potential([u_id.clone() for _ in range(3)],
                                           shape, edge_index, r_max=2, t_fixed=2)
    for _, (v, _) in potential.items():
        assert abs(v) < 1e-4


def _identity_pion_mass(bare_mass: float) -> float:
    shape = (6, 6)
    edge_index, edge_dirs, is_fwd, _ = create_lattice(shape)
    u_id = torch.eye(3, dtype=torch.complex64).expand(edge_index.size(1), 3, 3)
    phi = torch.zeros(36, 2, dtype=torch.complex64)
    dirac = WilsonDiracOperator(y=0.0, bare_mass=bare_mass)
    corr = pion_correlator(dirac, phi, u_id, edge_index, edge_dirs, is_fwd,
                           shape, max_dist=5)
    _, mass, _ = ensemble_pion_mass([corr], fit_start=1, fit_stop=4)
    return mass


def test_pion_mass_grows_with_quark_mass() -> None:
    m_light = _identity_pion_mass(0.05)
    m_heavy = _identity_pion_mass(0.40)
    assert m_light > 0
    assert m_heavy > m_light


def test_ensemble_electroweak_masses_on_ideal_vacua() -> None:
    torch.manual_seed(0)
    shape = (5, 5)
    edge_index, _, is_fwd, partner = create_lattice(shape)
    n, E = 25, edge_index.size(1)
    v = 1.0

    phi_samples, u2_samples, u1_samples = [], [], []
    for _ in range(4):
        # ideal vacuum with a small gauge-invariant magnitude jitter
        phi = torch.zeros(n, 2, dtype=torch.complex64)
        phi[:, 1] = v * (1 + 0.02 * torch.randn(n))
        phi_samples.append(phi)
        u2_samples.append(torch.eye(2, dtype=torch.complex64).expand(E, 2, 2).clone())
        u1_samples.append(torch.ones(E, 1, 1, dtype=torch.complex64))

    results = ensemble_electroweak_masses(
        phi_samples, u2_samples, u1_samples, edge_index, is_fwd,
        HiggsAction(v=v, lam=0.5), g_su2=1.0, g_u1=0.5)

    photon_mean, _ = results['Photon']
    rho_mean, rho_err = results['rho']
    w_mean, w_err = results['W Boson']
    assert photon_mean < 1e-3
    assert abs(rho_mean - 1.0) < 0.05
    assert w_mean > 0.5 and w_err < 0.1


def test_su2_2d_string_tension_matches_area_law() -> None:
    # in 2D plaquettes quasi-decouple, so V(R) = -ln<P> * R (area law) holds
    # exactly up to torus corrections: the fitted sigma must reproduce -ln<P>
    torch.manual_seed(0)
    shape = (6, 6)
    smc = StandardModelHMC(lattice_shape=shape, groups={'su2': 2},
                           betas={'su2': 2.0}, eps=0.25, n_leapfrog=8)
    history = smc.run(n_traj=100, warmup=40, sample_every=4, log_every=0)
    u_samples = [smc.full_links(s)['su2'] for s in history['samples']]

    p_mean, _ = jackknife_mean(history['plaquette']['su2'])
    potential = ensemble_cornell_potential(u_samples, shape, smc.edge_index,
                                           r_max=2, t_fixed=2)
    sigma, sigma_err = fit_string_tension(potential)
    sigma_expected = -np.log(p_mean)
    assert abs(sigma - sigma_expected) < max(0.15, 3 * sigma_err), (
        sigma, sigma_expected, sigma_err)
