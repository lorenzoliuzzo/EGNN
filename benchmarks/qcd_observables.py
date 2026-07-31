"""Quenched QCD observables on an HMC ensemble: sample pure-gauge SU(3)
configurations, then measure the Cornell potential, pion correlator, chiral
condensate, and the GMOR relation with jackknife errors.

In 2D the area law is exact (plaquettes quasi-decouple), so the fitted string
tension sigma is compared against -ln<P> as a quantitative cross-check."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.dirac import WilsonDiracOperator  # noqa: E402
from src.hmc import StandardModelHMC  # noqa: E402
from src.measure import (  # noqa: E402
    chiral_condensate_sample,
    ensemble_chiral_condensate,
    ensemble_cornell_potential,
    ensemble_gmor,
    ensemble_pion_mass,
    fit_string_tension,
    jackknife_mean,
    pion_correlator,
)

PLOTS = Path(__file__).parent.parent / "plots"


def main(lattice_shape: tuple[int, ...] = (8, 8), beta_su3: float = 6.0,
         n_traj: int = 160, warmup: int = 60, sample_every: int = 5,
         bare_mass: float = 0.10) -> None:
    PLOTS.mkdir(exist_ok=True)
    torch.manual_seed(0)

    smc = StandardModelHMC(lattice_shape=lattice_shape, groups={'su3': 3},
                           betas={'su3': beta_su3}, eps=0.15, n_leapfrog=10)
    history = smc.run(n_traj=n_traj, warmup=warmup, sample_every=sample_every)
    u_samples = [smc.full_links(s)['su3'] for s in history['samples']]
    print(f"\nEnsemble: {len(u_samples)} configs | "
          f"acceptance {history['acceptance_rate']:.2f}")

    p_mean, p_err = jackknife_mean(history['plaquette']['su3'])
    print(f"<P> = {p_mean:.4f} +/- {p_err:.4f}")

    phi_zero = smc._phi_background
    dirac = WilsonDiracOperator(y=0.0, bare_mass=bare_mass)

    # --- Cornell potential + string tension vs the 2D area-law prediction ---
    t_fixed = 2
    potential = ensemble_cornell_potential(u_samples, lattice_shape,
                                           smc.edge_index, r_max=3, t_fixed=t_fixed)
    sigma, sigma_err = fit_string_tension(potential)
    sigma_area_law = -np.log(max(p_mean, 1e-12))
    print("\nCornell potential (T = 2):")
    for r, (v, e) in potential.items():
        print(f"  V({r}) = {v:.4f} +/- {e:.4f}")
    print(f"string tension sigma = {sigma:.4f} +/- {sigma_err:.4f} "
          f"| 2D area law -ln<P> = {sigma_area_law:.4f}")

    # --- Pion correlator and mass ---
    correlators = [
        pion_correlator(dirac, phi_zero, u, smc.edge_index, smc.edge_dirs,
                        smc.is_fwd, lattice_shape, max_dist=6)
        for u in u_samples
    ]
    mean_corr, m_pi, m_pi_err = ensemble_pion_mass(correlators)
    print(f"\npion mass m_pi = {m_pi:.4f} +/- {m_pi_err:.4f} "
          f"(bare quark mass {bare_mass})")

    # --- Chiral condensate ---
    torch.manual_seed(1)
    cond_vals = [
        chiral_condensate_sample(dirac, phi_zero, u, smc.edge_index,
                                 smc.edge_dirs, smc.is_fwd)
        for u in u_samples
    ]
    cond, cond_err = ensemble_chiral_condensate(cond_vals)
    print(f"<psi_bar psi> = {cond:.4f} +/- {cond_err:.4f}")

    # --- GMOR: m_pi(m_q) on the same ensemble ---
    gmor = ensemble_gmor(u_samples, phi_zero, lattice_shape, smc.edge_index,
                         smc.edge_dirs, smc.is_fwd,
                         quark_masses=[0.05, 0.10, 0.15, 0.20, 0.25])
    print("\nGMOR sweep:")
    for mq, m, e in gmor:
        print(f"  m_q = {mq:.2f} -> m_pi = {m:.4f} +/- {e:.4f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rs = list(potential.keys())
    vs = [potential[r][0] for r in rs]
    es = [potential[r][1] for r in rs]
    axes[0].errorbar(rs, vs, yerr=es, fmt='o-', capsize=3, label='ensemble V(R)')
    axes[0].plot(rs, [sigma_area_law * r for r in rs], 'r--',
                 label=r'2D area law $-\ln\langle P\rangle \cdot R$')
    axes[0].set_title("Static potential (Cornell)")
    axes[0].set_xlabel("R")
    axes[0].set_ylabel("V(R)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    rr = list(mean_corr.keys())
    axes[1].errorbar(rr, np.log(list(mean_corr.values())), fmt='o-',
                     label='ln <C(r)>')
    axes[1].set_title(rf"Pion correlator ($m_\pi$ = {m_pi:.3f} $\pm$ {m_pi_err:.3f})")
    axes[1].set_xlabel("r")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    mqs = [g[0] for g in gmor]
    mpi_sq = [g[1] ** 2 for g in gmor]
    mpi_sq_err = [2 * g[1] * g[2] for g in gmor]
    axes[2].errorbar(mqs, mpi_sq, yerr=mpi_sq_err, fmt='bo-', capsize=3)
    slope, intercept = np.polyfit(mqs, mpi_sq, 1)
    axes[2].plot(mqs, [slope * m + intercept for m in mqs], 'r--',
                 label=f"fit: {slope:.2f} m_q + {intercept:.3f}")
    axes[2].set_title("GMOR relation on a fixed ensemble")
    axes[2].set_xlabel(r"$m_q$")
    axes[2].set_ylabel(r"$m_\pi^2$")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(PLOTS / "qcd_ensemble_observables.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
