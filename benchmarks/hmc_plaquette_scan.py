"""Validation milestone 1: HMC average plaquette vs. beta against known
references — exact I1/I0 for 2D U(1), strong/weak coupling expansions for
SU(2) — with jackknife error bars."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import iv  # noqa: E402

from src.hmc import StandardModelHMC  # noqa: E402
from src.measure import integrated_autocorrelation_time, jackknife_mean  # noqa: E402

PLOTS = Path(__file__).parent.parent / "plots"


def scan(group: str, dim: int, betas: list[float], shape: tuple[int, ...],
         n_traj: int, warmup: int) -> tuple[list[float], list[float]]:
    means, errs = [], []
    # a distinct seed per beta: reseeding to the same value made every beta
    # consume the same random stream, correlating the points (r ~ 0.3-0.7) so
    # that one common-mode fluctuation looked like a systematic deviation
    for i, beta in enumerate(betas):
        torch.manual_seed(1000 + i)
        # start coarse; run() auto-tunes eps during warmup toward ~70-85%
        # acceptance for each (beta, volume)
        smc = StandardModelHMC(lattice_shape=shape, groups={group: dim},
                               betas={group: beta}, eps=0.2, n_leapfrog=10)
        history = smc.run(n_traj=n_traj, warmup=warmup, log_every=0)
        mean, err = jackknife_mean(history['plaquette'][group])
        tau, _ = integrated_autocorrelation_time(history['plaquette'][group])
        means.append(mean)
        errs.append(err)
        print(f"{group.upper()} beta={beta:5.2f} | <P> = {mean:.4f} +/- {err:.4f} "
              f"| tau_int = {tau:5.2f} | acc = {history['acceptance_rate']:.2f} "
              f"| eps -> {smc.hmc.eps:.3f}")
    return means, errs


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- U(1) in 2D: exactly solvable ---
    betas_u1 = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    # 250 trajectories left a ~0.014 per-run scatter, wide enough that single
    # points wandered 2 sigma off the exact curve; 2000 cuts that ~3x
    means, errs = scan('u1', 1, betas_u1, (6, 6), n_traj=2000, warmup=200)
    bb = np.linspace(0.05, 3.2, 200)
    axes[0].errorbar(betas_u1, means, yerr=errs, fmt='o', capsize=3, label='HMC')
    axes[0].plot(bb, iv(1, bb) / iv(0, bb), 'r--', label=r'exact $I_1(\beta)/I_0(\beta)$')
    axes[0].set_title("2D U(1): average plaquette vs exact solution")
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel(r"$\langle \mathrm{Re}\, U_p \rangle$")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # --- SU(2) in 3D: strong/weak coupling envelopes ---
    betas_su2 = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    means, errs = scan('su2', 2, betas_su2, (4, 4, 4), n_traj=200, warmup=60)
    bb = np.linspace(0.2, 6.5, 200)
    axes[1].errorbar(betas_su2, means, yerr=errs, fmt='o', capsize=3, label='HMC')
    axes[1].plot(bb, bb / 4, 'g--', alpha=0.8, label=r'strong coupling $\beta/4$')
    axes[1].plot(bb, 1 - 3 / (4 * bb), 'r--', alpha=0.8, label=r'weak coupling $1 - 3/(4\beta)$')
    axes[1].set_ylim(0, 1)
    axes[1].set_title(r"3D SU(2): average plaquette vs $\beta$")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS / "hmc_plaquette_scan.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
