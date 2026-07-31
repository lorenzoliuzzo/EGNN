"""Electroweak boson masses on an HMC ensemble: sample the SU(2)xU(1)+Higgs
system, rotate every configuration to unitary gauge, and jackknife the
W/Z/photon masses, the rho parameter, and the VEV over the ensemble."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.actions import HiggsAction  # noqa: E402
from src.hmc import StandardModelHMC  # noqa: E402
from src.measure import ensemble_electroweak_masses  # noqa: E402


def main(lattice_shape: tuple[int, ...] = (6, 6), beta_su2: float = 4.0,
         beta_u1: float = 5.0, v: float = 1.0, lam: float = 0.5,
         n_traj: int = 200, warmup: int = 80, sample_every: int = 5) -> None:
    torch.manual_seed(0)

    smc = StandardModelHMC(
        lattice_shape=lattice_shape,
        groups={'su2': 2, 'u1': 1},
        betas={'su2': beta_su2, 'u1': beta_u1},
        v=v, lam=lam, include_higgs=True,
        eps=0.1, n_leapfrog=10,
    )
    history = smc.run(n_traj=n_traj, warmup=warmup, sample_every=sample_every)
    samples = history['samples']
    print(f"\nEnsemble: {len(samples)} configs | "
          f"acceptance {history['acceptance_rate']:.2f}")

    phi_samples = [s['phi'] for s in samples]
    u2_samples = [smc.full_links(s)['su2'] for s in samples]
    u1_samples = [smc.full_links(s)['u1'] for s in samples]

    # lattice mapping: beta_su2 = 4/g^2, beta_u1 = 1/g'^2
    g = np.sqrt(4.0 / beta_su2)
    gp = np.sqrt(1.0 / beta_u1)
    theta_w = np.degrees(np.arctan(gp / g))

    results = ensemble_electroweak_masses(
        phi_samples, u2_samples, u1_samples, smc.edge_index, smc.is_fwd,
        HiggsAction(v=v, lam=lam), g_su2=g, g_u1=gp)

    print(f"\nWeinberg angle (tree level): {theta_w:.2f} deg")
    for key in ('W Boson', 'Z Boson', 'Photon', 'rho', 'vev'):
        mean, err = results[key]
        print(f"{key:8}: {mean:.4f} +/- {err:.4f}")

    mw, _ = results['W Boson']
    mz, _ = results['Z Boson']
    print(f"M_W / M_Z = {mw / mz:.4f} (theory: cos(theta_W) = "
          f"{np.cos(np.radians(theta_w)):.4f})")


if __name__ == "__main__":
    main()
