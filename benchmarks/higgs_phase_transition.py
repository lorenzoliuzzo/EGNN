"""Sweep the Higgs mass parameter mu^2 to locate the electroweak phase
transition and compare the settled VEV to the tree-level prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.vacuum import ElectroweakSimulator  # noqa: E402

PLOTS = Path(__file__).parent.parent / "plots"


def main(lattice_shape: tuple[int, ...] = (8, 8), lam: float = 0.5,
         steps: int = 300) -> None:
    PLOTS.mkdir(exist_ok=True)

    mu_sq_vals = np.linspace(-1.5, 3.0, 20)
    vevs = []

    print("Executing Phase Transition Sweep...")
    for mu in mu_sq_vals:
        sim = ElectroweakSimulator(lattice_shape=lattice_shape, mu_sq=mu, lam=lam)
        hist = sim.cool_vacuum(steps=steps, lr=0.02)
        vevs.append(hist['vev'][-1])
        print(f"mu^2: {mu: .2f} | Final VEV: {vevs[-1]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(mu_sq_vals, vevs, 'bo-', linewidth=2, markersize=6)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5,
                label=r'Critical Point $\mu^2 = 0$')

    # tree level: v = sqrt(mu^2 / (2 lambda)) in the broken phase
    theoretical_v = [np.sqrt(max(0, m) / (2 * lam)) for m in mu_sq_vals]
    plt.plot(mu_sq_vals, theoretical_v, 'r--', label='Theoretical Prediction')

    plt.title('Electroweak Phase Transition via GNN Optimization')
    plt.xlabel(r'Mass Parameter $\mu^2$')
    plt.ylabel(r'Vacuum Expectation Value $\langle |\phi| \rangle$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOTS / "phase_transition.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
