"""Thermalize the standalone SU(3)xSU(2)xU(1)+Higgs vacuum with Langevin
dynamics, then measure confinement and chiral observables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.measure import (  # noqa: E402
    measure_chiral_condensate,
    measure_cornell_potential,
    measure_pion_mass,
    measure_rho_parameter,
    run_gmor_test,
)
from src.plotting import plot_qcd_action_landscape  # noqa: E402
from src.vacuum import QuantumVacuumFinder  # noqa: E402

PLOTS = Path(__file__).parent.parent / "plots"


def main(lattice_shape: tuple[int, ...] = (16, 16), steps: int = 1000) -> None:
    PLOTS.mkdir(exist_ok=True)

    model = QuantumVacuumFinder(
        lattice_shape=lattice_shape,
        groups={'su3': 3, 'su2': 2, 'u1': 1},
        betas={'su3': 6.0, 'su2': 4.0, 'u1': 5.0},
    )
    history = model.find_vacuum(steps=steps)
    plot_qcd_action_landscape(history, PLOTS / "actions.png")

    measure_rho_parameter(model)

    potential = measure_cornell_potential(model, r_max=5)
    r_values = list(potential.keys())
    v_values = list(potential.values())

    plt.figure(figsize=(8, 5))
    plt.plot(r_values, v_values, 'o-', label="Simulated Potential")
    slope, intercept = np.polyfit(r_values[1:], v_values[1:], 1)
    plt.plot(r_values, [slope * x + intercept for x in r_values], '--',
             label=f"String Tension (sigma) ~ {slope:.3f}")
    plt.xlabel("Distance R (Lattice Units)")
    plt.ylabel("Potential V(R)")
    plt.title("SU(3) Quark Confinement (Cornell Potential)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS / "confinement.png")
    plt.close()

    pion_signal = measure_pion_mass(model, max_dist=12)
    r_vals = list(pion_signal.keys())
    log_c_vals = np.log(list(pion_signal.values()))

    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, log_c_vals, 'o-', color='red', label="ln(Correlator)")
    # skip r = 1, 2 to dodge short-range lattice artifacts
    fit_start = 2
    slope, intercept = np.polyfit(r_vals[fit_start:], log_c_vals[fit_start:], 1)
    plt.plot(r_vals, [slope * x + intercept for x in r_vals], 'k--',
             label=f"Fit Mass (m_pi) ~ {-slope:.3f}")
    plt.xlabel("Distance r (Lattice Units)")
    plt.ylabel("ln(C(r))")
    plt.title("Pion Mass Extraction (Quark Propagator Decay)")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS / "pion_mass.png")
    plt.close()

    measure_chiral_condensate(model)

    test_masses, pion_masses = run_gmor_test(model)
    m_pi_sq = [m ** 2 for m in pion_masses]

    plt.figure(figsize=(7, 5))
    plt.plot(test_masses, m_pi_sq, 'bo-', label=r"$m_\pi^2$")
    slope, intercept = np.polyfit(test_masses, m_pi_sq, 1)
    plt.plot(test_masses, [slope * x + intercept for x in test_masses], 'r--',
             label=f"Fit: y = {slope:.2f}x + {intercept:.3f}")
    plt.xlabel("Input Quark Mass ($m_q$)")
    plt.ylabel(r"Pion Mass Squared ($m_\pi^2$)")
    plt.title("GMOR Relation Verification")
    plt.grid(True)
    plt.legend()
    plt.savefig(PLOTS / "gmor.png")
    plt.close()


if __name__ == "__main__":
    main()
