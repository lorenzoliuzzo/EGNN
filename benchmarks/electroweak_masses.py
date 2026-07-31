"""Cool the SU(2)xU(1)+Higgs vacuum, rotate to unitary gauge, and measure the
W/Z/photon masses, the Weinberg angle, and the rho parameter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from src.groups import get_gate  # noqa: E402
from src.measure import align_to_unitary_gauge, measure_electroweak_masses  # noqa: E402
from src.plotting import plot_cooling_landscape  # noqa: E402
from src.vacuum import ElectroweakSimulator  # noqa: E402

PLOTS = Path(__file__).parent.parent / "plots"


def main(lattice_shape: tuple[int, ...] = (8, 8, 8),
         beta_su2: float = 4.0, beta_u1: float = 5.0,
         steps: int = 600) -> None:
    PLOTS.mkdir(exist_ok=True)

    sim = ElectroweakSimulator(lattice_shape=lattice_shape, mu_sq=2.0, lam=0.5,
                               beta_su2=beta_su2, beta_u1=beta_u1)

    print("Beginning Vacuum Cooling...")
    history = sim.cool_vacuum(steps=steps, lr=0.01)
    plot_cooling_landscape(history, PLOTS / "action_landscape.png")

    phi_final = sim.phi.detach()
    u_su2_final = get_gate(sim.u_su2, sim.partner_map, sim.is_fwd, is_su=True).detach()
    u_u1_final = get_gate(sim.u_u1, sim.partner_map, sim.is_fwd, is_su=False).detach()

    phi_aligned, u_su2_aligned, u_u1_aligned = align_to_unitary_gauge(
        phi_final, sim.edge_index, u_su2_final, u_u1_final)

    # lattice mapping: beta_su2 = 4/g^2, beta_u1 = 1/g'^2
    g = np.sqrt(4.0 / beta_su2)
    gp = np.sqrt(1.0 / beta_u1)

    stats = measure_electroweak_masses(
        phi_aligned, u_su2_aligned, u_u1_aligned,
        sim.edge_index, sim.is_fwd, sim.h_calc, g_su2=g, g_u1=gp)

    print(f"\nFinal VEV: {history['vev'][-1]:.4f}")
    print(f"Final Rho: {stats['rho']:.6f}")


if __name__ == "__main__":
    main()
