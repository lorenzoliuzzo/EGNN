"""Langevin-thermalize the heterogeneous Standard Model GNN and plot the
per-sector action landscape."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting import plot_sm_action_landscape  # noqa: E402
from src.standard_model import HeteroQuantumVacuum  # noqa: E402

PLOTS = Path(__file__).parent.parent / "plots"


def main(lattice_shape: tuple[int, ...] = (16, 16), steps: int = 1000,
         thermalize_steps: int = 200) -> None:
    PLOTS.mkdir(exist_ok=True)

    model = HeteroQuantumVacuum(lattice_shape=lattice_shape)
    history = model.find_vacuum(steps=steps, thermalize_steps=thermalize_steps)
    plot_sm_action_landscape(history, PLOTS / "sm_action_landscape.png", smooth=True)


if __name__ == "__main__":
    main()
