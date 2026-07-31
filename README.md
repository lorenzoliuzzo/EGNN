# EGNN — Lattice Standard Model with Gauge-Equivariant GNNs

Simulations of lattice field theory where message passing *is* parallel
transport: SU(3)xSU(2)xU(1) gauge links live on graph edges, matter fields on
nodes, and every physics operator (Wilson-Dirac, covariant Laplacian, Wilson
action) is a gauge-equivariant graph operation.

Three pipelines share one physics core (`src/`):

- **Vacuum finders** (`src/vacuum.py`) — cool or Langevin-thermalize
  SU(3)xSU(2)xU(1) + Higgs + pseudofermion configurations, then measure
  confinement (Cornell potential), the pion correlator, the chiral condensate,
  and the GMOR relation.
- **Electroweak sector** (`benchmarks/electroweak_masses.py`) — cool the
  SU(2)xU(1) Higgs vacuum, rotate to unitary gauge, and extract W/Z/photon
  masses, the Weinberg angle, and the rho parameter.
- **Heterogeneous SM GNN** (`src/standard_model.py`, `src/models.py`,
  `src/main.py`) — one message-passing operator per particle species (18
  fermions + Higgs) with per-species hypercharge gates, plus an equivariant
  normalizing-flow pipeline over pre-generated gauge ensembles.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m pytest -q                        # physics invariants test suite
python benchmarks/electroweak_masses.py    # cool + measure W/Z/photon masses
python benchmarks/qcd_observables.py       # confinement + chiral observables
python -m src.make_dataset                 # generate flow-training data
python -m src.main                         # train the flow pipeline
```

## Workflow

Work moves research -> plan -> actions: investigations land as dated briefs in
`research/`, roadmaps as dated checkbox plans in `plans/` (only measured
numbers), and every figure or number cited anywhere is reproducible by a script
in `benchmarks/`. See `CLAUDE.md` for layout contracts and conventions, and
`report/` for the Typst write-up.
