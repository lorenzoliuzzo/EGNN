# EGNN — lattice Standard Model, agent instructions

Gauge-equivariant GNN simulations of lattice field theory: SU(3)xSU(2)xU(1)
gauge links, Wilson-Dirac fermions with pseudofermion CG, Higgs symmetry
breaking, and Langevin/optimizer vacuum search, plus a flow-based pipeline that
learns matter fields on fixed gauge ensembles.

## The hard constraints

1. **Gauge invariance is the grading criterion of this codebase.** Every action
   term must be invariant under `U'_e = G_dst U_e G_src^dag` with `phi' = G phi`
   (the message-passing transport convention). `tests/test_actions.py` and
   `tests/test_electroweak.py` pin this; they must stay green.
2. **The Dirac operator must satisfy `D^dag = g5 D g5`** and `D^dag D` must be
   positive — otherwise the CG pseudofermion action is meaningless. Pinned by
   `tests/test_dirac.py` and `tests/test_sampling.py`.
3. **Layout contracts are load-bearing.** `create_lattice` orders edges in
   (direction, orientation) blocks (`find_rectangular_loops` indexes into it
   arithmetically); `get_pbc_edge_index` interleaves links with even indices
   forward (the on-disk datasets store gauge links in that order). Pinned by
   `tests/test_lattice.py`. Never change an ordering without migrating both.

## Layout

- `src/lattice.py` — lattice graphs, Wilson-loop and plaquette indexing.
- `src/groups.py` — group projections (`get_gate`), generators, Kronecker gates.
- `src/dirac.py` — gamma matrices (single convention: chiral, `g5 = diag(I,-I)`),
  Wilson-Dirac operators (standalone and message-passing), generic CG.
- `src/actions.py` — Wilson, Higgs, and pseudofermion actions/losses.
- `src/hmc.py` — manifold HMC: Lie-algebra momenta, autograd forces (note the
  torch complex-grad convention documented there), leapfrog + Metropolis.
  `StandardModelHMC` covers pure gauge, gauge+Higgs, and the full system.
- `src/vacuum.py` — standalone vacuum finders (Adam cooling, Langevin; the
  Langevin sampler is unadjusted and biased — HMC is the exact sampler).
- `src/standard_model.py` — the heterogeneous SM GNN and its Langevin sampler.
  Physics caveat documented there: the chirality-split electroweak transport
  plus a Wilson term is not an exactly gauge-invariant chiral lattice theory
  (Nielsen-Ninomiya); quantitative claims rest on the vector-like sectors.
- `src/models.py` — flow-pipeline layers and `SM_HeteroGNN`.
- `src/measure.py` — observables: unitary-gauge alignment, boson masses,
  Cornell potential, pion correlator, condensate, GMOR, jackknife.
- `src/plotting.py` — all figure generation.
- `src/main.py` — flow-training entry point (`python -m src.main`).
- `src/make_dataset.py` — random configuration generator (`python -m src.make_dataset`).
- `benchmarks/` — the scripts every plotted/cited number comes from. If a plan
  or the report cites a measurement, a script here must reproduce it.
- `plans/` — dated roadmaps, `YYYY-MM-DD-slug.md`. Update checkboxes and the
  `Status:` line of the active plan as work lands rather than rewriting history.
  Numbers in a plan must come from a run or benchmark we actually did.
- `research/` — dated briefs recording investigations and their evidence.
- `report/` — Typst sources. Don't edit unless asked.
- `tests/` — pytest.

`src/` is a namespace package: imports are `from src.lattice import ...`;
`pyproject.toml` puts the repo root on `sys.path` for pytest. Benchmarks are
standalone scripts and add the root themselves.

## Conventions

- Python 3.12+. Type hints on every function signature, tests included.
- `pathlib` over `os.path`.
- Comment *why*, never *what*. No docstrings unless they state a physics
  convention the signature cannot.
- Lint and format with `ruff`; leave `ruff check .` clean.
- Tests are `pytest`, plain `def test_*() -> None:` functions, no classes.

## Testing without a GPU or datasets

`data/`, `sm_dataset/`, and `.venv/` are gitignored and absent in CI, so **no
test may read them or require CUDA**. Test physics on small CPU lattices
(4x4-ish) with random fields; identities (gauge invariance, g5-hermiticity,
S_pf = |eta|^2) beat golden values. A test that needs a GPU or a dataset is the
wrong test.

Run: `python -m pytest -q`
