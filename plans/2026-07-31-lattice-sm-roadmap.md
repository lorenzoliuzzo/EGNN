# Lattice Standard Model roadmap

**Status:** Phases 0–2 done (correctness review applied, consolidated `src/`
layout, 32 tests green, ruff clean); Phase 3 (sampling) not started; Phase 4
direction undecided · **Last updated:** 2026-07-31

Everything checked below is pinned by `tests/` or reproducible by a script in
`benchmarks/`. Nothing enters this file that we have not run.

---

## Phase 0 — Make it a project

- [x] git repository, public at `lorenzoliuzzo/EGNN`
- [x] datasets (136 GB) gitignored; regenerable via `src/make_dataset.py`
- [x] one codebase: physics core consolidated into `src/` (lattice, groups,
      dirac, actions, vacuum, standard_model, models, measure, plotting);
      duplicated `create_lattice`/`get_gate`/`WilsonAction`/gamma copies and all
      dead `*2` variants deleted (recoverable from git)
- [x] workflow structure adapted from `supervised-learning-on-food-images`:
      `research/` briefs, dated plans here, `benchmarks/` drivers, `report/`,
      `CLAUDE.md`, `pyproject.toml`, CPU-only CI

## Phase 1 — Correctness

- [x] 12 confirmed defects fixed and verified numerically — see
      `research/2026-07-31-correctness-review-brief.md`
- [x] gauge invariance of Wilson and Higgs actions verified to 1e-4 under
      random SU(2) transformations; cold vacuum S = 0 preserved
- [x] Wilson-Dirac operator gamma5-hermitian, `D^dag D` positive, CG converges
- [x] pseudofermion identity `S_pf = |eta|^2` exact within CG tolerance

## Phase 2 — Test suite

- [x] pytest suite (32 tests, ~15 s CPU) covering: gamma algebra, both lattice
      layout contracts, loop/plaquette closure, gate projections, Dirac/CG,
      action invariances, unitary-gauge alignment, ideal-vacuum W/Z/photon
      masses with rho ~ 1, heat-bath identity, flow-layer equivariance
- [x] CI: ruff + pytest on CPU-only torch

## Phase 3 — Sampling correctness (next)

- [ ] Metropolis-adjusted Langevin (MALA) or HMC; the current unadjusted
      Euler–Maruyama step has O(dt) bias and ignores `training.temperature`
- [ ] updates in the Lie algebra (project force to the tangent space,
      exponentiate) instead of perturbing raw pre-projection tensors
- [ ] ensemble measurement pipeline: decorrelated post-thermalization samples,
      jackknife error bars (`src/measure.py:jackknife_effective_mass`) on every
      plotted observable — no more single-configuration "measurements"

## Phase 4 — ML direction (decision pending)

Two coherent options; pick one before writing code:

- [ ] **(A) Normalizing-flow sampler**: train toward `e^{-S}` on fixed gauge
      ensembles (reverse-KL with the flow log-det; surrogate-gradient CG)
- [ ] **(B) Physics-first**: gauge links as parameters + HMC; the GNN is the
      equivariant Dirac operator (message passing = parallel transport)

Either way: AMP with complex tensors needs validation before trusting a run
(currently enabled in `src/main.py`, inherited behavior).

## Phase 5 — Performance and polish

- [x] `create_lattice` vectorized (torch ops + arithmetic partner map, no
      Python dict pass); unitary-gauge alignment vectorized
- [ ] vectorize `get_plaquette_indices` (still a Python dict loop; the
      interleaved-layout file contract must not change)
- [ ] profile the hetero CG on 4D lattices; batch the per-mu masks in
      `WilsonDiracConv.message`

## Validation milestones

- [ ] pure-gauge SU(2)/SU(3) average plaquette vs. beta against literature
- [ ] Higgs phase transition sweep with error bars
      (`benchmarks/higgs_phase_transition.py` exists; needs ensembles)
- [ ] quenched pion correlator with cosh fit + GMOR linearity with errors
- [ ] electroweak: rho ~ 1, massless photon, M_W/M_Z = cos(theta_W) on
      thermalized ensembles (ideal-vacuum version already pinned by tests)
