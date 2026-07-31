# Lattice Standard Model roadmap

**Status:** Phases 0–2 done; Phase 3 core done (manifold HMC with Metropolis,
exact 2D U(1) validation, SU(2) strong-coupling check, jackknife plaquettes);
Phase 4 decided: (B) physics-first HMC; 39 tests green, ruff clean ·
**Last updated:** 2026-07-31

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

## Phase 3 — Sampling correctness

- [x] HMC on the group manifold (`src/hmc.py`): momenta in the Lie algebra,
      `U <- exp(i eps pi) U` updates, autograd forces (torch's complex grad is
      `2 dS/dU*`; force `W = i(U G^dag - G U^dag)/2`, verified numerically),
      leapfrog + Metropolis accept/reject — exact for `e^{-S}`, replacing the
      biased unadjusted Langevin
- [x] integrator invariants pinned by tests: reversibility, O(eps^2) energy
      scaling, group preservation, momentum/kinetic normalization
      (E[T] = dim(algebra)/2 per link)
- [x] step-size auto-tuning during warmup toward 70–85% acceptance (also
      escapes the cold-start all-reject trap); sampling phase runs at fixed eps
- [x] ensemble plaquettes with jackknife errors (`average_plaquette`,
      `jackknife_mean` in `src/measure.py`)
- [ ] autocorrelation-aware errors (binning / integrated autocorrelation time);
      plain jackknife on correlated series underestimates the small-beta U(1)
      error bars by ~2x
- [ ] port pion/EW/condensate measurements onto HMC ensembles (currently they
      run on single cooled configurations)

## Phase 4 — ML direction: (B) physics-first, decided 2026-07-31

- [x] gauge links are the sampled state; the GNN operators are the physics
      (message passing = parallel transport); pseudofermion forces enter HMC
      through the CG surrogate gradient (`StandardModelHMC`,
      full-system smoke test in `tests/test_hmc.py`)
- [ ] fermionic HMC at scale: CG tolerance vs. acceptance study, per-sector
      trajectory lengths, 4D lattices
- [ ] decide the fate of the flow pipeline (`src/main.py`): keep as a side
      experiment or retire; AMP with complex tensors needs validation before
      trusting any run (inherited behavior, currently enabled)

## Phase 5 — Performance and polish

- [x] `create_lattice` vectorized (torch ops + arithmetic partner map, no
      Python dict pass); unitary-gauge alignment vectorized
- [ ] vectorize `get_plaquette_indices` (still a Python dict loop; the
      interleaved-layout file contract must not change)
- [ ] profile the hetero CG on 4D lattices; batch the per-mu masks in
      `WilsonDiracConv.message`

## Validation milestones

- [x] 2D U(1) average plaquette vs. the exact solution I1(b)/I0(b): matches
      across beta in [0.25, 3] (pinned at beta=1 by `test_hmc.py`; full scan
      in `benchmarks/hmc_plaquette_scan.py`)
- [x] 3D SU(2) plaquette: strong-coupling limit reproduced
      (<P> = 0.1257(30) vs beta/4 = 0.125 at beta = 0.5); full scan
      interpolates to the weak-coupling envelope
- [ ] SU(3) plaquette scan and 4D lattices vs. literature values
- [ ] Higgs phase transition sweep with error bars
      (`benchmarks/higgs_phase_transition.py` exists; needs ensembles)
- [ ] quenched pion correlator with cosh fit + GMOR linearity with errors
- [ ] electroweak: rho ~ 1, massless photon, M_W/M_Z = cos(theta_W) on
      thermalized ensembles (ideal-vacuum version already pinned by tests)
