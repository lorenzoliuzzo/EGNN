# Lattice Standard Model roadmap

**Status:** Phase 3 complete — manifold HMC with Metropolis, all observables on
HMC ensembles with autocorrelation-aware jackknife errors; exact 2D U(1), SU(2)
strong-coupling, area-law string tension, and Weinberg-relation validations;
Phase 4 = (B) physics-first, first milestones done; 50 tests green, ruff clean ·
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
- [x] all observables ported onto HMC ensembles (`src/measure.py`: per-config
      kernels + `ensemble_*` wrappers with jackknife errors, incl.
      `jackknife_transformed` for nonlinear observables like V(R)); the old
      single-cooled-configuration measurement API is gone
- [x] autocorrelation-aware errors: `integrated_autocorrelation_time` (Madras-
      Sokal windowing, lags capped at n/2) inflates every jackknife error by
      sqrt(2 tau_int) in `src/measure.py`. Validated against AR(1) chains with
      analytic tau_int = 1/2 + phi/(1-phi), and against the empirical spread of
      the mean over 60 independent chains (`tests/test_autocorrelation.py`).
      Inflation, not block-binning: our ensembles are O(10-100) configurations,
      too few for a variance over blocks.
      Findings from re-running the benchmarks:
      - the earlier "small-beta U(1) off by ~2x" note was backwards. tau_int
        *grows* with beta (critical slowing down): U(1) 6^2 goes 1.00 at
        beta=0.25 to 3.91 at beta=2.0, so the correction is 1.4x at small beta
        and 2.8x at large beta (`benchmarks/hmc_plaquette_scan.py`, which now
        prints tau_int per point).
      - every cited validation number below is unchanged, because they are all
        measured on ensembles thinned by `sample_every`: at sample_every=4 the
        SU(3) 8^2 series has tau_int = 0.50 (inflation 1.00) versus 0.87
        (inflation 1.32) unthinned. Thinning was already doing the work.

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
      across beta in [0.25, 3] — largest deviation 1.6 sigma, signs scattered
      (0.1177(42) vs 0.1240 at beta=0.25, 0.4405(52) vs 0.4464 at beta=1.0,
      0.6034(45) vs 0.5961 at beta=1.5). The earlier one-sided deficit was an
      artefact of `scan()` reseeding to 0 before every beta, which correlated
      the six points (r = 0.3-0.7); with distinct seeds and 2000 trajectories
      it is gone. Sampler exactness verified independently — see
      `research/2026-07-31-u1-plaquette-deficit.md`. (Pinned at beta=1 by
      `test_hmc.py`; full scan in `benchmarks/hmc_plaquette_scan.py`.)
- [x] 3D SU(2) plaquette: strong-coupling limit reproduced
      (<P> = 0.1257(30) vs beta/4 = 0.125 at beta = 0.5); full scan
      interpolates to the weak-coupling envelope
- [x] 2D SU(3) string tension vs. the exact area law on a quenched HMC
      ensemble: sigma = 0.833(140) vs -ln<P> = 0.860 at beta = 6, L = 8^2,
      20 configs (`benchmarks/qcd_observables.py`; SU(2) variant pinned by
      `test_ensembles.py`)
- [x] quenched pion mass + GMOR sweep with jackknife errors on a fixed
      ensemble: m_pi rises monotonically 0.7475(122) -> 0.9757(113) over
      m_q in [0.05, 0.25] (same benchmark)
- [x] electroweak on thermalized SU(2)xU(1)+Higgs HMC ensembles
      (`benchmarks/electroweak_masses.py`, 24 configs, L = 6^2): photon
      massless, rho = 1.0026(26), M_W/M_Z = 0.9141 vs cos(theta_W) = 0.9129
- [ ] SU(3) plaquette scan and 4D lattices vs. literature values
- [ ] Higgs phase transition sweep with error bars
      (`benchmarks/higgs_phase_transition.py` still uses cooling; port to
      HMC ensembles with the susceptibility/Binder machinery)
- [ ] cosh fit (periodic correlator) instead of the log-slope fit for m_pi
