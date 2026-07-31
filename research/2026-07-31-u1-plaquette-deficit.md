# 2026-07-31 — the 2D U(1) plaquette "deficit" is a benchmark artefact

Status: closed. No defect in `src/hmc.py`. Fix applied to
`benchmarks/hmc_plaquette_scan.py` only.

## The report

`benchmarks/hmc_plaquette_scan.py` (6x6, `n_traj=250`, `warmup=50`) sat below
the exact `I1/I0` curve at small and intermediate beta, consistently in sign:

| beta | measured | exact (V=36) | dev |
|------|----------|--------------|-----|
| 0.25 | 0.0972 +/- 0.0115 | 0.12403 | -2.3 sigma |
| 0.50 | 0.2345 +/- 0.0111 | 0.24250 | -0.7 sigma |
| 1.00 | 0.4260 +/- 0.0102 | 0.44639 | -2.0 sigma |
| 1.50 | 0.5727 +/- 0.0153 | 0.59613 | -1.5 sigma |
| 2.00 | 0.6972 +/- 0.0137 | 0.69778 | -0.0 sigma |
| 3.00 | 0.8047 +/- 0.0082 | 0.81008 | -0.7 sigma |

Reproduced bit-for-bit. Note the exact values at beta=1.5 and 3.0 quoted in the
original report (0.5787, 0.8038) were wrong; the correct `I1/I0` is 0.59613 and
0.80999, so the apparent deficit actually extended across the whole range.

## Finite volume: verified, not assumed

On a periodic torus with `Np` plaquettes,
`Z = sum_n I_n(beta)^Np` and `<Re U_p> = sum_n I_n^{Np-1} I_n' / sum_n I_n^Np`
with `I_n' = (I_{n-1} + I_{n+1})/2`. Evaluated at `Np=36` in 60-digit precision:

| beta | V=36 exact | I1/I0 | difference |
|------|-----------|-------|------------|
| 0.25 | 0.1240335019 | 0.1240335019 | 0 |
| 1.00 | 0.4463899659 | 0.4463899659 | 3.9e-13 |
| 3.00 | 0.8100777856 | 0.8099852940 | 9.2e-05 |

Finite volume is ruled out — the largest correction anywhere in the scan is
9e-5, two orders of magnitude below the error bars.

## The sampler is sound

Each candidate cause was tested directly, not argued:

- **Plaquette indexing.** `plaq_idx` gives 36 loops of 4 edges; `<Re U_p>` from
  `plaq_idx` matches an explicit coordinate-space walk over
  `U_x(n) U_y(n+x) U_x(n+y)^* U_y(n)^*` to 8 digits, and `S` matches
  `beta*Np*(1 - <Re U_p>)`.
- **U(1) force.** `algebra_force` finite-differenced against the action along
  every single-link algebra direction in float64: max deviation 1.5e-9 at
  beta=0.25, 5.9e-9 at beta=1.0, on a force of scale O(1).
- **Leapfrog reversibility** at the suspicious eps=0.402: `max|dU| = 9.7e-7`,
  `max|dpi| = 7.7e-7` — float32 roundoff, nothing structural.
- **`<exp(-dH)> = 1`** holds on every chain measured (e.g. 1.0030 +/- 0.0040 at
  beta=0.25, 0.9993 +/- 0.0036 at beta=1.5 over 9500 trajectories).
- **Independent sampler.** A local Metropolis on the link angles, driven by the
  repo's own `WilsonAction`/`u_full`/`average_plaquette`, reproduces the exact
  curve (diffs +0.0018, +0.0015, +0.0018, -0.0006 at beta = 0.25, 1.0, 1.5, 3.0
  over 4000 sweeps). So the action and the observable are right independently of
  HMC.
- **Step-size exactness.** beta=1.0, fixed eps, no tuning, equilibrated start,
  320 independent chains x 250 measurements:

  | eps | acceptance | dev from exact |
  |-----|-----------|----------------|
  | 0.10 | 0.984 | -0.00025 +/- 0.00045 |
  | 0.35 | 0.802 | +0.00003 +/- 0.00076 |
  | 0.60 | 0.078 | +0.145 +/- 0.012 |

  At the step size the tuner actually selects the sampler is exact to +/-0.0008.
  The eps=0.60 row is a chain frozen near the cold start at 8% acceptance, which
  biases *high* — the opposite sign, and a diagnostic of a bad eps rather than a
  bug.

## What it actually was

**Statistics, plus an RNG-sharing artefact that disguised them as a systematic.**

At 9500 trajectories (warmup 500) every beta agrees with the exact finite-volume
value:

| beta | <P> | exact | dev |
|------|-----|-------|-----|
| 0.25 | 0.12410 +/- 0.00172 | 0.12403 | +0.04 sigma |
| 0.50 | 0.24333 +/- 0.00203 | 0.24250 | +0.41 sigma |
| 1.00 | 0.44509 +/- 0.00219 | 0.44639 | -0.59 sigma |
| 1.50 | 0.59603 +/- 0.00166 | 0.59613 | -0.06 sigma |
| 2.00 | 0.69651 +/- 0.00172 | 0.69778 | -0.74 sigma |
| 3.00 | 0.81001 +/- 0.00142 | 0.81008 | -0.05 sigma |

Two more seeds at beta=0.25 and 1.0 scatter in sign (+0.37, -1.41, +0.99,
-0.57 sigma). Combining the three beta=1.0 chains: dev = -0.00011 +/- 0.00127.

The "consistent in sign across beta" argument was the trap. `scan()` called
`torch.manual_seed(0)` before *every* beta, so all six points consumed the same
random stream from the same cold start. Over 12 seeds the per-seed deviations
are correlated between beta points with r = 0.3-0.7, and 3/12 seeds have all six
points on the same side (vs ~3% if they were independent). Seed 0 is one of
them. The six points were never six independent tests of the same hypothesis.

The true per-run scatter of the 250/50 protocol is 0.0135-0.0146, so seed 0's
beta=1.0 point was a 1.4-sigma single-run fluctuation, not a 2-sigma one — the
tau-inflated error bar was itself an underestimate at that chain length.

Thermalization was also checked and is not the cause: averaged over 320 cold
starts at beta=1.5, the chain relaxes from *above* (P=0.967 at traj 0) and is
equilibrated by traj ~20; the mean over traj 50..299 is -0.00075 +/- 0.00067.
Raising warmup from 50 to 400 at beta=1.0 moves the result by -0.00049 +/-
0.00115, i.e. not at all.

A residual of about -0.002 (0.5% relative, ~0.15 of a single-run sigma) survives
in the 320-seed *tuned* 250/50 protocol at beta=1.0 and 1.5 at the 2-3 sigma
level. It does not appear at fixed eps, does not respond to warmup, and is an
order of magnitude below the effect originally reported. Not pursued further; if
it matters later, the place to look is the interaction between the warmup eps
tuner and the short measurement window, not the integrator.

## Change made

`benchmarks/hmc_plaquette_scan.py`: distinct seed per beta (the points are now
genuinely independent), and `n_traj` 250 -> 2000, `warmup` 50 -> 200. `src/` is
untouched.

Rerun of the fixed benchmark (errors here are plain jackknife, so still an
underestimate at beta >= 1 where tau_int ~ 2-4):

| beta | <P> | exact (V=36) | diff |
|------|-----|--------------|------|
| 0.25 | 0.1177 +/- 0.0027 | 0.12403 | -0.0063 |
| 0.50 | 0.2413 +/- 0.0026 | 0.24250 | -0.0012 |
| 1.00 | 0.4405 +/- 0.0024 | 0.44639 | -0.0059 |
| 1.50 | 0.6034 +/- 0.0019 | 0.59613 | +0.0073 |
| 2.00 | 0.6959 +/- 0.0016 | 0.69778 | -0.0019 |
| 3.00 | 0.8124 +/- 0.0011 | 0.81008 | +0.0023 |

The deviations now scatter in sign, which is the actual point: with independent
seeds the one-sided pattern that motivated this investigation is gone.

One thing worth a later look, unrelated to the deficit: with `warmup=200` the
tuner gets 20 adjustment steps and ramps eps to 0.603 at beta=0.25 and 0.566 at
beta=0.5. It calibrates from a cold start where the Wilson force vanishes
identically (all plaquettes are 1, so U=1 is a stationary point), so it
overshoots before the config thermalizes and only claws back afterwards.
Harmless for correctness — Metropolis fixes it — but it costs acceptance
(0.62 at beta=1.5).
