# Correctness review brief

**Date:** 2026-07-31 · **Scope:** full source tree (no prior git history) ·
**Method:** line-by-line review, every candidate verified numerically against
the project venv before being accepted.

## Confirmed and fixed (commits `6430fd8` and earlier)

Twelve confirmed defects, each verified by construction or by running the math:

1. **Diagonal-only SU(3) transport** — `einsum('ecc,ecs->ecs', U, psi)`
   contracts a repeated index and multiplies by only the diagonal of the link
   matrix. The Wilson-Dirac operator never applied a real color rotation
   (max elementwise error vs. correct transport: O(1)).
2. **Unstable `argsort` in `create_lattice`** — the `e = d*2V + node` edge
   indexing that `find_rectangular_loops` relies on silently broke; measured
   "plaquettes" were not even connected paths. Found by writing the loop-closure
   test; neither composition order was gauge invariant until this was fixed.
3. **Wilson loops composed in forward path order** — transport acts on column
   vectors, so the closed-loop product must be `U_ek ... U_e1` for the trace to
   telescope under `U'_e = G_dst U_e G_src^dag`. Verified numerically: forward
   order varies under gauge transformations, reverse order is invariant to 1e-4.
4. **Invalid Euclidean gammas in the flow pipeline** — `block_diag(-i tau, i tau)`
   gives `gamma^2 = -I`, and the hard-coded `g5 = diag(I,-I)` was not the product
   of those gammas, so `D^dag != g5 D g5` and CG solved an unjustified system.
5. **Wilson projector missing the direction sign** — `(I - gamma)` applied to
   both link orientations; the operator was not gamma5-hermitian.
6. **Swapped `u_gate`/`edge_dirs` arguments** in `apply_D_dag_D` — reproducible
   IndexError; the fermion loss could never run.
7. **Zero-gradient loss terms in the flow training** — the CG solve ran under
   `no_grad` on raw noise, and the Wilson terms act on dataset links; only the
   flow/Higgs terms trained. Fixed by routing `chi` through the flow output
   (with Y detached, `d(chi^dag Y)/d chi* = Y` is the exact gradient) and
   logging the constant Wilson terms outside the loss.
8. **Wrong unitary-gauge rotation** — `G = [[b*, -a*], [a, b]]/|phi|` does not
   send a complex doublet to `[0, |phi|]` (correct: `[[b, -a], [a*, b*]]/|phi|`),
   and the link-rotation einsum applied `conj` instead of the adjoint. The
   measured W/Z/photon masses were taken on a configuration not gauge-equivalent
   to the cooled vacuum.
9. **Pseudofermion heat bath sampled `phi ~ N(0,1)`** instead of
   `phi = M^dag eta` — the sampler targeted an ensemble whose fermionic weight
   was not `det(M^dag M)`. The fix is pinned by the exact identity
   `S_pf = |eta|^2` (verified: 175.7 vs. 192 dof before tightening CG, exact
   within CG tolerance after).
10. **GMOR sweep set a nonexistent attribute** (`.mass` vs. `.bare_mass`) — all
    five "quark masses" produced the identical operator.
11. **Dead gradient clipping** in the Langevin step; clipped tensor computed
    then discarded.
12. **Shadowed duplicate definitions** — two `HiggsAction` classes (the wrong
    one won at runtime and crashed the vacuum finders), duplicate
    `plot_dashboard`/`calculate_physics_observables`, and a commented-out
    `calculate_polyakov_loop` that was still being called.

## Standing physics caveat (not a bug to "fix")

The chirality-split electroweak transport (left feels SU(2)xU(1), right only
U(1)) combined with a Wilson term is not an exactly gauge-invariant chiral
lattice theory — this is the Nielsen-Ninomiya obstruction, not an
implementation error. Quantitative claims should rest on the vector-like
sectors (QCD, Higgs-gauge); the chiral sector is an effective construction.

## What now guards all of this

`tests/` (32 tests as of the refactor) pin: the Clifford algebra, g5
consistency, both edge-layout contracts, Wilson-loop closure, gate
unitarity/det/reversal, gauge invariance of every action, gamma5-hermiticity
and positivity of `D^dag D`, CG convergence, the `S_pf = |eta|^2` identity,
unitary-gauge alignment, ideal-vacuum boson masses (massless photon, rho ~ 1),
and Langevin force clipping.
