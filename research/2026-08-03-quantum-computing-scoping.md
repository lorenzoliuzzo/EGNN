# 2026-08-03 — quantum computing for this project: scoping

Status: open, scoping only. **Nothing here was run.** Every number is cited from
the literature, not measured, which is why none of it has entered
`plans/2026-07-31-lattice-sm-roadmap.md`. If any milestone below is adopted it
must come back with numbers from a script in `benchmarks/` first.

## The verdict up front

Quantum computing offers **no speedup for anything the repo currently does**.
Euclidean HMC works because `e^{-S}` is a positive weight, and on that terrain
classical hardware wins. The value is categorical extension, not acceleration:
Hamiltonian evolution dispenses with importance sampling, so the sign problem is
absent by construction rather than tamed. That buys access to three regimes our
sampler structurally cannot enter:

- finite baryon density (complex fermion determinant),
- a topological theta-term (complex weight),
- real-time Minkowski evolution — transport, thermalization, string breaking.

Everything else marketed under this banner is weaker than it sounds; see
"Rejected" below.

## What a port would cost, concretely

The repo is Lagrangian/Euclidean throughout: `WilsonAction` over plaquettes
(`src/actions.py:13`), links as the sampled state, `S` and not `H`. Quantum
simulation needs the Kogut-Susskind Hamiltonian,

    H = (g^2/2) sum_links E_a E_a + (1/2g^2 a) sum_plaq (2N - Tr U_p - Tr U_p^dag)

**What transfers.** The magnetic term is the sum `WilsonAction.forward` already
computes, over the same index structure from `find_rectangular_loops`
(`src/lattice.py:55`). `create_lattice` is dimension-generic (`dims =
len(shape)`, `src/lattice.py:22`). `su2_generators` / `su3_generators`
(`src/groups.py:28`) are the Casimir data the electric term needs.

**What does not.** The dynamics. And digitization: SU(3) has a continuous
manifold per link, so the Hilbert space per link is infinite-dimensional and must
be truncated — irreps up to `j_max`, a discrete subgroup, or a quantum-link /
qubit regularization. Truncation is where the physics error lives; it is an
active research area, not a preprocessing step.

## The conceptual bridge worth keeping

Hard constraint #1 in `CLAUDE.md` — invariance under `U' = G_dst U G_src^dag` —
has an exact Hamiltonian avatar: **Gauss's law** `G_a(x)|psi> = 0`, a constraint
on the physical Hilbert space, checkable the way `tests/test_actions.py` checks
invariance today. The analogue of "the GNN is equivariant" is "the variational
ansatz commutes with the Gauss-law generators."

Given that equivariance is this project's thesis, that is the distinctive thread
here rather than a bolt-on. It is also the one place where the repo's identity
suggests something the quantum-simulation literature has not exhausted.

## Feasibility, bluntly

State of the art as of the searches below:

- (1+1)D proof-of-concept simulations are done.
- SU(2) demos at ~18 qubits (energy loss, hadronization); (1+1)D SU(2) on ion
  qudits.
- SU(3) reached only at leading order in the large-`N_c` expansion.
- Resource-efficient algorithms exist for (1+1)D and (2+1)D; lower-dimensional
  gauge theories are estimated at **1,000-10,000 logical qubits**.
- Qubitization bought up to 25 orders of magnitude in spacetime volume over
  Trotterization for SU(2)/SU(3) — and full 3+1D QCD is *still* beyond estimates.

So the repo's SU(3)xSU(2)xU(1) target is a fault-tolerant-era problem. Nothing
about the full Standard Model sector is near-term.

## The one tractable on-ramp

**1+1D Schwinger model** (U(1) + staggered fermions) in Hamiltonian form. With
open boundaries the gauge field is eliminated via Gauss's law, leaving a spin
chain of dimension `2^N`. `N = 12-16` is laptop exact diagonalization: no GPU, no
dataset, CI-compatible under the testing rule in `CLAUDE.md`.

It self-validates against known continuum answers — the Schwinger mass `g/sqrt(pi)`
and the exact chiral condensate — so the first milestone does not depend on
comparing against our own code.

Sequence: reproduce those at `theta = 0`, confirm overlap with a Euclidean run,
then push to `theta != 0` and real-time quenches, where the HMC weight goes
complex and our sampler has nothing to say.

### Two caveats that must not be glossed

1. **Coupling matching.** Hamiltonian and Lagrangian couplings agree only in the
   anisotropic `a_t -> 0` limit. "ED vs. HMC at the same beta" is not
   apples-to-apples; it differs at O(a_t). Making the comparison rigorous needs
   the strong-coupling limit or a transfer-matrix argument.
2. **MPS/DMRG already solves 1+1D sign problems classically** and beats quantum
   hardware today. The ED / tensor-network work is the deliverable; running it on
   a device is a demonstration, not an advantage. This has to be stated out loud
   or a reader will assume otherwise.

### Scoping detail: the fermion sector is SU(3)-only

A U(1)+fermion Euclidean counterpart is not wired up today. `include_fermions`
raises without an `su3` link family (`src/hmc.py:219`), the pseudofermion field is
shaped `(nodes, 3, 4)` (`src/hmc.py:253`), and `get_gamma_matrices`
(`src/dirac.py:8`) is the 4D chiral convention. A Schwinger comparison needs a
color-decoupled fermion path and 2-component gammas.

## Rejected, with reasons

- **HHL / quantum linear systems for the CG solve** (`src/dirac.py:33`).
  `D^dag D x = b` is the textbook target and it does not work here: the output is
  the state `|x>`, not the vector, while lattice observables need the propagator
  contracted into correlators. Readout destroys the speedup. Add source-state
  preparation and the condition number near critical mass. Worth recording *why*,
  because it is the first thing anyone proposes.
- **Amplitude estimation for observable averages.** Quadratic in the statistical
  error only. Our measured bottleneck is critical slowing down — `tau_int` climbs
  1.00 -> 3.91 over `beta` in [0.25, 2.0] (`benchmarks/hmc_plaquette_scan.py`) —
  plus the fermion solve. Neither is a sample-count problem.
- **SU(3) in 3+1D at physical volumes.** See the resource estimates above.
- **Generic "quantum ML on gauge configurations" advantage claims.** No mechanism.

## Sources

- Review on Quantum Computing for Lattice Field Theory —
  https://arxiv.org/abs/2302.00467
- Digital Quantum Simulation of a (1+1)D SU(2) LGT with Ion Qudits, PRX Quantum
  5, 040309 — https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.040309
- Quantum Simulation of SU(3) Lattice Yang-Mills at Leading Order in Large N,
  PRL 133, 111901 — https://arxiv.org/pdf/2402.10265
- Efficient Truncations of SU(N_c) Lattice Gauge Theory for Quantum Simulation —
  https://arxiv.org/pdf/2503.11888
- Quantum circuits for SU(3) lattice gauge theory, Phys. Rev. D —
  https://journals.aps.org/prd/abstract/10.1103/k8f6-yft8
- SU(2) simulations with 18 qubits, energy loss and hadronization —
  https://quantumzeitgeist.com/quantum-qubits-simulations-lattice-gauge-theory-enable-energy-loss-hadronization/
