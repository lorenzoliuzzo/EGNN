from collections.abc import Callable

import numpy as np
import torch
from scipy.optimize import fsolve

from src.dirac import WilsonDiracOperator, conjugate_gradient, g5_spin_last
from src.lattice import find_rectangular_loops

# Observables are split into per-configuration kernels (take explicit field
# tensors) and ensemble_* wrappers that average over HMC samples and attach
# jackknife errors. Single-configuration numbers are for debugging only; the
# ensemble versions are what plans and the report may cite.


# ---------------------------------------------------------------------------
# Jackknife
# ---------------------------------------------------------------------------

def jackknife_mean(values: list[float] | np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float('nan')
    jk = np.array([np.mean(np.delete(v, i)) for i in range(n)])
    err = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    return float(v.mean()), float(err)


def jackknife_transformed(
    values: list[float] | np.ndarray,
    fn: Callable[[float], float],
) -> tuple[float, float]:
    """Jackknife error of fn(mean(values)) for a nonlinear fn (e.g. -log)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    center = float(fn(v.mean()))
    if n < 2:
        return center, float('nan')
    jk = np.array([fn(np.mean(np.delete(v, i))) for i in range(n)])
    err = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    return center, float(err)


def jackknife_columns(
    rows: np.ndarray,
    fn: Callable[[np.ndarray], float],
) -> tuple[float, float]:
    """Jackknife for a statistic fn computed from the column-mean of a
    [samples, features] array (e.g. a fit through an averaged correlator)."""
    n = rows.shape[0]
    center = float(fn(rows.mean(axis=0)))
    if n < 2:
        return center, float('nan')
    jk = np.array([fn(np.delete(rows, i, axis=0).mean(axis=0)) for i in range(n)])
    err = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    return center, float(err)


# ---------------------------------------------------------------------------
# Per-configuration kernels
# ---------------------------------------------------------------------------

@torch.no_grad()
def average_plaquette(u_gates: torch.Tensor, plaq_idx: torch.Tensor,
                      group_dim: int) -> float:
    """Mean Re Tr(U_p)/N over all plaquettes (reverse path composition,
    matching WilsonAction)."""
    u_p = u_gates[plaq_idx]
    loop = u_p[:, 0]
    for i in range(1, u_p.shape[1]):
        loop = torch.matmul(u_p[:, i], loop)
    tr = torch.einsum('pii -> p', loop).real
    return (tr / group_dim).mean().item()


@torch.no_grad()
def wilson_loop_average(
    u_gates: torch.Tensor,
    lattice_shape: tuple[int, ...],
    edge_index: torch.Tensor,
    R: int,
    T: int,
) -> float:
    """Mean Re Tr(W_{RxT})/N over all positions and orientations."""
    loop_idx = find_rectangular_loops(lattice_shape, edge_index, R=R, T=T)
    return average_plaquette(u_gates, loop_idx, u_gates.size(-1))


@torch.no_grad()
def pion_correlator(
    dirac: WilsonDiracOperator,
    phi: torch.Tensor,
    u_su3: torch.Tensor,
    edge_index: torch.Tensor,
    edge_dirs: torch.Tensor,
    is_fwd: torch.Tensor,
    lattice_shape: tuple[int, ...],
    max_dist: int = 8,
    max_iter: int = 300,
) -> dict[int, float]:
    """C(r): point-source quark propagator magnitude, radially averaged over
    Manhattan shells."""
    num_nodes = int(np.prod(lattice_shape))
    source = torch.zeros(num_nodes, 3, 4, dtype=torch.complex64,
                         device=u_su3.device)
    center_coords = [d // 2 for d in lattice_shape]
    center_node = int(np.ravel_multi_index(center_coords, lattice_shape))
    source[center_node] = 1.0 + 0.0j

    x = conjugate_gradient(dirac, source, phi, edge_index, edge_dirs, u_su3,
                           is_fwd, max_iter=max_iter, tol=1e-6)
    correlator = torch.sum(x.abs() ** 2, dim=(-1, -2)).cpu()

    grid = np.indices(lattice_shape)
    center_arr = np.array(center_coords).reshape([-1] + [1] * len(lattice_shape))
    distances = np.sum(np.abs(grid - center_arr), axis=0).flatten()

    C_r = {}
    for r in range(1, max_dist + 1):
        mask = (distances == r)
        if mask.any():
            C_r[r] = correlator[mask].mean().item()
    return C_r


@torch.no_grad()
def chiral_condensate_sample(
    dirac: WilsonDiracOperator,
    phi: torch.Tensor,
    u_su3: torch.Tensor,
    edge_index: torch.Tensor,
    edge_dirs: torch.Tensor,
    is_fwd: torch.Tensor,
    num_noise_vectors: int = 3,
) -> float:
    """<psi_bar psi> on one configuration via a Z2 stochastic trace."""
    num_nodes = phi.size(0)
    total = 0.0
    for _ in range(num_noise_vectors):
        shape = (num_nodes, 3, 4)
        noise = torch.randint(0, 2, shape, device=u_su3.device).float() * 2 - 1
        noise = noise + 1j * (torch.randint(0, 2, shape,
                                            device=u_su3.device).float() * 2 - 1)
        noise = (noise / np.sqrt(2)).to(torch.complex64)

        x = conjugate_gradient(dirac, noise, phi, edge_index, edge_dirs, u_su3,
                               is_fwd, max_iter=200)
        # D^dag x = D^-1 noise, using D^dag = g5 D g5
        g5_x = g5_spin_last(dirac.g5, x)
        d_g5_x = dirac(g5_x, phi, edge_index, edge_dirs, u_su3, is_fwd)
        d_dag_x = g5_spin_last(dirac.g5, d_g5_x)
        total += torch.sum(noise.conj() * d_dag_x).real.item()

    return total / (num_noise_vectors * num_nodes * 3 * 4)


# ---------------------------------------------------------------------------
# Ensemble observables (HMC samples in, central values with errors out)
# ---------------------------------------------------------------------------

def ensemble_cornell_potential(
    u_samples: list[torch.Tensor],
    lattice_shape: tuple[int, ...],
    edge_index: torch.Tensor,
    r_max: int = 4,
    t_fixed: int = 2,
) -> dict[int, tuple[float, float]]:
    """V(R) = -(1/T) ln <W(R,T)> with jackknife errors from the ensemble."""
    potential = {}
    for r in range(1, r_max + 1):
        w_vals = [wilson_loop_average(u, lattice_shape, edge_index, r, t_fixed)
                  for u in u_samples]
        potential[r] = jackknife_transformed(
            w_vals, lambda m: -(1.0 / t_fixed) * np.log(max(m, 1e-12)))
    return potential


def fit_string_tension(
    potential: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    """Linear fit V(R) = c + sigma R; the naive error propagates the V errors."""
    rs = np.array(sorted(potential.keys()), dtype=float)
    vs = np.array([potential[int(r)][0] for r in rs])
    errs = np.array([potential[int(r)][1] for r in rs])
    sigma, _ = np.polyfit(rs, vs, 1)
    denom = np.sqrt(np.sum((rs - rs.mean()) ** 2))
    sigma_err = float(np.sqrt(np.mean(errs ** 2)) / max(denom, 1e-12))
    return float(sigma), sigma_err


def ensemble_pion_mass(
    correlators: list[dict[int, float]],
    fit_start: int = 1,
    fit_stop: int = 5,
) -> tuple[dict[int, float], float, float]:
    """Effective mass from the log-slope of the ensemble-averaged correlator,
    jackknifed over configurations."""
    rs = sorted(set.intersection(*(set(c.keys()) for c in correlators)))
    rows = np.array([[c[r] for r in rs] for c in correlators])
    sl = slice(fit_start, min(fit_stop, len(rs)))
    rs_arr = np.array(rs, dtype=float)

    def fit(mean_c: np.ndarray) -> float:
        slope, _ = np.polyfit(rs_arr[sl], np.log(np.maximum(mean_c[sl], 1e-30)), 1)
        return -slope

    mass, err = jackknife_columns(rows, fit)
    mean_correlator = dict(zip(rs, rows.mean(axis=0)))
    return mean_correlator, mass, err


def ensemble_chiral_condensate(values: list[float]) -> tuple[float, float]:
    return jackknife_mean(values)


def ensemble_gmor(
    u_samples: list[torch.Tensor],
    phi: torch.Tensor,
    lattice_shape: tuple[int, ...],
    edge_index: torch.Tensor,
    edge_dirs: torch.Tensor,
    is_fwd: torch.Tensor,
    quark_masses: list[float],
    yukawa: float = 0.0,
    max_dist: int = 6,
) -> list[tuple[float, float, float]]:
    """GMOR check on a fixed (quenched) ensemble: m_pi(m_q) with errors,
    measured with the same configurations at every quark mass."""
    results = []
    for mq in quark_masses:
        dirac = WilsonDiracOperator(y=yukawa, bare_mass=mq)
        correlators = [
            pion_correlator(dirac, phi, u, edge_index, edge_dirs, is_fwd,
                            lattice_shape, max_dist=max_dist)
            for u in u_samples
        ]
        _, m_pi, err = ensemble_pion_mass(correlators)
        results.append((mq, m_pi, err))
    return results


def ensemble_electroweak_masses(
    phi_samples: list[torch.Tensor],
    u_su2_samples: list[torch.Tensor],
    u_u1_samples: list[torch.Tensor],
    edge_index: torch.Tensor,
    is_fwd: torch.Tensor,
    higgs_calc,
    g_su2: float = 1.0,
    g_u1: float = 0.5,
) -> dict:
    """Align every configuration to unitary gauge, measure W/Z/photon masses
    on each, and jackknife over the ensemble."""
    per_sample = {'W Boson': [], 'Z Boson': [], 'Photon': [], 'rho': [], 'vev': []}

    for phi, u2, u1 in zip(phi_samples, u_su2_samples, u_u1_samples):
        phi_a, u2_a, u1_a = align_to_unitary_gauge(phi, edge_index, u2, u1)
        stats = measure_electroweak_masses(
            phi_a, u2_a, u1_a, edge_index, is_fwd, higgs_calc,
            g_su2=g_su2, g_u1=g_u1, verbose=False)
        for name in ('W Boson', 'Z Boson', 'Photon'):
            per_sample[name].append(stats['masses'][name])
        per_sample['rho'].append(stats['rho'])
        per_sample['vev'].append(torch.norm(phi, dim=-1).mean().item())

    return {key: jackknife_mean(vals) for key, vals in per_sample.items()}


# ---------------------------------------------------------------------------
# Electroweak sector
# ---------------------------------------------------------------------------

@torch.no_grad()
def align_to_unitary_gauge(
    phi: torch.Tensor,
    edge_index: torch.Tensor,
    u_su2: torch.Tensor,
    u_u1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate phi to [0, |phi|] at every node and co-rotate the SU(2) links,
    leaving the action invariant (unitary gauge)."""
    norm = torch.norm(phi, dim=-1)
    a, b = phi[:, 0], phi[:, 1]

    # G = [[b, -a], [conj(a), conj(b)]] / |phi| sends [a, b] to [0, |phi|]
    g_mats = torch.stack([
        torch.stack([b, -a], dim=-1),
        torch.stack([a.conj(), b.conj()], dim=-1),
    ], dim=-2) / norm.clamp_min(1e-8).view(-1, 1, 1)

    eye = torch.eye(2, dtype=torch.complex64, device=phi.device)
    degenerate = (norm < 1e-8).view(-1, 1, 1)
    g_mats = torch.where(degenerate, eye, g_mats)

    phi_aligned = torch.zeros_like(phi)
    phi_aligned[:, 1] = norm.to(torch.complex64)

    # U'_e = G_dst U_e G_src^dag keeps every gauge-invariant quantity unchanged
    g_src = g_mats[edge_index[0]]
    g_tgt = g_mats[edge_index[1]]
    u_su2_aligned = torch.einsum('eij, ejk, ekl -> eil', g_tgt, u_su2, g_src.adjoint())

    # U(1) commutes with SU(2) rotations
    return phi_aligned, u_su2_aligned, u_u1


def measure_electroweak_masses(
    phi_vac: torch.Tensor,
    u_su2_vac: torch.Tensor,
    u_u1_vac: torch.Tensor,
    edge_index: torch.Tensor,
    is_fwd: torch.Tensor,
    higgs_calc,
    g_su2: float = 1.0,
    g_u1: float = 0.5,
    eps: float = 1e-3,
    verbose: bool = True,
) -> dict:
    device = phi_vac.device
    num_edges = u_su2_vac.shape[0]

    theta_w_theory = np.arctan(g_u1 / g_su2)

    tau_1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    tau_3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    I_su2 = torch.eye(2, dtype=torch.complex64, device=device)

    gen_W = tau_1
    # the photon generator annihilates phi = [0, v]: exactly massless if the
    # vacuum is aligned
    gen_A = tau_3 + I_su2
    gen_Z = (g_su2 ** 2 * tau_3 - g_u1 ** 2 * I_su2) / (g_su2 ** 2 + g_u1 ** 2) ** 0.5

    u_su2_pure = I_su2.expand(num_edges, -1, -1)
    u_u1_pure = torch.ones((num_edges, 1, 1), dtype=torch.complex64, device=device)
    num_fwd_edges = int(is_fwd.sum().item())

    masses = {}
    for name, generator in [("W Boson", gen_W), ("Z Boson", gen_Z), ("Photon", gen_A)]:
        with torch.no_grad():
            s_base = higgs_calc(phi_vac, edge_index, is_fwd, u_su2_pure, u_u1_pure).item()

        u_eps_all = torch.matrix_exp(1j * eps * generator).expand(num_edges, -1, -1)
        with torch.no_grad():
            s_new = higgs_calc(phi_vac, edge_index, is_fwd, u_eps_all, u_u1_pure).item()

        # the kinetic action only sums forward edges, so normalize per forward edge
        delta_s_per_edge = (s_new - s_base) / num_fwd_edges
        mass = np.sqrt(max(0, 2 * delta_s_per_edge) / (eps ** 2))
        masses[name] = mass
        if verbose:
            print(f"{name:8} Mass: {mass:.4f}")

    mw, mz, ma = masses["W Boson"], masses["Z Boson"], masses["Photon"]

    if mz > 1e-6:
        rho_measured = (mw ** 2) / ((mz ** 2) * (np.cos(theta_w_theory) ** 2))
    else:
        rho_measured = 0.0

    if mz > 1e-6 and mw <= mz:
        sin2 = 1.0 - (mw ** 2 / mz ** 2)
        theta_w_measured = np.arcsin(np.sqrt(sin2))
    else:
        theta_w_measured = 0.0

    if verbose:
        print(f"Measured Rho parameter:  {rho_measured:.6f} (Theory: 1.0)")
        print(f"Measured Weinberg Angle: {np.degrees(theta_w_measured):.2f} deg "
              f"(Theory: {np.degrees(theta_w_theory):.2f} deg)")
        if abs(rho_measured - 1.0) < 0.05 and ma < 1e-2:
            print("SUCCESS: custodial symmetry preserved (rho ~ 1), photon massless.")
        else:
            print("WARNING: significant deviation in rho or the photon mass.")

    return {
        "masses": masses,
        "theta_w_theory": theta_w_theory,
        "theta_w_measured": theta_w_measured,
        "rho": rho_measured,
    }


# ---------------------------------------------------------------------------
# Miscellaneous ensemble statistics
# ---------------------------------------------------------------------------

def calculate_polyakov_loop(u_su3: torch.Tensor, L: int) -> float:
    """Confinement order parameter, assuming the interleaved 2D edge layout."""
    u_grid = u_su3.reshape(-1, 4, 3, 3)
    u_temporal = u_grid[:, 2]
    traces = torch.real(torch.diagonal(u_temporal, dim1=-2, dim2=-1).sum(-1))
    return torch.mean(traces).item() / 3.0


def calculate_susceptibility(samples: list[dict]) -> tuple[float, float]:
    mags = np.array([torch.norm(s['phi'], dim=-1).mean().item() for s in samples])
    chi = np.var(mags)
    # Binder cumulant: ~2/3 at a second-order transition
    u4 = 1 - np.mean(mags ** 4) / (3 * np.mean(mags ** 2) ** 2)
    return chi, u4


def jackknife_effective_mass(samples: list[dict], L: int) -> None:
    """Cosh-ratio effective mass with jackknife error bars from an ensemble of
    field configurations."""
    print(f"\nJACKKNIFE ENSEMBLE ANALYSIS ({len(samples)} configs)")
    n_samples = len(samples)
    t_half = L // 2
    individual_Ct = []

    for s in samples:
        phi = s['phi']
        phi_sq = torch.norm(phi, dim=-1) ** 2
        C_raw = np.mean(phi_sq.mean(dim=1).view(L, L).numpy(), axis=1)
        C_raw -= np.mean(C_raw)
        individual_Ct.append(np.array([np.mean(C_raw * np.roll(C_raw, -t))
                                       for t in range(L)]))

    individual_Ct = np.array(individual_Ct)
    jack_m_eff = np.zeros((n_samples, t_half - 1))

    for i in range(n_samples):
        block_Ct = np.mean(np.delete(individual_Ct, i, axis=0), axis=0)
        for t in range(1, t_half):
            ratio = block_Ct[t] / block_Ct[t + 1]

            def root_func(m: float) -> float:
                return (np.cosh(m * (t - t_half))
                        / np.cosh(m * (t + 1 - t_half)) - ratio)

            try:
                jack_m_eff[i, t - 1] = fsolve(root_func, x0=0.5)[0]
            except Exception:
                jack_m_eff[i, t - 1] = np.nan

    for t_idx in range(t_half - 1):
        masses = jack_m_eff[:, t_idx]
        valid = masses[~np.isnan(masses)]
        if len(valid) > n_samples // 2:
            avg_m = np.mean(valid)
            err_m = np.sqrt((n_samples - 1) / n_samples * np.sum((valid - avg_m) ** 2))
            print(f"  m_eff(t={t_idx + 1}): {avg_m:.4f} +/- {err_m:.4f}")
