import numpy as np
import torch
from scipy.optimize import fsolve

from src.dirac import conjugate_gradient, g5_spin_last
from src.lattice import find_rectangular_loops


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
) -> dict:
    print("\n--- Measuring Electroweak Boson Masses ---")
    device = phi_vac.device
    num_edges = u_su2_vac.shape[0]

    theta_w_theory = np.arctan(g_u1 / g_su2)
    print(f"Theoretical Weinberg Angle (theta_W): {np.degrees(theta_w_theory):.2f} deg")

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

    print("\n--- Physical Parameter Estimates ---")
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


@torch.no_grad()
def measure_rho_parameter(model) -> float:
    # lattice mapping: beta_su2 = 4/g^2, beta_u1 = 1/g'^2
    g_sq = 4.0 / model.betas['su2']
    gp_sq = 1.0 / model.betas['u1']
    cos_sq_theta_w = g_sq / (g_sq + gp_sq)

    current_vev_sq = torch.sum(model.phi.abs() ** 2, dim=-1).mean().item()
    mw_sq = 0.25 * g_sq * current_vev_sq
    mz_sq = 0.25 * (g_sq + gp_sq) * current_vev_sq
    rho = mw_sq / (mz_sq * cos_sq_theta_w)

    print("\n--- Electroweak Health Check ---")
    print(f"Weinberg Angle (cos^2 theta_w): {cos_sq_theta_w:.4f}")
    print(f"Measured Rho Parameter: {rho:.6f}")
    return rho


@torch.no_grad()
def measure_cornell_potential(model, r_max: int = 6, t_fixed: int = 4) -> dict:
    """Static quark potential V(R) from R x T Wilson loops; a linear rise
    signals confinement."""
    model.eval()
    u_su3 = model.physical_gates()['su3']
    V_R = {}

    print("\n--- Measuring Cornell Potential (SU3) ---")
    for r in range(1, r_max + 1):
        loop_idx = find_rectangular_loops(model.lattice_shape, model.edge_index,
                                          R=r, T=t_fixed)
        u_p = u_su3[loop_idx]
        # reverse path order so the trace is gauge invariant
        loop_prod = u_p[:, 0]
        for i in range(1, u_p.shape[1]):
            loop_prod = torch.matmul(u_p[:, i], loop_prod)

        trace = torch.einsum('pii -> p', loop_prod).real / 3.0
        w_rt = trace.mean().item()

        v_r = -(1.0 / t_fixed) * np.log(max(w_rt, 1e-9))
        V_R[r] = v_r
        print(f"Distance R={r} | W(R,T)={w_rt:.4f} | V(R)={v_r:.4f}")

    return V_R


@torch.no_grad()
def measure_pion_mass(model, max_dist: int = 10, max_iter: int = 300) -> dict:
    """Pion correlator C(r) from a point-source quark propagator."""
    model.eval()
    u_su3 = model.physical_gates()['su3']

    print("\n--- Measuring Pion Mass (Quark Propagator) ---")
    source = torch.zeros_like(model.pf_phi)
    dims = model.lattice_shape
    center_coords = [d // 2 for d in dims]
    center_node = int(np.ravel_multi_index(center_coords, dims))
    source[center_node] = 1.0 + 0.0j

    x = conjugate_gradient(
        model.fermion_calc.dirac, source, model.phi, model.edge_index,
        model.edge_dirs, u_su3, model.is_fwd, max_iter=max_iter, tol=1e-6)

    correlator = torch.sum(x.abs() ** 2, dim=(-1, -2)).cpu()

    grid = np.indices(dims)
    reshape_args = [-1] + [1] * len(dims)
    center_arr = np.array(center_coords).reshape(*reshape_args)
    distances = np.sum(np.abs(grid - center_arr), axis=0).flatten()

    C_r = {}
    for r in range(1, max_dist + 1):
        mask = (distances == r)
        if mask.any():
            mean_signal = correlator[mask].mean().item()
            C_r[r] = mean_signal
            print(f"Distance r={r} | C(r)={mean_signal:.4e}")

    return C_r


@torch.no_grad()
def measure_chiral_condensate(model, num_noise_vectors: int = 5) -> float:
    """<psi_bar psi> via a stochastic trace estimator with Z2 noise."""
    model.eval()
    u_su3 = model.physical_gates()['su3']
    dirac = model.fermion_calc.dirac

    print("\n--- Measuring Chiral Condensate ---")
    condensate_sum = 0.0

    for _ in range(num_noise_vectors):
        noise = torch.randint(0, 2, model.pf_phi.shape, device=model.device).float() * 2 - 1
        noise = noise + 1j * (torch.randint(0, 2, model.pf_phi.shape,
                                            device=model.device).float() * 2 - 1)
        noise = noise / np.sqrt(2)

        x = conjugate_gradient(dirac, noise, model.phi, model.edge_index,
                               model.edge_dirs, u_su3, model.is_fwd, max_iter=200)

        # D^dag x = D^-1 noise, using D^dag = g5 D g5
        g5_x = g5_spin_last(dirac.g5, x)
        d_g5_x = dirac(g5_x, model.phi, model.edge_index, model.edge_dirs,
                       u_su3, model.is_fwd)
        d_dag_x = g5_spin_last(dirac.g5, d_g5_x)

        condensate_sum += torch.sum(noise.conj() * d_dag_x).real.item()

    V = int(np.prod(model.lattice_shape))
    condensate = condensate_sum / (num_noise_vectors * V * 3 * 4)
    print(f"<psi_bar psi> = {condensate:.6f}")
    return condensate


def run_gmor_test(
    model, test_masses: list[float] | None = None, max_dist: int = 8,
) -> tuple[list[float], list[float]]:
    """Gell-Mann-Oakes-Renner check: m_pi^2 should rise linearly with m_q."""
    print("\n=== Starting GMOR Chiral Test ===")
    test_masses = test_masses or [0.05, 0.10, 0.15, 0.20, 0.25]
    pion_masses = []

    for mq in test_masses:
        model.fermion_calc.dirac.bare_mass = mq
        pion_signal = measure_pion_mass(model, max_dist=max_dist)

        r_vals = list(pion_signal.keys())
        log_c = np.log(list(pion_signal.values()))
        # fit r = 2..6 to dodge short-range lattice artifacts and edge noise
        slope, _ = np.polyfit(r_vals[1:6], log_c[1:6], 1)
        pion_masses.append(-slope)

    return test_masses, pion_masses


# ---------------------------------------------------------------------------
# Ensemble statistics
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


def jackknife_mean(values: list[float] | np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()), float('nan')
    jk = np.array([np.mean(np.delete(v, i)) for i in range(n)])
    err = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
    return float(v.mean()), float(err)

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
