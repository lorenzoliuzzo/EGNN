from collections.abc import Callable

import numpy as np
import torch

from src.actions import HiggsAction, PseudofermionAction, WilsonAction
from src.dirac import g5_spin_last
from src.lattice import create_lattice, find_rectangular_loops

# Hybrid Monte Carlo on the group manifold. The state is the set of FORWARD
# links U in the gauge group (backward links are reconstructed as U^dag), plus
# optional scalar fields. Momenta for links live in the Lie algebra (hermitian,
# traceless for SU(N)); positions evolve as U <- exp(i eps pi) U so the links
# never leave the group. Forces come from autograd; a Metropolis accept/reject
# on Delta H makes the sampler exact for e^{-S} (no step-size bias, unlike the
# unadjusted Langevin in src/vacuum.py).
#
# Conventions: torch's complex gradient for a real loss is G = 2 dS/dU*, so a
# variation obeys delta S = Re[conj(G) dU] (verified numerically). Under
# U -> exp(i eps P) U this gives delta S = eps * Tr[P W] with
#   W = i (U G^dag - G U^dag) / 2,
# which is hermitian; its traceless part is the algebra force. Momenta sampled
# as pi = (A + A^dag)/sqrt(2), A ~ CN(0,1), have density exp(-Tr(pi^2)/2),
# matching the kinetic term T = Tr(pi^2)/2. Scalars: T = |p|^2/2 with
# Re/Im ~ N(0,1), and the real-coordinate force is exactly the packed grad.


def sample_link_momentum(u_fwd: torch.Tensor, is_su: bool) -> torch.Tensor:
    a = torch.randn_like(u_fwd)
    pi = (a + a.mH) / np.sqrt(2)
    if is_su:
        n = pi.size(-1)
        tr = torch.einsum('eii->e', pi) / n
        eye = torch.eye(n, dtype=pi.dtype, device=pi.device)
        pi = pi - tr.view(-1, 1, 1) * eye
    return pi


def algebra_force(u_fwd: torch.Tensor, grad: torch.Tensor, is_su: bool) -> torch.Tensor:
    w = 0.5j * (u_fwd @ grad.mH - grad @ u_fwd.mH)
    if is_su:
        n = w.size(-1)
        tr = torch.einsum('eii->e', w) / n
        eye = torch.eye(n, dtype=w.dtype, device=w.device)
        w = w - tr.view(-1, 1, 1) * eye
    return w


def evolve_links(u_fwd: torch.Tensor, pi: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.matrix_exp(1j * eps * pi) @ u_fwd


def reunitarize(u_fwd: torch.Tensor, is_su: bool) -> torch.Tensor:
    # counter float32 drift off the manifold: nearest unitary via SVD, then
    # rotate the determinant phase back to 1 for SU(N)
    svd_u, _, svd_vh = torch.linalg.svd(u_fwd)
    u = svd_u @ svd_vh
    if is_su:
        n = u.size(-1)
        theta = torch.angle(torch.linalg.det(u))
        u = u * torch.exp(-1j * theta / n).view(-1, 1, 1)
    return u


def leapfrog(
    links: dict,
    scalars: dict,
    momenta: dict,
    p_scalars: dict,
    eps: float,
    n_steps: int,
    force_fn: Callable[[dict, dict], tuple[dict, dict, float]],
) -> tuple[dict, dict, dict, dict, float]:
    links = {k: v.clone() for k, v in links.items()}
    scalars = {k: v.clone() for k, v in scalars.items()}
    momenta = {k: v.clone() for k, v in momenta.items()}
    p_scalars = {k: v.clone() for k, v in p_scalars.items()}

    f_links, f_scalars, _ = force_fn(links, scalars)
    for k in momenta:
        momenta[k] -= (eps / 2) * f_links[k]
    for k in p_scalars:
        p_scalars[k] -= (eps / 2) * f_scalars[k]

    for step in range(n_steps):
        for k in links:
            links[k] = evolve_links(links[k], momenta[k], eps)
        for k in scalars:
            scalars[k] = scalars[k] + eps * p_scalars[k]

        f_links, f_scalars, S = force_fn(links, scalars)
        weight = 1.0 if step < n_steps - 1 else 0.5
        for k in momenta:
            momenta[k] -= weight * eps * f_links[k]
        for k in p_scalars:
            p_scalars[k] -= weight * eps * f_scalars[k]

    return links, scalars, momenta, p_scalars, S


class HMC:
    """Generic manifold HMC over a dict of forward gauge links and scalars."""

    def __init__(
        self,
        links: dict,
        scalars: dict,
        action_fn: Callable[[dict, dict], torch.Tensor],
        is_su: dict,
        is_fwd: torch.Tensor,
        partner_map: torch.Tensor,
        eps: float = 0.05,
        n_leapfrog: int = 10,
    ):
        self.links = {k: v.detach().clone() for k, v in links.items()}
        self.scalars = {k: v.detach().clone() for k, v in scalars.items()}
        self.action_fn = action_fn
        self.is_su = is_su
        self.eps = eps
        self.n_leapfrog = n_leapfrog

        E = is_fwd.numel()
        self.num_edges = E
        self.fwd_idx = torch.nonzero(is_fwd, as_tuple=True)[0]
        self.bwd_idx = torch.nonzero(~is_fwd, as_tuple=True)[0]
        # compact position of the partner forward link for every backward edge
        compact = torch.full((E,), -1, dtype=torch.long)
        compact[self.fwd_idx] = torch.arange(self.fwd_idx.numel())
        self.bwd_partner_compact = compact[partner_map[self.bwd_idx]]

    def u_full(self, u_fwd: torch.Tensor) -> torch.Tensor:
        n = u_fwd.size(-1)
        full = torch.zeros(self.num_edges, n, n, dtype=u_fwd.dtype, device=u_fwd.device)
        full = full.index_copy(0, self.fwd_idx, u_fwd)
        # U(-mu) = U(mu)^dag by construction, differentiably
        return full.index_copy(0, self.bwd_idx, u_fwd[self.bwd_partner_compact].mH)

    def action(self, links: dict, scalars: dict) -> torch.Tensor:
        fulls = {k: self.u_full(v) for k, v in links.items()}
        return self.action_fn(fulls, scalars)

    def _forces(self, links: dict, scalars: dict) -> tuple[dict, dict, float]:
        link_leaves = {k: v.detach().clone().requires_grad_(True) for k, v in links.items()}
        scalar_leaves = {k: v.detach().clone().requires_grad_(True) for k, v in scalars.items()}
        S = self.action(link_leaves, scalar_leaves)
        S.backward()
        f_links = {k: algebra_force(link_leaves[k].detach(), link_leaves[k].grad,
                                    self.is_su[k])
                   for k in link_leaves}
        f_scalars = {k: scalar_leaves[k].grad for k in scalar_leaves}
        return f_links, f_scalars, S.item()

    @staticmethod
    def _kinetic(momenta: dict, p_scalars: dict) -> float:
        T = 0.0
        for pi in momenta.values():
            T += 0.5 * (pi.abs() ** 2).sum().item()
        for p in p_scalars.values():
            T += 0.5 * (p.abs() ** 2).sum().item()
        return T

    @torch.no_grad()
    def trajectory(self) -> tuple[bool, float]:
        momenta = {k: sample_link_momentum(v, self.is_su[k]) for k, v in self.links.items()}
        p_scalars = {k: torch.randn_like(v) * np.sqrt(2) for k, v in self.scalars.items()}

        with torch.enable_grad():
            S0 = self.action(self.links, self.scalars).item()
        H0 = self._kinetic(momenta, p_scalars) + S0

        with torch.enable_grad():
            new_links, new_scalars, momenta, p_scalars, S1 = leapfrog(
                self.links, self.scalars, momenta, p_scalars,
                self.eps, self.n_leapfrog, self._forces)

        H1 = self._kinetic(momenta, p_scalars) + S1
        dH = H1 - H0

        accept = torch.rand(1).item() < float(np.exp(min(0.0, -dH)))
        if accept:
            self.links = {k: reunitarize(v, self.is_su[k]) for k, v in new_links.items()}
            self.scalars = new_scalars
        return accept, dH


class StandardModelHMC:
    """HMC for SU(3)xSU(2)xU(1) links + Higgs doublet + pseudofermions on the
    block-layout lattice. Sectors are optional so the same sampler covers pure
    gauge theory, gauge+Higgs, and the full system."""

    def __init__(
        self,
        lattice_shape: tuple[int, ...],
        groups: dict | None = None,
        betas: dict | None = None,
        v: float = 1.0,
        lam: float = 0.5,
        yukawa: float = 1.0,
        include_higgs: bool = False,
        include_fermions: bool = False,
        eps: float = 0.05,
        n_leapfrog: int = 10,
    ):
        groups = groups or {'su3': 3, 'su2': 2, 'u1': 1}
        betas = betas or {'su3': 6.0, 'su2': 4.0, 'u1': 5.0}
        if include_higgs and not {'su2', 'u1'} <= groups.keys():
            raise ValueError("the Higgs sector needs 'su2' and 'u1' link families")
        if include_fermions and 'su3' not in groups:
            raise ValueError("the fermion sector needs the 'su3' link family")

        self.lattice_shape = lattice_shape
        self.groups = groups
        self.include_higgs = include_higgs
        self.include_fermions = include_fermions

        edge_index, edge_dirs, is_fwd, partner_map = create_lattice(lattice_shape)
        self.edge_index = edge_index
        self.edge_dirs = edge_dirs
        self.is_fwd = is_fwd
        plaq_idx = find_rectangular_loops(lattice_shape, edge_index)

        self.wilson = {name: WilsonAction(plaq_idx, dim, betas[name])
                       for name, dim in groups.items()}
        self.plaq_idx = plaq_idx
        self.higgs_calc = HiggsAction(v=v, lam=lam) if include_higgs else None
        self.fermion_calc = PseudofermionAction(y=yukawa) if include_fermions else None

        num_nodes = int(np.prod(lattice_shape))
        n_fwd = int(is_fwd.sum().item())
        # cold start on the group identity
        links = {
            name: torch.eye(dim, dtype=torch.complex64).expand(n_fwd, dim, dim).clone()
            for name, dim in groups.items()
        }
        scalars = {}
        if include_higgs:
            scalars['phi'] = torch.randn(num_nodes, 2, dtype=torch.complex64) * 0.1
        # fermions couple to |phi|; without a dynamical Higgs the Yukawa mass
        # background is zero and only the bare mass remains
        self._phi_background = torch.zeros(num_nodes, 2, dtype=torch.complex64)
        if include_fermions:
            self.pf = torch.zeros(num_nodes, 3, 4, dtype=torch.complex64)

        self.hmc = HMC(links, scalars, self._action, {k: k != 'u1' for k in groups},
                       is_fwd, partner_map, eps=eps, n_leapfrog=n_leapfrog)

    def _phi(self, scalars: dict) -> torch.Tensor:
        return scalars['phi'] if self.include_higgs else self._phi_background

    def _action(self, fulls: dict, scalars: dict) -> torch.Tensor:
        S = sum(self.wilson[name](fulls[name]) for name in self.groups)
        if self.include_higgs:
            S = S + self.higgs_calc(scalars['phi'], self.edge_index, self.is_fwd,
                                    fulls['su2'], fulls['u1'])
        if self.include_fermions:
            S = S + self.fermion_calc(self.pf, self._phi(scalars), self.edge_index,
                                      self.edge_dirs, fulls['su3'], self.is_fwd)
        return S

    @torch.no_grad()
    def _refresh_pseudofermions(self) -> None:
        # pf = M^dag eta, eta ~ CN(0,1): S_pf = |eta|^2 at refresh, and the
        # trajectory then evolves the links against this fixed heat bath
        dirac = self.fermion_calc.dirac
        u_su3 = self.hmc.u_full(self.hmc.links['su3'])
        phi = self._phi(self.hmc.scalars)
        eta = torch.randn_like(self.pf)
        g5_eta = g5_spin_last(dirac.g5, eta)
        d_eta = dirac(g5_eta, phi, self.edge_index, self.edge_dirs, u_su3, self.is_fwd)
        self.pf.copy_(g5_spin_last(dirac.g5, d_eta))

    def trajectory(self) -> tuple[bool, float]:
        if self.include_fermions:
            self._refresh_pseudofermions()
        return self.hmc.trajectory()

    def average_plaquette(self, name: str) -> float:
        from src.measure import average_plaquette
        u = self.hmc.u_full(self.hmc.links[name])
        return average_plaquette(u, self.plaq_idx, self.groups[name])

    def full_links(self, sample: dict) -> dict:
        """Reconstruct full (forward + backward) edge tensors from a stored
        HMC sample, for ensemble observables."""
        return {name: self.hmc.u_full(sample[name]) for name in self.groups}

    def run(
        self,
        n_traj: int,
        warmup: int = 0,
        sample_every: int = 1,
        log_every: int = 50,
        tune_eps: bool = True,
    ) -> dict:
        history = {'dH': [], 'accept': [], 'plaquette': {k: [] for k in self.groups}}
        samples = []

        for t in range(n_traj):
            accept, dH = self.trajectory()
            history['dH'].append(dH)
            history['accept'].append(float(accept))

            # adapt the step size during warmup only, so the sampling phase is
            # a valid fixed-parameter chain; this also recovers from the
            # cold-start trap where a too-coarse eps rejects every trajectory
            if tune_eps and t < warmup and (t + 1) % 10 == 0:
                recent = float(np.mean(history['accept'][-10:]))
                if recent > 0.85:
                    self.hmc.eps *= 1.15
                elif recent < 0.6:
                    self.hmc.eps *= 0.7

            if t >= warmup:
                for name in self.groups:
                    history['plaquette'][name].append(self.average_plaquette(name))
                if (t - warmup) % sample_every == 0:
                    sample = {name: v.clone() for name, v in self.hmc.links.items()}
                    sample.update({k: v.clone() for k, v in self.hmc.scalars.items()})
                    samples.append(sample)

            if log_every and t % log_every == 0:
                acc = float(np.mean(history['accept'][-log_every:]))
                print(f"[HMC] traj {t:04d} | dH: {dH:+.4f} | acc(recent): {acc:.2f}")

        history['acceptance_rate'] = float(np.mean(history['accept']))
        history['samples'] = samples
        return history
