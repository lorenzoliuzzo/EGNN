import pytest
import torch

from src.dirac import WilsonDiracOperator, cg_solve, conjugate_gradient, g5_spin_last
from src.groups import get_gate
from src.lattice import create_lattice
from src.standard_model import HeteroQuantumVacuum


def test_langevin_step_clips_large_forces(monkeypatch: pytest.MonkeyPatch) -> None:
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.complex64))
    p.grad = torch.full((4,), 1e6 + 0j, dtype=torch.complex64)
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))
    dt = 0.01
    # langevin_step does not touch self, so it can be exercised unbound
    HeteroQuantumVacuum.langevin_step(None, [p], dt=dt)
    assert torch.norm(p.detach()) <= dt * 10.0 * (1 + 1e-5)


def _qcd_setup() -> tuple:
    torch.manual_seed(0)
    ei, ed, isf, pm = create_lattice((4, 4))
    u = get_gate(torch.randn(ei.size(1), 3, 3, dtype=torch.complex64), pm, isf)
    dirac = WilsonDiracOperator(y=0.5, bare_mass=0.05)
    phi = torch.randn(16, 2, dtype=torch.complex64) * 0.3
    return dirac, phi, ei, ed, u, isf


def test_standalone_dirac_gamma5_hermitian() -> None:
    dirac, phi, ei, ed, u, isf = _qcd_setup()
    x = torch.randn(16, 3, 4, dtype=torch.complex64)
    y = torch.randn(16, 3, 4, dtype=torch.complex64)
    g5 = dirac.g5
    lhs = torch.sum(torch.conj(y) * dirac(x, phi, ei, ed, u, isf))
    rhs = torch.sum(torch.conj(
        g5_spin_last(g5, dirac(g5_spin_last(g5, y), phi, ei, ed, u, isf))) * x)
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_pseudofermion_heat_bath_identity() -> None:
    # For pf = M^dag eta:  S = pf^dag (D^dag D)^-1 pf = |eta|^2 exactly, which
    # is what makes the heat-bath sampling represent det(M^dag M)
    dirac, phi, ei, ed, u, isf = _qcd_setup()
    eta = torch.randn(16, 3, 4, dtype=torch.complex64)
    g5 = dirac.g5
    pf = g5_spin_last(g5, dirac(g5_spin_last(g5, eta), phi, ei, ed, u, isf))
    x = conjugate_gradient(dirac, pf, phi, ei, ed, u, isf, max_iter=1000, tol=1e-8)
    S = torch.sum(pf.conj() * x).real
    expected = torch.sum(eta.abs() ** 2)
    assert abs(S - expected) / expected < 5e-3


def test_generic_cg_matches_direct_solve() -> None:
    torch.manual_seed(0)
    m = torch.randn(12, 12, dtype=torch.complex64)
    A = m @ m.mH + 0.5 * torch.eye(12, dtype=torch.complex64)
    b = torch.randn(12, dtype=torch.complex64)
    x = cg_solve(lambda v: A @ v, b, tol=1e-8, max_iter=200)
    assert torch.allclose(A @ x, b, atol=1e-4)
