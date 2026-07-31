import torch

import SM


def test_langevin_step_clips_large_forces(monkeypatch):
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.complex64))
    p.grad = torch.full((4,), 1e6 + 0j, dtype=torch.complex64)
    monkeypatch.setattr(torch, "randn_like", lambda t: torch.zeros_like(t))
    dt = 0.01
    # langevin_step does not touch self, so it can be exercised unbound
    SM.HeteroQuantumVacuum.langevin_step(None, [p], dt=dt)
    assert torch.norm(p.detach()) <= dt * 10.0 * (1 + 1e-5)


def test_legacy_wilson_dirac_gamma5_hermitian(legacy):
    tm = legacy
    torch.manual_seed(0)
    ei, ed, isf, pm = tm.create_lattice((4, 4))
    u = tm.get_gate(torch.randn(ei.size(1), 3, 3, dtype=torch.complex64), pm, isf)
    dirac = tm.WilsonDiracOperator(y=0.5, bare_mass=0.05)
    phi = torch.randn(16, 2, dtype=torch.complex64) * 0.3
    x = torch.randn(16, 3, 4, dtype=torch.complex64)
    y = torch.randn(16, 3, 4, dtype=torch.complex64)
    g5 = dirac.g5

    def g5_mul(v):
        return torch.einsum('ss, ecs -> ecs', g5, v)

    lhs = torch.sum(torch.conj(y) * dirac(x, phi, ei, ed, u, isf))
    rhs = torch.sum(torch.conj(g5_mul(dirac(g5_mul(y), phi, ei, ed, u, isf))) * x)
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_legacy_pseudofermion_heat_bath_identity(legacy):
    # For pf = M^dag eta:  S = pf^dag (M^dag M)^-1 pf = |eta|^2 exactly,
    # which is what makes the heat-bath sampling represent det(M^dag M)
    tm = legacy
    torch.manual_seed(0)
    ei, ed, isf, pm = tm.create_lattice((4, 4))
    u = tm.get_gate(torch.randn(ei.size(1), 3, 3, dtype=torch.complex64), pm, isf)
    dirac = tm.WilsonDiracOperator(y=0.5, bare_mass=0.05)
    phi = torch.randn(16, 2, dtype=torch.complex64) * 0.3
    eta = torch.randn(16, 3, 4, dtype=torch.complex64)
    g5 = dirac.g5

    def g5_mul(v):
        return torch.einsum('ss, ecs -> ecs', g5, v)

    pf = g5_mul(dirac(g5_mul(eta), phi, ei, ed, u, isf))
    x = tm.conjugate_gradient(dirac, pf, phi, ei, ed, u, isf,
                              max_iter=1000, tol=1e-8)
    S = torch.sum(pf.conj() * x).real
    expected = torch.sum(eta.abs() ** 2)
    assert abs(S - expected) / expected < 5e-3
