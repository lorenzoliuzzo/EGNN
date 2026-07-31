import numpy as np
import torch

from src.actions import ElectroweakHiggsAction
from src.groups import get_gate
from src.lattice import create_lattice
from src.measure import align_to_unitary_gauge, measure_electroweak_masses


def _random_cfg(shape: tuple[int, ...] = (4, 4), seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    edge_index, _, is_fwd, partner = create_lattice(shape)
    n = int(np.prod(shape))
    E = edge_index.size(1)
    phi = torch.randn(n, 2, dtype=torch.complex64)
    u2 = get_gate(torch.randn(E, 2, 2, dtype=torch.complex64), partner, is_fwd, is_su=True)
    u1 = get_gate(torch.randn(E, 1, 1, dtype=torch.complex64), partner, is_fwd, is_su=False)
    return edge_index, is_fwd, phi, u2, u1


def test_alignment_rotates_phi_to_unitary_gauge() -> None:
    edge_index, _, phi, u2, u1 = _random_cfg()
    pa, ua, _ = align_to_unitary_gauge(phi, edge_index, u2, u1)
    assert pa[:, 0].abs().max() < 1e-5
    assert torch.allclose(pa[:, 1].real, torch.norm(phi, dim=-1), atol=1e-5)
    eye = torch.eye(2, dtype=torch.complex64).expand_as(ua)
    assert torch.allclose(ua @ ua.mH, eye, atol=1e-4)


def test_alignment_preserves_the_action() -> None:
    edge_index, is_fwd, phi, u2, u1 = _random_cfg(seed=2)
    calc = ElectroweakHiggsAction(mu_sq=1.3, lam=0.4)
    s0 = calc(phi, edge_index, is_fwd, u2, u1).item()
    pa, ua, u1a = align_to_unitary_gauge(phi, edge_index, u2, u1)
    s1 = calc(pa, edge_index, is_fwd, ua, u1a).item()
    assert abs(s0 - s1) / abs(s0) < 1e-3


def test_ideal_vacuum_boson_masses() -> None:
    shape = (6, 6)
    edge_index, _, is_fwd, _ = create_lattice(shape)
    n, E = int(np.prod(shape)), edge_index.size(1)
    v = 1.0
    phi = torch.zeros(n, 2, dtype=torch.complex64)
    phi[:, 1] = v
    u2 = torch.eye(2, dtype=torch.complex64).expand(E, 2, 2)
    u1 = torch.ones(E, 1, 1, dtype=torch.complex64)
    calc = ElectroweakHiggsAction(mu_sq=1.0, lam=0.5)
    stats = measure_electroweak_masses(phi, u2, u1, edge_index, is_fwd, calc,
                                       g_su2=1.0, g_u1=0.5)
    masses = stats['masses']
    # the photon generator leaves phi = [0, v] invariant: exactly massless
    assert masses['Photon'] < 1e-3
    assert masses['W Boson'] > 0.5
    assert masses['Z Boson'] > masses['W Boson']
    # single-doublet custodial symmetry at tree level
    assert abs(stats['rho'] - 1.0) < 0.05
