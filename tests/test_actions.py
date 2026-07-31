import numpy as np
import torch
from helpers import gauge_transform_links, random_su

from src.actions import HiggsAction, WilsonAction, higgs_potential_loss, wilson_plaquette_loss
from src.groups import get_gate
from src.lattice import (
    create_lattice,
    find_rectangular_loops,
    get_pbc_edge_index,
    get_plaquette_indices,
)

SHAPE = (4, 4)
V = int(np.prod(SHAPE))


def test_wilson_action_zero_on_cold_vacuum() -> None:
    edge_index, _, _, _ = create_lattice(SHAPE)
    plaq = find_rectangular_loops(SHAPE, edge_index)
    calc = WilsonAction(plaq, group_dim=3, beta=2.5)
    u = torch.eye(3, dtype=torch.complex64).expand(edge_index.size(1), 3, 3)
    assert abs(calc(u).item()) < 1e-4


def test_wilson_action_gauge_invariant() -> None:
    torch.manual_seed(0)
    edge_index, _, is_fwd, partner = create_lattice(SHAPE)
    plaq = find_rectangular_loops(SHAPE, edge_index)
    calc = WilsonAction(plaq, group_dim=2, beta=1.7)
    raw = torch.randn(edge_index.size(1), 2, 2, dtype=torch.complex64)
    u = get_gate(raw, partner, is_fwd, is_su=True)
    g = random_su(2, V, seed=7)
    u_t = gauge_transform_links(u, g, edge_index)
    assert torch.allclose(calc(u), calc(u_t), rtol=1e-4)


def test_higgs_action_gauge_invariant() -> None:
    torch.manual_seed(0)
    edge_index, _, is_fwd, partner = create_lattice(SHAPE)
    phi = torch.randn(V, 2, dtype=torch.complex64)
    u2 = get_gate(torch.randn(edge_index.size(1), 2, 2, dtype=torch.complex64),
                  partner, is_fwd, is_su=True)
    u1 = get_gate(torch.randn(edge_index.size(1), 1, 1, dtype=torch.complex64),
                  partner, is_fwd, is_su=False)
    calc = HiggsAction(v=1.0, lam=0.3)
    s = calc(phi, edge_index, is_fwd, u2, u1)
    g = random_su(2, V, seed=5)
    phi_t = torch.einsum('nij,nj->ni', g, phi)
    u2_t = gauge_transform_links(u2, g, edge_index)
    s_t = calc(phi_t, edge_index, is_fwd, u2_t, u1)
    assert torch.allclose(s, s_t, rtol=1e-4)


def test_plaquette_loss_zero_on_cold_vacuum_and_invariant() -> None:
    ei, _ = get_pbc_edge_index(3, 2, 'cpu')
    p1, p2, p3, p4 = get_plaquette_indices(3, 2, ei, 'cpu')
    E = ei.size(1)
    u_id = torch.eye(2, dtype=torch.complex64).expand(E, 2, 2)
    assert abs(wilson_plaquette_loss(u_id, p1, p2, p3, p4).item()) < 1e-6

    torch.manual_seed(1)
    h = torch.randn(E, 2, 2, dtype=torch.complex64)
    u = torch.matrix_exp(1j * (h + h.mH))
    g = random_su(2, 9, seed=3)
    u_t = gauge_transform_links(u, g, ei)
    assert torch.allclose(wilson_plaquette_loss(u, p1, p2, p3, p4),
                          wilson_plaquette_loss(u_t, p1, p2, p3, p4), rtol=1e-4)


def test_higgs_potential_minimum_at_target_vev() -> None:
    v = 1.3
    phi_min = torch.zeros(10, 4, 2, dtype=torch.complex64)
    phi_min[..., 1] = v / np.sqrt(2)  # |phi|^2 = v^2 / 2
    loss_min, mag = higgs_potential_loss(phi_min, v, 0.5)
    assert abs(loss_min.item()) < 1e-6
    assert abs(mag.item() - v ** 2 / 2) < 1e-5
    loss_off, _ = higgs_potential_loss(phi_min * 1.5, v, 0.5)
    assert loss_off.item() > loss_min.item()
