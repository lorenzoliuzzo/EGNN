import torch
from helpers import is_unitary

from src.groups import SMGateFactory, exp_sm_algebra_to_group, get_gate, get_sm_generators
from src.lattice import create_lattice


def test_get_gate_unitary_su_det_and_reversal() -> None:
    torch.manual_seed(0)
    edge_index, _, is_fwd, partner = create_lattice((4, 4))
    raw = torch.randn(edge_index.size(1), 3, 3, dtype=torch.complex64)
    u = get_gate(raw, partner, is_fwd, is_su=True)
    assert is_unitary(u)
    ones = torch.ones(u.size(0), dtype=torch.complex64)
    assert torch.allclose(torch.linalg.det(u), ones, atol=1e-4)
    # U(-mu) = U(mu)^dag
    assert torch.allclose(u[partner], u.mH, atol=1e-5)


def test_get_gate_u1_is_pure_phase() -> None:
    torch.manual_seed(0)
    edge_index, _, is_fwd, partner = create_lattice((4, 4))
    raw = torch.randn(edge_index.size(1), 1, 1, dtype=torch.complex64)
    u = get_gate(raw, partner, is_fwd, is_su=False)
    assert torch.allclose(u.abs(), torch.ones_like(u.abs()), atol=1e-5)


def test_exp_algebra_produces_group_elements() -> None:
    torch.manual_seed(0)
    gens = get_sm_generators('cpu')
    a = {'su3': torch.randn(10, 8), 'su2': torch.randn(10, 3), 'u1': torch.randn(10, 1)}
    u3, u2, u1 = exp_sm_algebra_to_group(a, gens)
    assert is_unitary(u3) and is_unitary(u2)
    ones = torch.ones(10, dtype=torch.complex64)
    assert torch.allclose(torch.linalg.det(u3), ones, atol=1e-4)
    assert torch.allclose(torch.linalg.det(u2), ones, atol=1e-4)
    assert torch.allclose(u1.abs(), torch.ones_like(u1.abs()), atol=1e-5)


def test_sm_gate_factory_gates_are_unitary() -> None:
    torch.manual_seed(0)
    gens = get_sm_generators('cpu')
    a = {'su3': torch.randn(6, 8), 'su2': torch.randn(6, 3), 'u1': torch.randn(6, 1)}
    u3, u2, u1 = exp_sm_algebra_to_group(a, gens)
    gates = SMGateFactory({'su3': u3, 'su2': u2, 'u1': u1}).get_all_gates()
    for key, g in gates.items():
        assert is_unitary(g), key
