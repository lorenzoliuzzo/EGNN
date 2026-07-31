import torch

import SM
from src.utils.lattice import get_pbc_edge_index, get_plaquette_indices


def test_partner_map_involution_and_reversal():
    edge_index, edge_dirs, is_fwd, partner = SM.create_lattice((4, 3))
    e = torch.arange(edge_index.size(1))
    assert torch.equal(partner[partner], e)
    assert torch.equal(edge_index[0, partner], edge_index[1])
    assert torch.equal(edge_index[1, partner], edge_index[0])
    assert torch.equal(is_fwd[partner], ~is_fwd)
    assert torch.equal(edge_dirs[partner], edge_dirs)


def test_edge_block_layout_contract():
    # find_rectangular_loops assumes edge d*2V + n is the forward link of node n
    # in direction d, and d*2V + V + n its backward link (requires stable sort)
    shape = (4, 4)
    V = 16
    edge_index, edge_dirs, is_fwd, _ = SM.create_lattice(shape)
    nodes = torch.arange(V)
    for d in range(len(shape)):
        fwd = slice(d * 2 * V, d * 2 * V + V)
        bwd = slice(d * 2 * V + V, (d + 1) * 2 * V)
        assert torch.equal(edge_index[0, fwd], nodes)
        assert torch.equal(edge_index[0, bwd], nodes)
        assert is_fwd[fwd].all() and (~is_fwd[bwd]).all()
        assert (edge_dirs[fwd] == d).all() and (edge_dirs[bwd] == d).all()


def test_rectangular_loops_are_closed_paths():
    shape = (4, 4)
    edge_index, *_ = SM.create_lattice(shape)
    for R, T in [(1, 1), (2, 1), (2, 3)]:
        loops = SM.find_rectangular_loops(shape, edge_index, R=R, T=T)
        src, dst = edge_index[0, loops], edge_index[1, loops]
        assert torch.equal(dst[:, :-1], src[:, 1:]), (R, T)
        assert torch.equal(dst[:, -1], src[:, 0]), (R, T)


def test_pbc_edges_interleaved_fwd_bwd():
    ei, ed = get_pbc_edge_index(4, 2, 'cpu')
    assert torch.equal(ei[0, 0::2], ei[1, 1::2])
    assert torch.equal(ei[1, 0::2], ei[0, 1::2])
    assert torch.equal(ed[0::2], ed[1::2])


def test_plaquette_indices_form_closed_squares():
    ei, _ = get_pbc_edge_index(3, 2, 'cpu')
    p1, p2, p3, p4 = get_plaquette_indices(3, 2, ei, 'cpu')
    # p1: n00->n10, p2: n10->n11, p3: n01->n11, p4: n00->n01
    assert torch.equal(ei[1, p1], ei[0, p2])
    assert torch.equal(ei[0, p1], ei[0, p4])
    assert torch.equal(ei[1, p4], ei[0, p3])
    assert torch.equal(ei[1, p2], ei[1, p3])
