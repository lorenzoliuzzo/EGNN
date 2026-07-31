import itertools

import numpy as np
import torch

# Two graph layouts coexist and are NOT interchangeable:
#
# - create_lattice: edges grouped into (direction, orientation) blocks —
#   d0-forward, d0-backward, d1-forward, ... — each block in node order, plus an
#   explicit partner map. find_rectangular_loops relies on the block layout via
#   e = d*2V + node (forward) and e = d*2V + V + node (backward). Used by the
#   vacuum finders.
#
# - get_pbc_edge_index: forward/backward links interleaved per (node, dim), even
#   indices forward. This ordering is a file-format contract: the datasets under
#   data/ store gauge links in it. Used by the flow pipeline.


def create_lattice(
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dims = len(shape)
    num_nodes = int(np.prod(shape))
    grid = torch.arange(num_nodes).reshape(shape)

    srcs, dsts, dirs_, fwds = [], [], [], []
    for d in range(dims):
        for is_forward in (True, False):
            # roll(-1) pairs each node with its +1 neighbour along d (periodic)
            shift = -1 if is_forward else 1
            neighbor = torch.roll(grid, shifts=shift, dims=d)
            srcs.append(grid.flatten())
            dsts.append(neighbor.flatten())
            dirs_.append(torch.full((num_nodes,), d, dtype=torch.long))
            fwds.append(torch.full((num_nodes,), is_forward, dtype=torch.bool))

    edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)])
    edge_dirs = torch.cat(dirs_)
    is_fwd = torch.cat(fwds)

    # The reverse of forward edge (n -> m) is the backward edge whose source is
    # m, which lives at offset V inside the same direction block (and vice versa)
    partners = []
    for d in range(dims):
        base = d * 2 * num_nodes
        fwd_dst = edge_index[1, base:base + num_nodes]
        bwd_dst = edge_index[1, base + num_nodes:base + 2 * num_nodes]
        partners.append(base + num_nodes + fwd_dst)
        partners.append(base + bwd_dst)
    partner_map = torch.cat(partners)

    return edge_index, edge_dirs, is_fwd, partner_map


def find_rectangular_loops(
    lattice_shape: tuple[int, ...],
    edge_index: torch.Tensor,
    R: int = 1,
    T: int = 1,
) -> torch.Tensor:
    dims = len(lattice_shape)
    V = int(np.prod(lattice_shape))
    device = edge_index.device
    all_loops = []

    base_nodes = torch.arange(V, dtype=torch.long, device=device)

    for mu in range(dims):
        for nu in range(mu + 1, dims):
            orientations = [(mu, nu, R, T)]
            if R != T:
                orientations.append((mu, nu, T, R))

            for dir1, dir2, len1, len2 in orientations:
                current_nodes = base_nodes.clone()
                loop_edges = []

                # walk R steps along dir1 then T along dir2 on forward edges,
                # then return on the backward edges: a closed R x T rectangle
                for d, length in [(dir1, len1), (dir2, len2)]:
                    for _ in range(length):
                        e = (d * 2 * V) + current_nodes
                        loop_edges.append(e)
                        current_nodes = edge_index[1, e]

                for d, length in [(dir1, len1), (dir2, len2)]:
                    for _ in range(length):
                        e = (d * 2 * V) + V + current_nodes
                        loop_edges.append(e)
                        current_nodes = edge_index[1, e]

                all_loops.append(torch.stack(loop_edges, dim=1))

    return torch.cat(all_loops, dim=0)


def get_pbc_edge_index(
    L: int, dimensions: int, device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = L ** dimensions
    edges = []
    edge_dirs = []

    strides = [L ** i for i in range(dimensions)]

    for i in range(num_nodes):
        for d in range(dimensions):
            coord_d = (i // strides[d]) % L

            if coord_d < L - 1:
                neighbor = i + strides[d]
            else:
                neighbor = i - (L - 1) * strides[d]

            edges.append([i, neighbor])
            edge_dirs.append(d)
            edges.append([neighbor, i])
            edge_dirs.append(d)

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()
    return edge_index, torch.tensor(edge_dirs, dtype=torch.long, device=device)


def get_plaquette_indices(
    L: int, dimensions: int, edge_index: torch.Tensor, device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = L ** dimensions
    strides = [L ** i for i in range(dimensions)]

    edge_index_cpu = edge_index.cpu()
    edge_map = {(src.item(), dst.item()): idx
                for idx, (src, dst) in enumerate(edge_index_cpu.T)}

    p1_list, p2_list, p3_list, p4_list = [], [], [], []

    for i in range(num_nodes):
        for d1, d2 in itertools.combinations(range(dimensions), 2):

            def get_neighbor(node_idx: int, dim_idx: int) -> int:
                coord_d = (node_idx // strides[dim_idx]) % L
                if coord_d < L - 1:
                    return node_idx + strides[dim_idx]
                return node_idx - (L - 1) * strides[dim_idx]

            n00 = i
            n10 = get_neighbor(n00, d1)
            n01 = get_neighbor(n00, d2)
            n11 = get_neighbor(n10, d2)

            # all four stored as forward edges; the action takes adjoints of
            # p3/p4 when composing the loop n00 -> n10 -> n11 -> n01 -> n00
            p1_list.append(edge_map[(n00, n10)])
            p2_list.append(edge_map[(n10, n11)])
            p3_list.append(edge_map[(n01, n11)])
            p4_list.append(edge_map[(n00, n01)])

    return (
        torch.tensor(p1_list, device=device),
        torch.tensor(p2_list, device=device),
        torch.tensor(p3_list, device=device),
        torch.tensor(p4_list, device=device),
    )
