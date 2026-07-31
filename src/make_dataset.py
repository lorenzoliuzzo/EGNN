import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.groups import exp_sm_algebra_to_group, get_sm_generators
from src.lattice import get_pbc_edge_index


def generate_single_snapshot(args: tuple) -> int:
    """One random N-D universe snapshot with 3 generations of matter."""
    idx, L, dims, hidden_dim, save_dir, sm_gen_dict, edge_index, edge_dirs = args

    assert hidden_dim % 4 == 0, "hidden_dim must be a multiple of 4 for Wilson-Dirac physics"

    # each worker process must be re-seeded or every sample comes out identical
    torch.manual_seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)
    torch.set_num_threads(1)

    n_nodes = L ** dims
    n_channels = hidden_dim // 4
    n_forward_edges = dims * n_nodes

    # complex vector dimension per species: color x weak isospin
    base_dims = {
        'quark_left': 6,
        'quark_up_right': 3,
        'quark_down_right': 3,
        'lepton_left': 2,
        'lepton_right': 1,
        'lepton_nu_right': 1,
    }

    # alternate cold / hot samples for data diversity
    is_cold = (idx % 2 == 0)
    scale = 0.05 if is_cold else np.random.uniform(0.5, 2.0)

    a_fwd_dict = {
        'su3': torch.randn(n_forward_edges, 8) * scale,
        'su2': torch.randn(n_forward_edges, 3) * scale,
        'u1': torch.randn(n_forward_edges, 1) * scale,
    }
    u_fwd_su3, u_fwd_su2, u_fwd_u1 = exp_sm_algebra_to_group(a_fwd_dict, sm_gen_dict)

    def make_bidirectional(u_fwd: torch.Tensor) -> torch.Tensor:
        # interleave forward links and their adjoints, matching the even=forward
        # ordering of get_pbc_edge_index (a file-format contract)
        N = u_fwd.size(-1)
        u_full = torch.zeros((n_forward_edges * 2, N, N), dtype=u_fwd.dtype)
        u_full[0::2] = u_fwd
        u_full[1::2] = u_fwd.mH
        return u_full

    u_dict = {
        'su3': make_bidirectional(u_fwd_su3),
        'su2': make_bidirectional(u_fwd_su2),
        'u1': make_bidirectional(u_fwd_u1),
    }

    z_dict = {}
    for g in [1, 2, 3]:
        for base, group_dim in base_dims.items():
            name = f"{base}_g{g}"
            z_real = torch.randn(n_nodes, n_channels, 4, group_dim)
            z_imag = torch.randn(n_nodes, n_channels, 4, group_dim)
            z_dict[name] = torch.complex(z_real, z_imag)

    z_higgs_real = torch.randn(n_nodes, hidden_dim, 2)
    z_higgs_imag = torch.randn(n_nodes, hidden_dim, 2)
    z_dict['higgs'] = torch.complex(z_higgs_real, z_higgs_imag)

    edge_index_dict = {}
    edge_dirs_dict = {}
    for species_name in z_dict.keys():
        key = (species_name, 'transport', species_name)
        edge_index_dict[key] = edge_index
        edge_dirs_dict[key] = edge_dirs

    snapshot = {
        'z_dict': z_dict,
        'u_dict': u_dict,
        'edge_index_dict': edge_index_dict,
        'edge_dirs_dict': edge_dirs_dict,
        'temperature': scale,
    }

    torch.save(snapshot, Path(save_dir) / f"sample_{idx:04d}.pt")
    return idx


def generate_and_save_dataset_parallel(
    num_samples: int, L: int, dims: int, hidden_dim: int,
    save_dir: str = "sm_dataset",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Generating {num_samples} snapshots (L={L}, D={dims}) on {num_cores} cores...")

    edge_index, edge_dirs = get_pbc_edge_index(L, dims, 'cpu')
    sm_gen_dict = get_sm_generators('cpu')

    tasks = [
        (i, L, dims, hidden_dim, save_dir, sm_gen_dict, edge_index, edge_dirs)
        for i in range(num_samples)
    ]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(generate_single_snapshot, task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=num_samples, desc="Generating"):
            pass

    print("\nDataset generation complete.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    generate_and_save_dataset_parallel(
        num_samples=100, L=8, dims=4, hidden_dim=16, save_dir="data/L8_4D")
