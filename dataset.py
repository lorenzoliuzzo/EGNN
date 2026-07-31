import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.utils.lattice import get_pbc_edge_index
from src.physics.gauge_groups import get_sm_generators, exp_sm_algebra_to_group




def generate_single_snapshot2(args):
    """
    Worker function to generate a single N-Dimensional universe snapshot.
    """
    # 1. Use the full ALL_SPECIES list we defined
    idx, L, dims, hidden_dim, save_dir, sm_gen_dict, edge_index = args
    

    # 1. CRITICAL: Re-seed the random number generators for each process!
    torch.manual_seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)
    torch.set_num_threads(1)

    n_nodes = L ** dims
    n_forward_edges = dims * n_nodes  # Only D forward directions per node

    species_dims = {
        'quark_left': 6, 'quark_up_right': 3, 'quark_down_right': 3,
        'lepton_left': 2, 'lepton_right': 1, 'higgs': 2
    }

    # Vary the "Temperature" across the dataset
    is_cold = (idx % 2 == 0) 
    scale = 0.05 if is_cold else np.random.uniform(0.5, 2.0)
    
    # -------------------------------------------------------------------------
    # 2. GENERATE GAUGE LINKS (PHYSICALLY CORRECT)
    # -------------------------------------------------------------------------
    # We ONLY generate random algebra for the FORWARD links
    a_fwd_dict = {
        'su3': torch.randn(n_forward_edges, 8) * scale,
        'su2': torch.randn(n_forward_edges, 3) * scale,
        'u1':  torch.randn(n_forward_edges, 1) * scale
    }
    
    # Exponentiate forward links
    u_fwd_su3, u_fwd_su2, u_fwd_u1 = exp_sm_algebra_to_group(a_fwd_dict, sm_gen_dict)
    
    # Helper to interleave forward links and their Hermitian Adjoints (backward links)
    def make_bidirectional(u_fwd):
        N = u_fwd.size(-1)
        # Create tensor twice the size to hold both directions
        u_full = torch.zeros((n_forward_edges * 2, N, N), dtype=u_fwd.dtype)
        # Assuming edge_index alternates [fwd, bwd, fwd, bwd...]
        u_full[0::2] = u_fwd          # Evens are Forward
        u_full[1::2] = u_fwd.mH       # Odds are Adjoints (Backward)
        return u_full

    u_dict = {
        'su3': make_bidirectional(u_fwd_su3),
        'su2': make_bidirectional(u_fwd_su2),
        'u1':  make_bidirectional(u_fwd_u1)
    }

    # -------------------------------------------------------------------------
    # 3. GENERATE MATTER FIELDS
    # -------------------------------------------------------------------------
    z_dict = {}
    for name, dim in species_dims.items():
        z_real = torch.randn(n_nodes, hidden_dim, dim, dtype=torch.float32)
        z_imag = torch.randn(n_nodes, hidden_dim, dim, dtype=torch.float32)
        z_dict[name] = torch.complex(z_real, z_imag)

    # -------------------------------------------------------------------------
    # 4. PACKAGE AND SAVE
    # -------------------------------------------------------------------------
    edge_index_dict = {
        (name, 'transport', name): edge_index
        for name in species_dims.keys()
    }

    snapshot = {
        'z_dict': z_dict,
        'u_dict': u_dict,
        'edge_index_dict': edge_index_dict,
        'temperature': scale
    }
    
    file_path = os.path.join(save_dir, f"sample_{idx:04d}.pt")
    torch.save(snapshot, file_path)
    
    return idx



def generate_single_snapshot(args):
    """
    Generates a single N-Dimensional universe snapshot with 3 generations of matter.
    """
    idx, L, dims, hidden_dim, save_dir, sm_gen_dict, edge_index, edge_dirs = args
    
    assert hidden_dim % 4 == 0, "hidden_dim must be a multiple of 4 for Wilson-Dirac physics"

    # 1. Seed isolation for multiprocessing
    torch.manual_seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)
    torch.set_num_threads(1)

    n_nodes = L ** dims    
    n_channels = hidden_dim // 4 
    n_forward_edges = dims * n_nodes  # D forward links per node

    # Define the 18 matter species + Higgs
    # Values represent the complex vector dimension (group_dim)
    base_dims = {
        'quark_left': 6,          # 3(color) * 2(weak)
        'quark_up_right': 3,     # 3(color) * 1(weak)
        'quark_down_right': 3,   # 3(color) * 1(weak)
        'lepton_left': 2,        # 1(color) * 2(weak)
        'lepton_right': 1,       # 1(color) * 1(weak)
        'lepton_nu_right': 1     # 1(color) * 1(weak)
    }

    # Vary the "Temperature" for data diversity
    is_cold = (idx % 2 == 0) 
    scale = 0.05 if is_cold else np.random.uniform(0.5, 2.0)
    
    # -------------------------------------------------------------------------
    # 2. GENERATE GAUGE LINKS (The infrastructure of the forces)
    # -------------------------------------------------------------------------
    a_fwd_dict = {
        'su3': torch.randn(n_forward_edges, 8) * scale,
        'su2': torch.randn(n_forward_edges, 3) * scale,
        'u1':  torch.randn(n_forward_edges, 1) * scale
    }
    
    # Transform algebra to Group Elements U = exp(iA)
    u_fwd_su3, u_fwd_su2, u_fwd_u1 = exp_sm_algebra_to_group(a_fwd_dict, sm_gen_dict)
    
    def make_bidirectional(u_fwd):
        # Interleave Forward and Backward links
        N = u_fwd.size(-1)
        u_full = torch.zeros((n_forward_edges * 2, N, N), dtype=u_fwd.dtype)
        u_full[0::2] = u_fwd          # Evens: Forward
        u_full[1::2] = u_fwd.mH       # Odds: Backward (Adjoint)
        return u_full

    u_dict = {
        'su3': make_bidirectional(u_fwd_su3),
        'su2': make_bidirectional(u_fwd_su2),
        'u1':  make_bidirectional(u_fwd_u1)
    }

    # -------------------------------------------------------------------------
    # 3. GENERATE MATTER FIELDS with Spin Dimension
    # -------------------------------------------------------------------------
    z_dict = {}
    
    for g in [1, 2, 3]:
        for base, group_dim in base_dims.items():
            name = f"{base}_g{g}"
            # NEW SHAPE: [nodes, channels, 4, group_dim]
            z_real = torch.randn(n_nodes, n_channels, 4, group_dim)
            z_imag = torch.randn(n_nodes, n_channels, 4, group_dim)
            z_dict[name] = torch.complex(z_real, z_imag)
            
    # Higgs is a scalar (no spin), but for GNN consistency, 
    # we can give it a dummy spin dim of 1 or keep it as is.
    # Let's keep it [nodes, hidden_dim, 2] and handle it in the GNN.
    z_higgs_real = torch.randn(n_nodes, hidden_dim, 2)
    z_higgs_imag = torch.randn(n_nodes, hidden_dim, 2)
    z_dict['higgs'] = torch.complex(z_higgs_real, z_higgs_imag)

    # -------------------------------------------------------------------------
    # 4. PREPARE THE GRAPH DICTIONARIES
    # -------------------------------------------------------------------------
    # Every species needs an entry for PyG to route correctly
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
        'temperature': scale
    }
    
    file_path = os.path.join(save_dir, f"sample_{idx:04d}.pt")
    torch.save(snapshot, file_path)
    
    return idx

def generate_and_save_dataset_parallel(num_samples, L, dims, hidden_dim, save_dir="sm_dataset"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Generating {num_samples} Snapshots (L={L}, D={dims}) using {num_cores} cores...")

    # Pre-compute shared structures on the main process
    device = 'cpu'
    
    # Make sure this is the N-dimensional version we wrote earlier!
    edge_index, edge_dirs = get_pbc_edge_index(L, dims, 'cpu')
    sm_gen_dict = get_sm_generators(device)

    # Package arguments (now passing dims!)
    tasks = [
        (i, L, dims, hidden_dim, save_dir, sm_gen_dict, edge_index, edge_dirs) 
        for i in range(num_samples)
    ]
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(generate_single_snapshot, task) for task in tasks]
        for _ in tqdm(as_completed(futures), total=num_samples, desc="Generating"):
            pass 

    print("\nDataset Generation Complete!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    LATTICE_L = 8
    LATTICE_DIMS = 4
    
    generate_and_save_dataset_parallel(
        num_samples=100, 
        L=LATTICE_L, 
        dims=LATTICE_DIMS, 
        hidden_dim=16,
        save_dir=f"data/L{LATTICE_L}_{LATTICE_DIMS}D"
    )