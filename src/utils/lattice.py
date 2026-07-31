import torch 
import itertools

def get_pbc_edge_index(L, dimensions, device):
    """
    Constructs an N-dimensional torus-topology graph with directional metadata.
    """
    num_nodes = L ** dimensions
    edges = []
    edge_dirs = [] # NEW: Tracks the dimension mu
    
    strides = [L**i for i in range(dimensions)]
    
    for i in range(num_nodes):
        for d in range(dimensions): # 'd' is our mu index (0=x, 1=y, 2=z, 3=t)
            coord_d = (i // strides[d]) % L
            
            if coord_d < L - 1:
                neighbor = i + strides[d]
            else:
                neighbor = i - (L - 1) * strides[d]
            
            # Forward edge
            edges.append([i, neighbor])
            edge_dirs.append(d) 
            
            # Backward edge
            edges.append([neighbor, i])
            edge_dirs.append(d) # The backward edge belongs to the same dimension
            
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()
    edge_dirs = torch.tensor(edge_dirs, dtype=torch.long, device=device)
    
    return edge_index, edge_dirs


def get_plaquette_indices(L, dimensions, edge_index, device):
    """
    Generalizes plaquette identification for N-dimensional lattices.
    
    Args:
        L (int): Lattice side length.
        dimensions (int): Number of dimensions (e.g., 4).
        edge_index (Tensor): The [2, E] edge index tensor from get_pbc_edge_index.
    """
    num_nodes = L ** dimensions
    strides = [L**i for i in range(dimensions)]
    
    # 1. Fast lookup: (src, dst) -> edge_index
    # Moving to CPU for dictionary construction (faster than GPU dicts in Python)
    edge_index_cpu = edge_index.cpu()
    edge_map = {(src.item(), dst.item()): idx for idx, (src, dst) in enumerate(edge_index_cpu.T)}
    
    p1_list, p2_list, p3_list, p4_list = [], [], [], []

    # 2. Iterate over every node in the hypercube
    for i in range(num_nodes):
        # 3. For every node, find all unique planes (combinations of 2 dimensions)
        # e.g., for 4D, this gives (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        for d1, d2 in itertools.combinations(range(dimensions), 2):
            
            # Helper to handle PBC wrap-around for a step in dimension 'd'
            def get_neighbor(node_idx, dim_idx):
                coord_d = (node_idx // strides[dim_idx]) % L
                if coord_d < L - 1:
                    return node_idx + strides[dim_idx]
                return node_idx - (L - 1) * strides[dim_idx]

            # Define the 4 corners of the plaquette in the (d1, d2) plane
            n00 = i
            n10 = get_neighbor(n00, d1)
            n01 = get_neighbor(n00, d2)
            n11 = get_neighbor(n10, d2) # Step in d1 then d2

            # 4. Map corners to the edge indices
            # Path: n00 -> n10 -> n11 (back to) n01 (back to) n00
            try:
                p1_list.append(edge_map[(n00, n10)]) # Forward
                p2_list.append(edge_map[(n10, n11)]) # Forward
                p3_list.append(edge_map[(n01, n11)]) # Forward (Loss uses .mH)
                p4_list.append(edge_map[(n00, n01)]) # Forward (Loss uses .mH)
            except KeyError:
                # This would only happen if the edge_index was built incorrectly
                continue

    return (
        torch.tensor(p1_list, device=device),
        torch.tensor(p2_list, device=device),
        torch.tensor(p3_list, device=device),
        torch.tensor(p4_list, device=device)
    )



# def get_pbc_edge_index(L, device):
#     """
#     Constructs a torus-topology graph for an L x L lattice.
#     Ensures periodic boundary conditions to avoid edge effects.
#     """
    
#     edges = []
#     for y in range(L):
#         for x in range(L):
#             src = y * L + x
            
#             # Neighbors with wrap-around (modulo L)
#             dst_right = y * L + ((x + 1) % L)
#             dst_up = ((y + 1) % L) * L + x
            
#             # Add bi-directional edges to allow message passing both ways
#             edges.append([src, dst_right])
#             edges.append([dst_right, src])
#             edges.append([src, dst_up])
#             edges.append([dst_up, src])
            
#     return torch.tensor(edges, dtype=torch.long, device=device).T.contiguous()


# def get_plaquette_indices(L, edge_index, device):
#     """
#     Finds the 4 edge indices that make up every 1x1 plaquette on a PBC grid.
#     """
#     edge_index_cpu = edge_index.cpu()
#     edge_map = {(src.item(), dst.item()): idx for idx, (src, dst) in enumerate(edge_index_cpu.T)}
    
#     p1, p2, p3, p4 = [], [], [], []
#     for y in range(L):
#         for x in range(L):
#             n00 = y * L + x
#             n10 = y * L + ((x + 1) % L)
#             n01 = ((y + 1) % L) * L + x
#             n11 = ((y + 1) % L) * L + ((x + 1) % L)
            
#             p1.append(edge_map[(n00, n10)])
#             p2.append(edge_map[(n10, n11)])
#             p3.append(edge_map[(n11, n01)])
#             p4.append(edge_map[(n01, n00)])
            
#     return (torch.tensor(p1, device=device), torch.tensor(p2, device=device), 
#             torch.tensor(p3, device=device), torch.tensor(p4, device=device))
            

def initialize_data(n_nodes, hidden_dim, device):
    """
        Initializes complex Gaussian noise for matter fields.
        These serve as inputs for the Equivariant Normalizing Flow.
    """
    return {
        'su3': torch.randn(n_nodes, hidden_dim, 3, dtype=torch.complex64, device=device) * 0.2,
        'su2': torch.randn(n_nodes, hidden_dim, 2, dtype=torch.complex64, device=device) * 0.2,
        'u1':  torch.randn(n_nodes, hidden_dim, 1, dtype=torch.complex64, device=device) * 0.2
    }


