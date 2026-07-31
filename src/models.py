# src/models.py

import torch 
import torch.nn as nn
from torch.nn.functional import relu
from torch_geometric.nn import MessagePassing, HeteroConv

from .physics.gauge_groups import SMGateFactory, get_gamma_matrices


# =============================================================================
# 2. CORE EQUIVARIANT LAYERS
# =============================================================================

class ComplexLinear(nn.Module):
    """Linear layer handling complex-valued tensors: W * z."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.w_real = nn.Linear(in_c, out_c, bias=False)
        self.w_imag = nn.Linear(in_c, out_c, bias=False)
        self.b_real = nn.Parameter(torch.zeros(out_c))
        self.b_imag = nn.Parameter(torch.zeros(out_c))

    def forward(self, x):
        res_real = self.w_real(x.real) - self.w_imag(x.imag) + self.b_real
        res_imag = self.w_real(x.imag) + self.w_imag(x.real) + self.b_imag
        return torch.complex(res_real, res_imag)


class ModReLU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, z):
        # z shape: [N, C, G] OR [N, C, S, G]
        norm = torch.norm(z, dim=-1, keepdim=True)
        
        # Broadcast bias across all dims except Channel
        # We assume Channel is always at index 1
        bias_shape = [1] * z.dim()
        bias_shape[1] = -1
        bias = self.b.view(*bias_shape) 
        
        scale = torch.relu(norm + bias) / (norm + 1e-8)
        return scale * z


class ModReLU2(nn.Module):
    """Universal ModReLU for any particle dimension."""
    def __init__(self, channels):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, z):
        # z shape: [nodes, channels, group_dim]
        norm = torch.norm(z, dim=-1, keepdim=True)
        bias = self.b.view(1, -1, 1) # Broadcast across nodes and group_dim
        scale = relu(norm + bias) / (norm + 1e-8)
        return scale * z


class EquivariantScalingFlow(nn.Module):
    def __init__(self, total_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_features, total_features * 2),
            nn.ReLU(),
            nn.Linear(total_features * 2, total_features),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [N, C, G] or [N, C, S, G]
        orig_shape = x.shape
        group_dim = x.size(-1)
        
        # 1. Magnitude squared: [N, C] or [N, C, S]
        mag_sq = torch.norm(x, dim=-1)**2  
        
        # 2. Flatten for MLP: [N, Total_Features]
        mag_sq_flat = mag_sq.view(orig_shape[0], -1)
        
        # 3. Predict scaling
        s_flat = self.net(mag_sq_flat) 
        
        # 4. Restore shape for scaling: [N, C, 1] or [N, C, S, 1]
        s = s_flat.view(*orig_shape[:-1]).unsqueeze(-1)
        
        x_transformed = x * torch.exp(s)
        
        # 5. LJD correction
        log_det = torch.sum(2 * group_dim * s_flat)
        return x_transformed, log_det


class EquivariantScalingFlow2(nn.Module):
    """
    Gauge-equivariant Normalizing Flow layer. 
    Adapts to the group dimension of the specific particle type.
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: [nodes, channels, group_dim]
        group_dim = x.size(-1)
        
        # 1. Gauge-invariant magnitude squared
        mag_sq = torch.norm(x, dim=-1)**2  # [nodes, channels]
        
        # 2. Predict channel-wise scaling factor s
        s = self.net(mag_sq) # [nodes, channels]
        
        # 3. Transform field
        x_transformed = x * torch.exp(s).unsqueeze(-1)
        
        # 4. Correct Complex Log-Jacobian Determinant (LJD)
        # Because we scale a complex vector of length `group_dim`,
        # we are scaling 2 * group_dim real variables.
        log_det = torch.sum(2 * group_dim * s)
        
        return x_transformed, log_det


class GaugeTransportConv(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add', node_dim=0)
        self.lin = ComplexLinear(channels, channels)

    def forward(self, x, edge_index, u_gate):
        # out shape matches x: [N, C, (S), G]
        out = self.propagate(edge_index, x=x, u_gate=u_gate)

        # We swap the Channel dim (index 1) with the Group dim (last index)
        # to allow ComplexLinear to mix flavors/channels
        out = out.transpose(1, -1) 
        out = self.lin(out) 
        out = out.transpose(1, -1)
        return out

    def message(self, x_j, u_gate):
        # u_gate: [E, G, G]
        # x_j: [E, C, (S), G]
        # Multiply last dim of x_j by the gate
        return torch.einsum('enm,e...m->e...n', u_gate, x_j)


class GaugeTransportConv2(MessagePassing):
    """
    Universal Equivariant Message Passing Layer.
    Automatically adapts to the 6D, 3D, 2D, or 1D group dimension of the u_gate.
    """
    def __init__(self, channels):
        super().__init__(aggr='add', node_dim=0)
        self.lin = ComplexLinear(channels, channels)

    def forward(self, x, edge_index, u_gate):
        # out shape matches x: [N, C, (S), G]
        out = self.propagate(edge_index, x=x, u_gate=u_gate)

        # We swap the Channel dim (index 1) with the Group dim (last index)
        # to allow ComplexLinear to mix flavors/channels
        out = out.transpose(1, -1) 
        out = self.lin(out) 
        out = out.transpose(1, -1)
        return out

    def message(self, x_j, u_gate):
        # u_gate: [E, G, G], x_j: [E, C, (S), G]
        # Multiply last dim of x_j by the gate
        return torch.einsum('enm,e...m->e...n', u_gate, x_j)


class WilsonDiracConv(MessagePassing):
    """
    Applies the Wilson-Dirac Operator D(U) to a pseudofermion field.
    """
    def __init__(self, kappa: float, device):
        super().__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.kappa = kappa
        self.gammas = get_gamma_matrices(device)
        self.I4 = torch.eye(4, dtype=torch.complex64, device=device)

    def forward(self, x, edge_index, edge_dirs, u_gate):
        # x is already [Nodes, 4, 4, Color]
        # get_pbc_edge_index interleaves links: even indices are +mu, odd are -mu
        is_fwd = torch.arange(edge_index.size(1), device=edge_index.device) % 2 == 0
        hopping_term = self.propagate(edge_index, x=x, edge_dirs=edge_dirs, u_gate=u_gate, is_fwd=is_fwd)
        return x - self.kappa * hopping_term

    def message(self, x_j, edge_dirs, u_gate, is_fwd):
        """
        x_j: [Edges, Channels, Spin(4), Color]
        u_gate: [Edges, Color, Color]
        is_fwd: [Edges] bool, True on +mu links
        """
        out_messages = torch.zeros_like(x_j)

        for mu in range(4):
            mask = (edge_dirs == mu)
            if not mask.any(): continue

            # Wilson projector: (I - gamma_mu) on +mu links, (I + gamma_mu) on -mu links
            sign = torch.where(is_fwd[mask], -1.0, 1.0).view(-1, 1, 1).to(self.I4.dtype)
            spin_proj = self.I4 + sign * self.gammas[mu]  # [E_masked, 4, 4]

            # 1. Apply Spin Projection to the 'Spin' dimension (index 2)
            # x_j[mask] is [E_masked, C, 4, Col]
            # 'eab' is spin_proj, 'ecbf' is x_j (edge, channel, spin, field/color)
            x_spin = torch.einsum('eab, ecbf -> ecaf', spin_proj, x_j[mask])

            # 2. Apply Gauge Transport to the 'Color' dimension (index 3)
            # 'ij' is u_gate, 'ecsj' is x_spin
            msg = torch.einsum('eij, ecsj -> ecsi', u_gate[mask], x_spin)

            out_messages[mask] = msg

        return out_messages

    # def message(self, x_j, edge_dirs, u_gate):
    #     """
    #     x_j: [Edges, Features, Spin(4), Color]
    #     u_gate: [Edges, Color, Color]
    #     """
    #     # We will collect messages for each direction and combine them
    #     # to avoid the memory-intensive zeros_like(x_j)
    #     out_messages = torch.zeros_like(x_j)

    #     for mu in range(4):
    #         mask = (edge_dirs == mu)
    #         if not mask.any(): continue

    #         # Get the spin projection (1 - gamma_mu)
    #         # Shape: [4, 4]
    #         spin_proj = self.I4 - self.gammas[mu]

    #         # A. Apply Spin Projection: (1 - gamma_mu) * psi
    #         # 'ab' is spin_proj [4,4], 'efbc' is x_j [Edges, Feat, Spin, Color]
    #         # We want to multiply spin_proj by the 'b' (Spin) dimension
    #         x_masked = x_j[mask]
    #         x_spin = torch.einsum('ab, efbc -> efac', spin_proj, x_masked)

    #         # B. Apply Gauge Transport: U * (Spin_Projected_Psi)
    #         # 'enc' is u_gate [Edges, Color, Color], 'efsc' is x_spin
    #         # We multiply the 'c' (Color) dimensions
    #         msg = torch.einsum('enc, efsc -> efsn', u_gate[mask], x_spin)
            
    #         out_messages[mask] = msg

    #     return out_messages
        
    # def forward(self, x, edge_index, edge_dirs, u_gate):
    #     """
    #     x: Pseudofermion field [nodes, channels, 4 (spin), N_c (color)]
    #     edge_dirs: Tensor of shape [edges] indicating the direction mu (0,1,2,3)
    #     """
    #     out = x 
    #     hopping_term = self.propagate(edge_index, x=x, edge_dirs=edge_dirs, u_gate=u_gate)
    #     return out - self.kappa * hopping_term

    # def message(self, x_j, edge_dirs, u_gate):
    #     messages = torch.zeros_like(x_j)
    #     for mu in range(4):
    #         mask = (edge_dirs == mu)
    #         if not mask.any(): continue
            
    #         spin_proj = self.I4 - self.gammas[mu]
    #         x_j_spin = torch.einsum('ab, ...bc -> ...ac', spin_proj, x_j[mask])
    #         msg = torch.einsum('enc, ...sc -> ...sn', u_gate[mask], x_j_spin)
    #         messages[mask] = msg
    #     return messages


class SM_HeteroConv(nn.Module):
    """
    Routes distinct particle representations through physical gauge interactions.
    Fermions use the Wilson-Dirac operator; Bosons use standard Parallel Transport.
    """
    def __init__(self, hidden_dim, kappa, device):
        super().__init__()
        
        # 1. Map edge types to their specific physics layers
        self.conv = HeteroConv({
            # FERMIONS: Require spinor projections and the Dirac Operator
            ('quark_left', 'transport', 'quark_left'): WilsonDiracConv(kappa, device),
            ('quark_up_right', 'transport', 'quark_up_right'): WilsonDiracConv(kappa, device),
            ('quark_down_right', 'transport', 'quark_down_right'): WilsonDiracConv(kappa, device),
            ('lepton_left', 'transport', 'lepton_left'): WilsonDiracConv(kappa, device),
            ('lepton_right', 'transport', 'lepton_right'): WilsonDiracConv(kappa, device),
            
            # BOSONS: Require only simple parallel transport
            ('higgs', 'transport', 'higgs'): GaugeTransportConv(hidden_dim),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict, edge_dirs_dict, u_dict):
        """
        x_dict: The particle fields (x for pseudofermions, phi for Higgs)
        edge_index_dict: Lattice connections per species
        edge_dirs_dict: Spacetime directions (mu) for each edge
        u_dict: SU(3), SU(2), U(1) gauge links
        """
        # Build the physical gates dynamically based on current links
        factory = SMGateFactory(u_dict)
        u_gate_dict = factory.get_all_gates()
        
        # PyG HeteroConv will automatically pass 'u_gate' and 'edge_dirs' 
        # to the specific layers that have those arguments in their signature.
        return self.conv(
            x_dict, 
            edge_index_dict, 
            edge_dirs=edge_dirs_dict,
            u_gate=u_gate_dict, 
        )


GENERATIONS = [1, 2, 3]

base_species = [
    'quark_left', 'quark_up_right', 'quark_down_right', 
    'lepton_left', 'lepton_right', 'lepton_nu_right'
]

ALL_SPECIES = [f"{s}_g{g}" for s in base_species for g in GENERATIONS] + ['higgs']


class SM_HeteroGNN(nn.Module):
    """
    The full Standard Model Graph Neural Network.
    """
    def __init__(self, hidden_dim, kappa_dict, device):
        super().__init__()
        self.fermions = [k for k in ALL_SPECIES if k != 'higgs']
        self.bosons = ['higgs']
        
        self.flows = nn.ModuleDict({k: EquivariantScalingFlow(hidden_dim) for k in ALL_SPECIES})
        self.acts = nn.ModuleDict({k: ModReLU(hidden_dim) for k in ALL_SPECIES})
        
        # FIX: Use a standard ModuleDict instead of HeteroConv
        self.physics_layers = nn.ModuleDict()
        
        for f in self.fermions:
            kappa = kappa_dict.get(f, 0.1)
            self.physics_layers[f] = WilsonDiracConv(kappa, device)
            
        for b in self.bosons:
            self.physics_layers[b] = GaugeTransportConv(hidden_dim)

    def forward(self, z_dict, edge_index_dict, edge_dirs_dict, u_dict):
        phi_dict = {}
        total_ljd = 0.0
        
        # 1. Flow (Noise -> Field)
        for species in ALL_SPECIES:
            if species in z_dict:
                phi, ljd = self.flows[species](z_dict[species])
                phi_dict[species] = phi
                total_ljd += ljd
                
        factory = SMGateFactory(u_dict)
        u_gate_dict = factory.get_all_gates()
        
        out_dict = {}
        
        # 2. Physics Message Passing (Explicit & Safe)
        for species in phi_dict.keys():
            layer = self.physics_layers[species]
            edge_type = (species, 'transport', species)
            
            # Extract precise data for this species
            x = phi_dict[species]
            edge_index = edge_index_dict[edge_type]
            u_gate = u_gate_dict[edge_type]
            
            # Route carefully depending on particle spin (Fermion vs Boson)
            if species in self.fermions:
                edge_dirs = edge_dirs_dict[edge_type]
                out_dict[species] = layer(x=x, edge_index=edge_index, edge_dirs=edge_dirs, u_gate=u_gate)
            else:
                out_dict[species] = layer(x=x, edge_index=edge_index, u_gate=u_gate)
        
        # 3. Activations
        final_dict = {k: self.acts[k](v) for k, v in out_dict.items()}
        return final_dict, total_ljd



