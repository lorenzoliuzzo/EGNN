import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from src.dirac import WilsonDiracConv
from src.groups import SMGateFactory

GENERATIONS = [1, 2, 3]

base_species = [
    'quark_left', 'quark_up_right', 'quark_down_right',
    'lepton_left', 'lepton_right', 'lepton_nu_right',
]

ALL_SPECIES = [f"{s}_g{g}" for s in base_species for g in GENERATIONS] + ['higgs']


class ComplexLinear(nn.Module):
    """W z + b for complex z, stored as two real linear maps (W = A + iB)."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.w_real = nn.Linear(in_c, out_c, bias=False)
        self.w_imag = nn.Linear(in_c, out_c, bias=False)
        self.b_real = nn.Parameter(torch.zeros(out_c))
        self.b_imag = nn.Parameter(torch.zeros(out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_real = self.w_real(x.real) - self.w_imag(x.imag) + self.b_real
        res_imag = self.w_real(x.imag) + self.w_imag(x.real) + self.b_imag
        return torch.complex(res_real, res_imag)


class ModReLU(nn.Module):
    """Gauge-invariant activation: rescales |z| along the group dimension."""

    def __init__(self, channels: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, C, G] or [N, C, S, G]; channel is always index 1
        norm = torch.norm(z, dim=-1, keepdim=True)
        bias_shape = [1] * z.dim()
        bias_shape[1] = -1
        bias = self.b.view(*bias_shape)
        scale = torch.relu(norm + bias) / (norm + 1e-8)
        return scale * z


class EquivariantScalingFlow(nn.Module):
    """Normalizing-flow layer scaling each channel by exp(s(|z|^2))."""

    def __init__(self, total_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_features, total_features * 2),
            nn.ReLU(),
            nn.Linear(total_features * 2, total_features),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N, C, G] or [N, C, S, G]
        orig_shape = x.shape
        group_dim = x.size(-1)

        mag_sq = torch.norm(x, dim=-1) ** 2
        mag_sq_flat = mag_sq.view(orig_shape[0], -1)
        s_flat = self.net(mag_sq_flat)
        s = s_flat.view(*orig_shape[:-1]).unsqueeze(-1)

        x_transformed = x * torch.exp(s)
        # scaling a complex G-vector scales 2G real degrees of freedom
        log_det = torch.sum(2 * group_dim * s_flat)
        return x_transformed, log_det


class GaugeTransportConv(MessagePassing):
    """Parallel transport U_e psi_src followed by channel mixing."""

    def __init__(self, channels: int):
        super().__init__(aggr='add', node_dim=0)
        self.lin = ComplexLinear(channels, channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, u_gate: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, u_gate=u_gate)
        # swap channel (index 1) with group (last) so the linear map only
        # mixes channels and equivariance in the group index is preserved
        out = out.transpose(1, -1)
        out = self.lin(out)
        return out.transpose(1, -1)

    def message(self, x_j: torch.Tensor, u_gate: torch.Tensor) -> torch.Tensor:
        return torch.einsum('enm,e...m->e...n', u_gate, x_j)


class SM_HeteroGNN(nn.Module):
    """Flow + physics transport for every Standard Model species."""

    def __init__(self, hidden_dim: int, kappa_dict: dict, device: torch.device | str):
        super().__init__()
        self.fermions = [k for k in ALL_SPECIES if k != 'higgs']
        self.bosons = ['higgs']

        self.flows = nn.ModuleDict({k: EquivariantScalingFlow(hidden_dim) for k in ALL_SPECIES})
        self.acts = nn.ModuleDict({k: ModReLU(hidden_dim) for k in ALL_SPECIES})

        self.physics_layers = nn.ModuleDict()
        for f in self.fermions:
            self.physics_layers[f] = WilsonDiracConv(kappa_dict.get(f, 0.1), device)
        for b in self.bosons:
            self.physics_layers[b] = GaugeTransportConv(hidden_dim)

    def forward(
        self,
        z_dict: dict,
        edge_index_dict: dict,
        edge_dirs_dict: dict,
        u_dict: dict,
    ) -> tuple[dict, torch.Tensor]:
        phi_dict = {}
        total_ljd = 0.0

        for species in ALL_SPECIES:
            if species in z_dict:
                phi, ljd = self.flows[species](z_dict[species])
                phi_dict[species] = phi
                total_ljd += ljd

        u_gate_dict = SMGateFactory(u_dict).get_all_gates()

        out_dict = {}
        for species in phi_dict.keys():
            layer = self.physics_layers[species]
            edge_type = (species, 'transport', species)
            x = phi_dict[species]
            edge_index = edge_index_dict[edge_type]
            u_gate = u_gate_dict[edge_type]

            if species in self.fermions:
                out_dict[species] = layer(x=x, edge_index=edge_index,
                                          edge_dirs=edge_dirs_dict[edge_type],
                                          u_gate=u_gate)
            else:
                out_dict[species] = layer(x=x, edge_index=edge_index, u_gate=u_gate)

        final_dict = {k: self.acts[k](v) for k, v in out_dict.items()}
        return final_dict, total_ljd
