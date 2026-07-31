import torch
from helpers import gauge_transform_links, random_su

from src.lattice import get_pbc_edge_index
from src.models import EquivariantScalingFlow, GaugeTransportConv, ModReLU


def test_gauge_transport_conv_is_equivariant() -> None:
    torch.manual_seed(0)
    ei, _ = get_pbc_edge_index(4, 2, 'cpu')
    n, E, C, G = 16, ei.size(1), 3, 2

    conv = GaugeTransportConv(C)
    psi = torch.randn(n, C, G, dtype=torch.complex64)
    h = torch.randn(E, G, G, dtype=torch.complex64)
    u = torch.matrix_exp(1j * (h + h.mH))

    out = conv(psi, ei, u)

    g = random_su(G, n, seed=4)
    psi_t = torch.einsum('nij,ncj->nci', g, psi)
    u_t = gauge_transform_links(u, g, ei)
    out_t = conv(psi_t, ei, u_t)

    # transforming the inputs must equal transforming the output: the channel
    # mixing acts only on channels, so equivariance in the group index survives
    expected = torch.einsum('nij,ncj->nci', g, out)
    assert torch.allclose(out_t, expected, atol=1e-4)


def test_modrelu_is_gauge_invariant_in_magnitude() -> None:
    torch.manual_seed(0)
    n, C, G = 10, 3, 2
    act = ModReLU(C)
    z = torch.randn(n, C, G, dtype=torch.complex64)
    g = random_su(G, n, seed=1)
    z_t = torch.einsum('nij,ncj->nci', g, z)
    assert torch.allclose(torch.norm(act(z), dim=-1),
                          torch.norm(act(z_t), dim=-1), atol=1e-5)


def test_scaling_flow_logdet_matches_scaling() -> None:
    torch.manual_seed(0)
    n, C, G = 6, 4, 3
    flow = EquivariantScalingFlow(C)
    x = torch.randn(n, C, G, dtype=torch.complex64)
    x_out, log_det = flow(x)

    # each channel is scaled by exp(s): recover s from the norms and check the
    # complex log-Jacobian 2G * sum(s)
    s = torch.log(torch.norm(x_out, dim=-1) / torch.norm(x, dim=-1))
    assert torch.allclose(log_det, torch.sum(2 * G * s), rtol=1e-3)
