import torch


def random_su(n: int, num: int, seed: int) -> torch.Tensor:
    """Random SU(n) matrices via the exponential map, det normalized to 1."""
    torch.manual_seed(seed)
    h = torch.randn(num, n, n, dtype=torch.complex64)
    u = torch.matrix_exp(1j * (h + h.mH))
    return u / torch.linalg.det(u).view(-1, 1, 1) ** (1 / n)


def is_unitary(u: torch.Tensor, atol: float = 1e-4) -> bool:
    eye = torch.eye(u.size(-1), dtype=u.dtype).expand_as(u)
    return torch.allclose(u @ u.mH, eye, atol=atol)


def gauge_transform_links(u: torch.Tensor, g: torch.Tensor,
                          edge_index: torch.Tensor) -> torch.Tensor:
    """U'_e = G_dst U_e G_src^dag — the transform matching message passing
    (messages are U_e psi_src delivered to dst)."""
    return torch.einsum('eij,ejk,ekl->eil', g[edge_index[1]], u, g[edge_index[0]].mH)
