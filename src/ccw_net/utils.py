import torch


def squash(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Capsule squash: rescale each vector to length |v|^2 / (1 + |v|^2)."""
    r2 = torch.sum(v * v, dim=-1, keepdim=True)
    r = torch.sqrt(r2 + eps)
    scale = r2 / (1. + r2) / r
    return scale * v
