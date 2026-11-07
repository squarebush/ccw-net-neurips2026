"""Zero-ablation attribution of the CUB-200 models."""
from typing import Callable, Dict

import numpy as np
import torch

ABSENT_LOGIT = float('-inf')
CHUNK_BYTES = 4 * 1024 * 1024 * 1024


def ablation_shares(concepts: np.ndarray, forward: Callable, reference: float) -> np.ndarray:
    """Per-sample share of each slot in the summed norm of Y(M) - Y(M with slot j set to `reference`)."""
    N, J, D = concepts.shape
    M = torch.from_numpy(concepts).float()
    with torch.no_grad():
        Y_full = forward(M).numpy()
    A = Y_full.shape[1]

    IE_attr = np.zeros((N, J, A), dtype=np.float64)
    slots_per_chunk = min(max(1, CHUNK_BYTES // max(N * J * D * 4, 1)), J)
    for s_start in range(0, J, slots_per_chunk):
        s_end = min(s_start + slots_per_chunk, J)
        k = s_end - s_start
        M_chunk = M.unsqueeze(0).expand(k, N, J, D).contiguous()
        for offset, s in enumerate(range(s_start, s_end)):
            M_chunk[offset, :, s, :] = reference
        M_chunk = M_chunk.view(k * N, J, D)
        with torch.no_grad():
            Y_ablated = forward(M_chunk).view(k, N, A).numpy()
        for offset, s in enumerate(range(s_start, s_end)):
            IE_attr[:, s, :] = Y_full - Y_ablated[offset]
        del M_chunk

    slot_norms = np.linalg.norm(IE_attr, axis=-1)
    return slot_norms / (slot_norms.sum(axis=-1, keepdims=True) + 1e-12)


def attribution(model, concepts: np.ndarray, attributes: np.ndarray, arch: str) -> Dict[str, float]:
    """Mean share of the present attributes' slots (All Active) and of the absent ones (All Inactive)."""
    reference = ABSENT_LOGIT if arch == 'cbm' else 0.0
    shares = ablation_shares(concepts, model.forward_from_concepts, reference)
    pm_active = np.array([shares[i, attributes[i] > 0].sum() for i in range(len(shares))])
    pm_forbidden = 1.0 - pm_active
    return {'pm_active': float(pm_active.mean()), 'pm_forbidden': float(pm_forbidden.mean())}
