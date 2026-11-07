"""Scalar DCI of the test concepts against the ground-truth factors."""
from typing import Dict

import numpy as np

from ccw_net.disentanglement_lib_compat import compute_dci


def ground_truth_factors(leadlag: np.ndarray, climbdive: np.ndarray) -> np.ndarray:
    """CUB: the 112 attributes; aircraft: the two binary axes; CarRacing: one-hot direction."""
    if leadlag.ndim == 2:
        return leadlag.astype(np.int64)
    if len(np.unique(climbdive)) >= 2:
        return np.stack([leadlag.astype(np.int64), climbdive.astype(np.int64)], axis=1)
    n_cls = int(leadlag.max()) + 1
    factors = np.zeros((leadlag.shape[0], n_cls), dtype=np.int64)
    for c in range(n_cls):
        factors[:, c] = (leadlag == c).astype(np.int64)
    return factors


def dci(concepts: np.ndarray, leadlag: np.ndarray, climbdive: np.ndarray) -> Dict[str, float]:
    """DCI of the flattened concept slots; factors constant in the DCI training split are skipped."""
    return compute_dci(concepts.reshape(len(concepts), -1), ground_truth_factors(leadlag, climbdive))
