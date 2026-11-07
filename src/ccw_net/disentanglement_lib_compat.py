"""Scalar DCI after the definitions of google-research/disentanglement_lib (Apache-2.0)."""
from typing import Dict

import numpy as np
import sklearn.ensemble
from scipy.stats import entropy as scipy_entropy


def _train_test_split(Z: np.ndarray, factors: np.ndarray, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.RandomState(seed)
    N = Z.shape[0]
    perm = rng.permutation(N)
    n_test = int(N * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return Z[train_idx], factors[train_idx], Z[test_idx], factors[test_idx]


def compute_dci(Z: np.ndarray, factors: np.ndarray, test_frac: float = 0.2, n_estimators: int = 10,
                seed: int = 0) -> Dict[str, float]:
    """Disentanglement, completeness and test-split informativeness of gradient-boosted factor predictors."""
    Z_tr, F_tr, Z_te, F_te = _train_test_split(Z, factors, test_frac, seed)
    D_total = Z_tr.shape[1]
    K = factors.shape[1]

    importance_full = np.zeros((D_total, K))
    test_acc_full = np.zeros(K)
    valid_factors = []
    for k in range(K):
        if len(np.unique(F_tr[:, k])) < 2:
            continue
        valid_factors.append(k)
        clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(Z_tr, F_tr[:, k])
        importance_full[:, k] = np.abs(clf.feature_importances_)
        test_acc_full[k] = (clf.predict(Z_te) == F_te[:, k]).mean()

    K_eff = len(valid_factors)
    importance = importance_full[:, valid_factors]
    test_acc = test_acc_full[valid_factors]

    eps = 1e-11
    R = importance + eps

    row_sums = R.sum(axis=1, keepdims=True)
    P_d = R / row_sums
    if K_eff < 2:
        disent_per_code = np.full(D_total, np.nan)
    else:
        H_d = scipy_entropy(P_d.T, base=K_eff)
        disent_per_code = 1.0 - H_d
    code_weights = importance.sum(axis=1)
    if code_weights.sum() > 0:
        code_weights = code_weights / code_weights.sum()
    else:
        code_weights = np.ones(D_total) / D_total
    if np.any(np.isnan(disent_per_code)):
        disentanglement = float('nan')
    else:
        disentanglement = float(np.sum(disent_per_code * code_weights))

    col_sums = R.sum(axis=0, keepdims=True)
    Q_k = R / col_sums
    if D_total < 2:
        complete_per_factor = np.full(K_eff if K_eff > 0 else 1, np.nan)
    else:
        H_k = scipy_entropy(Q_k, base=D_total)
        complete_per_factor = 1.0 - H_k
    factor_weights = importance.sum(axis=0)
    if factor_weights.sum() > 0:
        factor_weights = factor_weights / factor_weights.sum()
    else:
        factor_weights = np.ones(K_eff if K_eff > 0 else 1) / max(K_eff, 1)
    if np.any(np.isnan(complete_per_factor)):
        completeness = float('nan')
    else:
        completeness = float(np.sum(complete_per_factor * factor_weights))

    informativeness = float(test_acc.mean()) if len(test_acc) > 0 else float('nan')
    return {'disentanglement': disentanglement, 'completeness': completeness, 'informativeness': informativeness}
