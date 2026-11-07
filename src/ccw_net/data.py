"""Loading a domain's arrays and its stratified train/val/test split."""
import numpy as np
from sklearn.model_selection import train_test_split

from ccw_net.domain import get_domain


def _load_labels(data_dir, domain):
    """(leadlag, climbdive, labelled-row mask, joint label used for stratification) of a domain."""
    if domain.num_axes == 2:
        leadlag = np.load(data_dir / domain.label_files[0])
        climbdive = np.load(data_dir / domain.label_files[1])
        return leadlag, climbdive, (leadlag >= 0) & (climbdive >= 0), leadlag * domain.axis_sizes[1] + climbdive
    if domain.num_axes == 1:
        concept = np.load(data_dir / domain.label_files[0])
        return concept.copy(), np.zeros_like(concept), concept >= 0, concept
    attributes = np.load(data_dir / domain.label_files[0])
    if attributes.ndim != 2 or attributes.shape[1] != domain.num_axes:
        raise ValueError(f"Domain {domain.name} expects [N, {domain.num_axes}] concept labels; got {attributes.shape}")
    width = min(16, domain.num_axes)
    joint = (attributes[:, :width] * (2 ** np.arange(width)[None, :])).sum(axis=1)
    return (attributes.astype(np.int64), np.zeros(attributes.shape[0], dtype=np.int64),
            np.ones(attributes.shape[0], dtype=bool), joint)


def load_data(data_dir, cfg):
    """Latent inputs, actions or classes, concept labels, flow covariates and the split of one config at cfg.seed."""
    domain = get_domain(cfg.dataset.domain)
    latent = np.load(data_dir / "latent.npy")
    action = np.load(data_dir / ("class.npy" if domain.num_axes >= 3 else "action.npy"))
    leadlag, climbdive, label_mask, labels_joint = _load_labels(data_dir, domain)

    data_cfg = cfg.get('data') or {}
    obs_path = data_dir / "observations.npy"
    if data_cfg.get('covariates', 'observations') == 'latents' or not obs_path.exists():
        covariates = latent
    else:
        covariates = np.load(obs_path)

    labeled_idx = np.where(label_mask)[0]
    unlabeled_idx = np.where(~label_mask)[0]
    labels_for_strat = labels_joint[labeled_idx]
    trainval_idx, test_idx = train_test_split(
        labeled_idx, test_size=0.15, stratify=labels_for_strat, random_state=cfg.seed)
    val_stratify = data_cfg.get('val_stratify', 'legacy')
    if val_stratify == 'aligned':
        trainval_labels = labels_joint[trainval_idx]
    elif val_stratify == 'legacy':
        trainval_labels = labels_for_strat[np.isin(labeled_idx, trainval_idx)]
    else:
        raise ValueError(f"data.val_stratify={val_stratify!r}; use 'legacy' or 'aligned'")
    train_labeled_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.15, stratify=trainval_labels, random_state=cfg.seed)

    return {
        'input': latent,
        'action': action,
        'leadlag': leadlag,
        'climbdive': climbdive,
        'covariates': covariates,
        'train_idx': np.concatenate([train_labeled_idx, unlabeled_idx]),
        'train_labeled_idx': train_labeled_idx,
        'unlabeled_idx': unlabeled_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'domain': domain,
    }
