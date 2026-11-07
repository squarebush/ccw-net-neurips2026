"""Concept-label ablations: label fraction, label noise, merges, removal and a placebo concept."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

CONCEPT_INDEX_MAP = {"lead": 0, "lag": 1, "climb": 2, "dive": 3, "straight": 0, "left": 1, "right": 2}
CARRACING_MERGES = {"straight_left": (1, 0), "straight_right": (2, 0), "left_right": (2, 1)}
MIN_LABELS_PER_CLASS = 10


@dataclass
class AblationConfig:
    """The stress_test settings of one configuration."""

    remove_concept: Optional[str] = None
    merge_concept: Optional[str] = None
    label_noise: float = 0.0
    add_placebo: bool = False
    label_fraction: float = 1.0
    base_num_concepts: int = 4
    seed: int = 0


def ablation_config(cfg, base_num_concepts):
    """AblationConfig of a configuration whose domain has base_num_concepts concepts."""
    stress = cfg.stress_test
    return AblationConfig(
        remove_concept=stress.remove_concept, merge_concept=stress.merge_concept, label_noise=stress.label_noise,
        add_placebo=stress.add_placebo, label_fraction=stress.label_fraction, seed=cfg.seed,
        base_num_concepts=base_num_concepts)


def generate_placebo_labels(n_samples, seed):
    """Random binary labels of the placebo concept."""
    return np.random.RandomState(seed + 3000).randint(0, 2, size=n_samples)


def subsample_labels(leadlag, climbdive, train_labeled_idx, config):
    """Keep label_fraction of the labelled training rows per class (at least 10); the others lose their labels."""
    if config.label_fraction >= 1.0:
        return train_labeled_idx
    rng = np.random.RandomState(config.seed + 2000)
    ll, cd = leadlag[train_labeled_idx], climbdive[train_labeled_idx]
    cd_values = np.unique(cd)
    if len(cd_values) == 1 and cd_values[0] == 0:
        classes = [ll == value for value in np.unique(ll) if value >= 0]
    else:
        classes = [(ll == ll_value) & (cd == cd_value) for ll_value in (0, 1) for cd_value in (0, 1)]
    kept = []
    for in_class in classes:
        members = train_labeled_idx[in_class]
        if len(members) > 0:
            n_keep = min(max(int(len(members) * config.label_fraction), MIN_LABELS_PER_CLASS), len(members))
            kept.extend(rng.choice(members, n_keep, replace=False))
    kept_idx = np.array(sorted(kept))
    dropped = np.setdiff1d(train_labeled_idx, kept_idx)
    leadlag[dropped] = -1
    climbdive[dropped] = -1
    return kept_idx


def stress_concepts(leadlag, climbdive, train_labeled_idx, unlabeled_idx, config):
    """Placebo labels, merged classes and noisy labels; returns the labels, placebo labels and number of slots."""
    placebo = None
    num_concepts = config.base_num_concepts
    if config.add_placebo:
        placebo = generate_placebo_labels(len(leadlag), config.seed)
        num_concepts += 1

    if config.merge_concept == "leadlag":
        leadlag[train_labeled_idx[leadlag[train_labeled_idx] >= 0]] = 0
    elif config.merge_concept == "climbdive":
        climbdive[train_labeled_idx[climbdive[train_labeled_idx] >= 0]] = 0
    elif config.merge_concept in CARRACING_MERGES:
        source, target = CARRACING_MERGES[config.merge_concept]
        leadlag[train_labeled_idx[leadlag[train_labeled_idx] == source]] = target

    if config.label_noise > 0:
        rng = np.random.RandomState(config.seed + 1000)
        num_classes = int(max(2, leadlag[leadlag >= 0].max() + 1)) if (leadlag >= 0).any() else 2
        for idx in train_labeled_idx:
            if leadlag[idx] >= 0 and rng.random() < config.label_noise:
                if num_classes == 2:
                    leadlag[idx] = 1 - leadlag[idx]
                else:
                    leadlag[idx] = (leadlag[idx] + rng.randint(1, num_classes)) % num_classes
            if climbdive[idx] >= 0 and rng.random() < config.label_noise:
                climbdive[idx] = 1 - climbdive[idx]

    if placebo is not None:
        placebo[unlabeled_idx] = -1
    return leadlag, climbdive, placebo, num_concepts


def apply_ablations(leadlag, climbdive, train_labeled_idx, unlabeled_idx, config, action=None):
    """(leadlag, climbdive, labelled train rows, placebo labels, number of slots, {}) after every stress test."""
    train_labeled_idx = subsample_labels(leadlag, climbdive, train_labeled_idx, config)
    leadlag, climbdive, placebo, num_concepts = stress_concepts(
        leadlag, climbdive, train_labeled_idx, unlabeled_idx, config)
    return leadlag, climbdive, train_labeled_idx, placebo, num_concepts, {}


def get_removed_concept_idx(config):
    """Slot index of the removed concept, or None."""
    if config.remove_concept is None:
        return None
    return CONCEPT_INDEX_MAP.get(config.remove_concept)


def zero_removed_concept(v_concepts, config):
    """The concepts [B, J, D] with the removed slot multiplied by zero."""
    removed_idx = get_removed_concept_idx(config)
    if removed_idx is None:
        return v_concepts
    mask = torch.ones(v_concepts.shape[1], device=v_concepts.device, dtype=v_concepts.dtype)
    mask[removed_idx] = 0.0
    return v_concepts * mask.view(1, -1, 1)
