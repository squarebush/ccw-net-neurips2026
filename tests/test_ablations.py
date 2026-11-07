import numpy as np
import pytest
import torch

from ccw_net.ablations import (
    AblationConfig,
    apply_ablations,
    generate_placebo_labels,
    get_removed_concept_idx,
    stress_concepts,
    subsample_labels,
    zero_removed_concept,
)

N = 20000
TRAIN = np.arange(15000)
UNLABELED = np.arange(15000, N)


def labels(num_classes=2, seed=0):
    rng = np.random.RandomState(seed)
    leadlag = rng.randint(0, num_classes, N)
    climbdive = rng.randint(0, 2, N) if num_classes == 2 else np.zeros(N, dtype=int)
    leadlag[:200] = -1
    return leadlag, climbdive


def stress(leadlag, climbdive, **settings):
    return stress_concepts(leadlag.copy(), climbdive.copy(), TRAIN, UNLABELED, AblationConfig(**settings))


def test_binary_noise_rate_and_determinism():
    leadlag, climbdive = labels()
    ll, cd, placebo, num_concepts = stress(leadlag, climbdive, label_noise=0.2, seed=5)
    labelled = TRAIN[leadlag[TRAIN] >= 0]
    assert abs((ll[labelled] != leadlag[labelled]).mean() - 0.2) < 0.015
    assert abs((cd[TRAIN] != climbdive[TRAIN]).mean() - 0.2) < 0.015
    assert np.array_equal(ll[labelled] != leadlag[labelled], ll[labelled] == 1 - leadlag[labelled])
    assert np.array_equal(ll[:200], leadlag[:200])
    assert np.array_equal(ll[UNLABELED], leadlag[UNLABELED]) and np.array_equal(cd[UNLABELED], climbdive[UNLABELED])
    assert placebo is None and num_concepts == 4
    again, _, _, _ = stress(leadlag, climbdive, label_noise=0.2, seed=5)
    other, _, _, _ = stress(leadlag, climbdive, label_noise=0.2, seed=6)
    assert np.array_equal(ll, again)
    assert not np.array_equal(ll, other)


def test_three_class_noise_moves_to_another_class():
    leadlag, climbdive = labels(num_classes=3)
    ll, _, _, _ = stress(leadlag, climbdive, label_noise=0.3, base_num_concepts=3)
    labelled = TRAIN[leadlag[TRAIN] >= 0]
    changed = labelled[ll[labelled] != leadlag[labelled]]
    assert abs(len(changed) / len(labelled) - 0.3) < 0.015
    for value in range(3):
        assert set(ll[changed[leadlag[changed] == value]]) == {0, 1, 2} - {value}


def class_rows(sizes):
    leadlag = np.concatenate([np.full(size, ll) for (ll, _), size in sizes.items()])
    climbdive = np.concatenate([np.full(size, cd) for (_, cd), size in sizes.items()])
    return np.concatenate([leadlag, [0, 1]]), np.concatenate([climbdive, [0, 1]]), np.arange(len(leadlag))


@pytest.mark.parametrize("sizes, fraction, kept", [
    ({(0, 0): 400, (0, 1): 400, (1, 0): 400, (1, 1): 20}, 0.25, {(0, 0): 100, (0, 1): 100, (1, 0): 100, (1, 1): 10}),
    ({(0, 0): 200, (1, 0): 200, (2, 0): 8}, 0.5, {(0, 0): 100, (1, 0): 100, (2, 0): 8}),
])
def test_label_fraction_per_class(sizes, fraction, kept):
    leadlag, climbdive, train = class_rows(sizes)
    ll, cd = leadlag.copy(), climbdive.copy()
    kept_idx = subsample_labels(ll, cd, train, AblationConfig(label_fraction=fraction, seed=1))
    assert np.array_equal(kept_idx, np.unique(kept_idx)) and np.isin(kept_idx, train).all()
    assert {key: int(((leadlag[kept_idx] == key[0]) & (climbdive[kept_idx] == key[1])).sum()) for key in sizes} == kept
    dropped = np.setdiff1d(train, kept_idx)
    assert (ll[dropped] == -1).all() and (cd[dropped] == -1).all()
    assert np.array_equal(ll[kept_idx], leadlag[kept_idx]) and np.array_equal(ll[-2:], leadlag[-2:])
    again = subsample_labels(leadlag.copy(), climbdive.copy(), train, AblationConfig(label_fraction=fraction, seed=1))
    assert np.array_equal(kept_idx, again)


def test_full_label_fraction_is_identity():
    leadlag, climbdive = labels()
    ll, cd, train, placebo, num_concepts, extra = apply_ablations(
        leadlag.copy(), climbdive.copy(), TRAIN, UNLABELED, AblationConfig())
    assert train is TRAIN and placebo is None and num_concepts == 4 and extra == {}
    assert np.array_equal(ll, leadlag) and np.array_equal(cd, climbdive)


@pytest.mark.parametrize("merge, axis", [("leadlag", 0), ("climbdive", 1)])
def test_aircraft_merge_sets_axis_to_zero(merge, axis):
    original = labels()
    merged = stress(*original, merge_concept=merge)
    labelled = TRAIN[original[axis][TRAIN] >= 0]
    assert (merged[axis][labelled] == 0).all()
    assert np.array_equal(merged[axis][UNLABELED], original[axis][UNLABELED])
    assert np.array_equal(merged[1 - axis], original[1 - axis])


@pytest.mark.parametrize("merge, source, target", [("straight_left", 1, 0), ("straight_right", 2, 0), ("left_right", 2, 1)])
def test_carracing_merge_relabels_source(merge, source, target):
    leadlag, climbdive = labels(num_classes=3)
    ll, _, _, _ = stress(leadlag, climbdive, merge_concept=merge, base_num_concepts=3)
    expected = leadlag.copy()
    expected[TRAIN[leadlag[TRAIN] == source]] = target
    assert np.array_equal(ll, expected)


def test_placebo_labels():
    leadlag, climbdive = labels(num_classes=3)
    ll, _, placebo, num_concepts = stress(leadlag, climbdive, add_placebo=True, base_num_concepts=3, seed=2)
    assert num_concepts == 4 and np.array_equal(ll, leadlag)
    assert (placebo[UNLABELED] == -1).all()
    assert np.array_equal(placebo[TRAIN], generate_placebo_labels(N, 2)[TRAIN])
    assert set(placebo[TRAIN]) == {0, 1}


@pytest.mark.parametrize("name, slot", [
    ("lead", 0), ("lag", 1), ("climb", 2), ("dive", 3), ("straight", 0), ("left", 1), ("right", 2), (None, None)])
def test_removed_concept_slot(name, slot):
    assert get_removed_concept_idx(AblationConfig(remove_concept=name)) == slot


def test_removal_zeroes_only_its_slot():
    concepts = torch.randn(5, 4, 3)
    zeroed = zero_removed_concept(concepts, AblationConfig(remove_concept="climb"))
    assert (zeroed[:, 2] == 0).all()
    assert torch.equal(zeroed[:, [0, 1, 3]], concepts[:, [0, 1, 3]])
    assert zero_removed_concept(concepts, AblationConfig()) is concepts
    leadlag, climbdive = labels()
    ll, cd, _, _, num_concepts, _ = apply_ablations(
        leadlag.copy(), climbdive.copy(), TRAIN, UNLABELED, AblationConfig(remove_concept="climb"))
    assert num_concepts == 4 and np.array_equal(ll, leadlag) and np.array_equal(cd, climbdive)
