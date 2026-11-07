"""Independent causal attribution with matched test inputs."""
import itertools
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ccw_net.ablations import CARRACING_MERGES, CONCEPT_INDEX_MAP
from ccw_net.scoring.models import SAMPLED_ARCHS, check_decoding, encode
from ccw_net.scoring.runs import domain_of, stress_test

SEED_OFFSET = 42
SUBJECT_SEED_OFFSET = 1000
HEAD_BATCH = 8192
MIN_GROUP = 100
N_SUBJECTS = 1000
NUM_DRAWS = 96
CALIPER_SD = 1.5
KERNEL_TAU = 0.25
AIRCRAFT_SLOTS = ('Lead', 'Lag', 'Climb', 'Dive')
CARRACING_SLOTS = ('Straight', 'Left', 'Right')
AIRCRAFT_AXES = ('leadlag', 'climbdive')
PLACEBO_SLOT = 'Placebo'
COVARIATES = {'aircraft': 'observations.npy', 'carracing': 'latent.npy'}
STATE_FIELDS = ('pm', 'pm_active', 'pm_forbidden', 'active_slots')


def aircraft_contrasts(labels: Sequence[np.ndarray]) -> List[Dict]:
    """One axis flips, the other is held; groups are [flipping pair, held axis's two slots one by one]."""
    lead_lag, climb_dive = np.asarray(labels[0]), np.asarray(labels[1])
    out: List[Dict] = []
    for hold in (0, 1):
        out.append({'treated': (lead_lag == 1) & (climb_dive == hold),
                    'reference': (lead_lag == 0) & (climb_dive == hold),
                    'states': [[0, hold], [1, hold]], 'groups': [[0, 1], [2 + hold], [3 - hold]]})
        out.append({'treated': (climb_dive == 1) & (lead_lag == hold),
                    'reference': (climb_dive == 0) & (lead_lag == hold),
                    'states': [[hold, 0], [hold, 1]], 'groups': [[2, 3], [hold], [1 - hold]]})
    return out


def carracing_contrasts(labels: Sequence[np.ndarray]) -> List[Dict]:
    """One class against another; the two contested class slots are drawn jointly, the third class slot alone."""
    classes = np.asarray(labels[0])
    return [{'treated': classes == k, 'reference': classes == i, 'states': [[i], [k]], 'groups': [[k, i], [3 - i - k]]}
            for i, k in itertools.combinations(range(3), 2)]


def aircraft_merged(labels: Sequence[np.ndarray], merge: str) -> Tuple[List[Dict], List[Dict]]:
    """States by the unmerged axis with the merged slot active in both; the unmerged axis flips, then the merged one."""
    a = AIRCRAFT_AXES.index(merge)
    u = 1 - a
    axes = [np.asarray(labels[0]), np.asarray(labels[1])]
    merged, dead = 2 * a, 2 * a + 1
    merged_name = AIRCRAFT_SLOTS[merged] + '|' + AIRCRAFT_SLOTS[dead]
    states = []
    for v in (0, 1):
        parts = [AIRCRAFT_SLOTS[2 * u + v], merged_name]
        states.append({'key': [v], 'name': '+'.join(parts if u == 0 else parts[::-1]), 'mask': axes[u] == v,
                       'active': sorted([2 * u + v, merged])})
    contrasts = [{'treated': axes[u] == 1, 'reference': axes[u] == 0, 'states': [[0], [1]],
                  'groups': [[2 * u, 2 * u + 1], [merged], [dead]]}]
    for v in (0, 1):
        contrasts.append({'treated': (axes[a] == 1) & (axes[u] == v), 'reference': (axes[a] == 0) & (axes[u] == v),
                          'states': [[v], [v]], 'groups': [[merged, dead], [2 * u + v], [2 * u + 1 - v]]})
    return states, contrasts


def carracing_merged(labels: Sequence[np.ndarray], merge: str) -> Tuple[List[Dict], List[Dict]]:
    """States: the merged pair of classes and the remaining class; merged against remaining, then inside the pair."""
    a, b = CARRACING_MERGES[merge]
    r = 3 - a - b
    classes = np.asarray(labels[0])
    lo, hi = sorted((a, b))
    masks = {b: (classes == a) | (classes == b), r: classes == r}
    names = {b: CARRACING_SLOTS[lo] + '|' + CARRACING_SLOTS[hi], r: CARRACING_SLOTS[r]}
    first, second = sorted((b, r))
    states = [{'key': [s], 'name': names[s], 'mask': masks[s], 'active': [s]} for s in (first, second)]
    contrasts = [{'treated': masks[second], 'reference': masks[first], 'states': [[first], [second]],
                  'groups': [[second, first], [a]]},
                 {'treated': classes == hi, 'reference': classes == lo, 'states': [[b], [b]],
                  'groups': [[hi, lo], [r]]}]
    return states, contrasts


def canonical_states(domain: str, labels: Sequence[np.ndarray], removed) -> List[Dict]:
    """The label states and their active slots; a removed concept's slot is active in none."""
    if domain == 'aircraft':
        lead_lag, climb_dive = np.asarray(labels[0]), np.asarray(labels[1])
        states = [([ll, cd], AIRCRAFT_SLOTS[ll] + '+' + AIRCRAFT_SLOTS[2 + cd], (lead_lag == ll) & (climb_dive == cd),
                   [ll, 2 + cd]) for ll in (0, 1) for cd in (0, 1)]
    else:
        classes = np.asarray(labels[0])
        states = [([c], CARRACING_SLOTS[c], classes == c, [c]) for c in range(3)]
    return [{'key': key, 'name': name, 'mask': mask, 'active': [j for j in active if j != removed]}
            for key, name, mask, active in states]


def without_removed(contrast: Dict, removed: int) -> Dict:
    """The removed slot leaves the contested group and becomes a held one-slot group."""
    contested = contrast['groups'][0]
    if removed not in contested:
        return contrast
    return dict(contrast, groups=[[j for j in contested if j != removed]] + contrast['groups'][1:] + [[removed]])


def make_design(domain: str, labels: Sequence[np.ndarray], stress: Dict) -> Dict:
    """Slots, states and contrasts of a config; every contrast's groups partition the slots, its contested group first."""
    merge, remove = stress.get('merge_concept'), stress.get('remove_concept')
    base = AIRCRAFT_SLOTS if domain == 'aircraft' else CARRACING_SLOTS
    removed = None if remove is None else CONCEPT_INDEX_MAP[remove]
    if merge is not None:
        states, contrasts = (aircraft_merged if domain == 'aircraft' else carracing_merged)(labels, merge)
    else:
        states = canonical_states(domain, labels, removed)
        contrasts = (aircraft_contrasts if domain == 'aircraft' else carracing_contrasts)(labels)
        if removed is not None:
            contrasts = [without_removed(c, removed) for c in contrasts]
    placebo = [len(base)] if stress.get('add_placebo', False) else []
    contrasts = [dict(c, groups=c['groups'] + [[j] for j in placebo]) for c in contrasts]
    slots = list(base) + [PLACEBO_SLOT] * len(placebo)
    for c in contrasts:
        if sorted(j for group in c['groups'] for j in group) != list(range(len(slots))):
            raise ValueError(f"contrast groups {c['groups']} do not partition {len(slots)} slots")
    return {'slots': slots, 'states': states, 'contrasts': contrasts, 'placebo': placebo}


def match_weights(C_subjects: torch.Tensor, C_pool: torch.Tensor, subject_rows: torch.Tensor,
                  pool_rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian kernel weights of the pool rows within the caliper of each subject (self excluded), and distances."""
    Cb2 = (C_subjects ** 2).sum(dim=1, keepdim=True)
    Ck2 = (C_pool ** 2).sum(dim=1, keepdim=True).T
    cross = C_subjects @ C_pool.T
    D = (Cb2 + Ck2 - 2 * cross) / C_subjects.shape[1]
    eligible = (D <= CALIPER_SD ** 2) & (subject_rows[:, None] != pool_rows[None, :])
    sim = torch.exp(-D / (2.0 * KERNEL_TAU ** 2))
    W = torch.where(eligible, sim, torch.zeros_like(sim))
    W = W / (W.sum(dim=1, keepdim=True) + 1e-8)
    return W, D


def draw_donors(W: torch.Tensor, D: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """NUM_DRAWS pool rows per subject by Gumbel-max on the weights; the nearest row when none is in the caliper."""
    B, K = W.shape
    logits = torch.where(W > 1e-10, torch.log(W), torch.full_like(W, -float('inf')))
    fallback = W.sum(dim=1) <= 1e-10
    nearest = D.argmin(dim=1)
    draws = []
    for _ in range(NUM_DRAWS):
        u = torch.rand(B, K, device=W.device, generator=generator)
        g = -torch.log(-torch.log(u.clamp(1e-10, 1.0 - 1e-10)))
        draws.append(torch.where(fallback, nearest, (logits + g).argmax(dim=1)))
    return torch.stack(draws, dim=0)


@torch.no_grad()
def head(model, M: torch.Tensor) -> torch.Tensor:
    return torch.cat([model.forward_from_concepts(M[i:i + HEAD_BATCH]) for i in range(0, M.shape[0], HEAD_BATCH)])


@torch.no_grad()
def group_effect_norms(model, M: torch.Tensor, groups: List[List[int]], pool_rows: Dict,
                       draws: List[Dict]) -> List[float]:
    """|IE_g| per mediator group: the sequential V&D decomposition averaged over every ordering of the groups."""
    K = len(groups)
    n_subjects = draws[0][0].shape[1]
    outcomes = {}

    def mean_outcome(levels):
        key = tuple(levels)
        if key not in outcomes:
            total = None
            for s in range(NUM_DRAWS):
                M_batch = torch.zeros(n_subjects, M.shape[1], M.shape[2], dtype=M.dtype)
                for g in range(K):
                    rows = M[pool_rows[levels[g]][draws[g][levels[g]][s].numpy()]]
                    M_batch[:, groups[g], :] = rows[:, groups[g], :]
                Y = head(model, M_batch)
                total = Y if total is None else total + Y
            outcomes[key] = total / float(NUM_DRAWS)
        return outcomes[key]

    orders = list(itertools.permutations(range(K)))
    E_zero = mean_outcome([0] * K)
    per_group = np.zeros((K, n_subjects, E_zero.shape[1]))
    for order in orders:
        levels = [0] * K
        previous = E_zero
        for g in order:
            levels[g] = 1
            current = mean_outcome(levels)
            per_group[g] += (current - previous).numpy() / float(len(orders))
            previous = current
    IE = per_group.mean(1)
    return [float(np.linalg.norm(IE[g])) for g in range(K)]


def state_subjects(states: Sequence[Dict], seed: int) -> Dict[Tuple, np.ndarray]:
    """One fixed set of up to N_SUBJECTS test rows per state, from its own random stream."""
    rng = np.random.RandomState(seed + SUBJECT_SEED_OFFSET)
    out = {}
    for state in states:
        rows = np.flatnonzero(state['mask'])
        out[tuple(state['key'])] = rng.choice(rows, size=N_SUBJECTS, replace=False) if len(rows) > N_SUBJECTS else rows
    return out


def score_contrasts(model, M, covariates, dataset_indices, contrasts, subject_sets, seed) -> List[Dict]:
    """Group effect norms of every contrast with MIN_GROUP rows in both arms, on the fixed subjects of its states."""
    scored = []
    for contrast in contrasts:
        treated, reference = np.flatnonzero(contrast['treated']), np.flatnonzero(contrast['reference'])
        if len(treated) < MIN_GROUP or len(reference) < MIN_GROUP:
            continue
        keys = list(dict.fromkeys(tuple(s) for s in contrast['states']))
        subjects = np.concatenate([subject_sets[k] for k in keys])
        subjects = subjects[contrast['treated'][subjects] | contrast['reference'][subjects]]
        pool_rows = {1: treated, 0: reference}
        groups = contrast['groups']
        draws = [{} for _ in groups]
        for level, level_rows in pool_rows.items():
            W, D = match_weights(covariates[subjects], covariates[level_rows], dataset_indices[subjects],
                                 dataset_indices[level_rows])
            for g in range(len(groups)):
                draws[g][level] = draw_donors(W, D, torch.Generator().manual_seed(seed + 100 * g + level))
            del W, D
        scored.append({'states': contrast['states'], 'groups': groups,
                       'IE_norm': group_effect_norms(model, M, groups, pool_rows, draws)})
    return scored


def slot_sources(contrasts: Sequence[Dict], state: List[int], slot: int, active: List[int]) -> List[Tuple[int, int]]:
    """(contrast, group) sources of a slot in a state: its held groups, else, if active, its contested groups."""
    mine = [i for i, c in enumerate(contrasts) if state in c['states']]
    held = [(i, g) for i in mine for g, group in enumerate(contrasts[i]['groups']) if slot in group and g != 0]
    if held or slot not in active:
        return held
    return [(i, g) for i in mine for g, group in enumerate(contrasts[i]['groups']) if slot in group]


def state_table(contrasts: Sequence[Dict], design: Dict) -> Dict[str, Dict]:
    """Per scored state, PM_j = |IE_j| / sum_k |IE_k| with |IE_j| the mean over the slot's sources."""
    num_slots = len(design['slots'])
    table = {}
    for state in design['states']:
        active = state['active']
        found = [slot_sources(contrasts, state['key'], j, active) for j in range(num_slots)]
        if not all(found):
            continue
        effects = [float(np.mean([contrasts[i]['IE_norm'][g] for i, g in f])) for f in found]
        pm = np.asarray(effects) / sum(effects)
        forbidden = [j for j in range(num_slots) if j not in active]
        row = {'pm': [float(x) for x in pm], 'pm_active': float(pm[active].sum()) if active else None,
               'pm_forbidden': float(pm[forbidden].sum()), 'active_slots': active, 'n': int(state['mask'].sum())}
        if design['placebo']:
            row['pm_placebo'] = float(pm[design['placebo']].sum())
        table[state['name']] = row
    return table


def overall_summary(table: Dict[str, Dict]) -> Dict[str, float]:
    """Per-state values weighted by test count; PM_act and PM_forb only over states with an active slot."""
    def mean(key, names):
        weights = np.asarray([table[name]['n'] for name in names], float)
        return float(np.dot(weights, [table[name][key] for name in names]) / weights.sum())

    defined = [name for name, row in table.items() if row['pm_active'] is not None]
    overall = {'pm_active': mean('pm_active', defined), 'pm_forbidden': mean('pm_forbidden', defined)}
    if all('pm_placebo' in row for row in table.values()):
        overall['pm_placebo'] = mean('pm_placebo', list(table))
    return overall


def input_width(model) -> int:
    for name in ('encoder', 'classifier'):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Sequential):
            return module[0].in_features
    return model.concept_layer.in_features


def input_file(data_dir: Path, width: int) -> Path:
    return next(p for p in (data_dir / 'latent.npy', data_dir / 'observations.npy')
                if p.exists() and np.load(p, mmap_mode='r').shape[1] == width)


def test_concepts(model, cfg: Dict, stored: Dict, data_dir: Path) -> torch.Tensor:
    """Test-split concepts re-encoded from the raw inputs, or for btcvae/ivae the draws stored by training."""
    if cfg['model']['name'] in SAMPLED_ARCHS:
        return torch.from_numpy(np.asarray(stored['concepts'])).float()
    z = np.load(input_file(data_dir, input_width(model)), mmap_mode='r')
    a_pred, M = encode(model, z[np.asarray(stored['indices'])])
    check_decoding(model, M, a_pred)
    return M


def independent_scores(model, cfg: Dict, stored: Dict, data_root: Path, seed: int) -> Dict:
    """Slots, per-state shares, overall averages and unscored states of one seed."""
    domain = domain_of(cfg)
    data_dir = data_root / cfg['dataset']['data_subdir']
    test_idx = np.asarray(stored['indices'])
    M = test_concepts(model, cfg, stored, data_dir)
    if domain == 'aircraft':
        labels = [np.load(data_dir / name)[test_idx].astype(np.int64) for name in ('leadlag.npy', 'climbdive.npy')]
    else:
        labels = [np.load(data_dir / 'concept.npy')[test_idx].astype(np.int64)]
    C = np.asarray(np.load(data_dir / COVARIATES[domain], mmap_mode='r')[test_idx], dtype=np.float32)
    covariates = torch.tensor((C - C.mean(0)) / (C.std(0) + 1e-8))
    design = make_design(domain, labels, stress_test(cfg))
    estimator_seed = SEED_OFFSET + seed
    contrasts = score_contrasts(model, M, covariates, torch.tensor(test_idx.astype(np.int64)), design['contrasts'],
                                state_subjects(design['states'], estimator_seed), estimator_seed)
    table = state_table(contrasts, design)
    return {'slots': design['slots'],
            'states': {name: {key: row[key] for key in STATE_FIELDS} for name, row in table.items()},
            'overall': overall_summary(table),
            'unscored_states': [state['name'] for state in design['states'] if state['name'] not in table]}
