"""Task metrics and concepts of the trained weights on the full test split."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from ccw_net.scoring.models import SAMPLED_ARCHS, check_decoding
from ccw_net.scoring.runs import load_stored_test_metrics


def training_data(cfg: Dict, data_root: Path) -> Dict:
    """The run's data as training saw it: load_data's split, apply_ablations' labels and placebo labels."""
    from ccw_net.ablations import ablation_config, apply_ablations
    from ccw_net.data import load_data
    conf = OmegaConf.create(cfg)
    data = load_data(data_root / cfg['dataset']['data_subdir'], conf)
    ablation = ablation_config(conf, data['domain'].num_concepts)
    leadlag, climbdive, _, placebo, num_concepts, _ = apply_ablations(
        data['leadlag'], data['climbdive'], data['train_labeled_idx'], data['unlabeled_idx'], ablation,
        action=data.get('action'))
    data.update({'leadlag': leadlag, 'climbdive': climbdive, 'placebo': placebo, 'num_concepts': num_concepts,
                 'ablation': ablation})
    return data


def require_equal(name: str, stored: np.ndarray, regenerated: np.ndarray) -> None:
    if stored.shape != regenerated.shape or not np.array_equal(stored, regenerated):
        raise RuntimeError(f'regenerated test {name} differs from the test predictions')


def test_tensors(data: Dict, stored: Dict) -> Dict:
    """The test rows of every eval_on_split input, as training builds them, checked against the stored test split."""
    rows = data['test_idx']
    require_equal('indices', np.asarray(stored['indices']), rows)
    for name in ('leadlag', 'climbdive'):
        require_equal(name, np.asarray(stored[name]), data[name][rows])
    action = data['action'][rows]
    integer_action = action.ndim == 1 and np.issubdtype(action.dtype, np.integer)
    a_true = np.asarray(stored['a_true'])
    if integer_action and a_true.ndim == 2:
        action_as_stored = np.eye(a_true.shape[1], dtype=a_true.dtype)[action]
    else:
        action_as_stored = action if integer_action else action.astype(np.float32)
    require_equal('actions', a_true, action_as_stored)
    return {
        'input': torch.tensor(data['input'][rows], dtype=torch.float32),
        'action': torch.tensor(action, dtype=torch.long if integer_action else torch.float32),
        'leadlag': torch.tensor(data['leadlag'][rows], dtype=torch.long),
        'climbdive': torch.tensor(data['climbdive'][rows], dtype=torch.long),
        'covariates': torch.tensor(data['covariates'][rows], dtype=torch.float32),
        'placebo': None if data['placebo'] is None else torch.tensor(data['placebo'][rows], dtype=torch.long),
    }


def re_encoded(model, cfg: Dict, data_root: Path, stored: Dict) -> Tuple[Dict[str, float], np.ndarray]:
    """eval_on_split over every test row and the concepts of every test row."""
    from ccw_net.evaluation import eval_on_split
    data = training_data(cfg, data_root)
    t = test_tensors(data, stored)
    eval_cfg = OmegaConf.create(cfg)
    eval_cfg.train.eval_samples = len(t['input'])
    metrics = eval_on_split(model, np.arange(len(t['input'])), t['input'], t['action'], t['leadlag'],
                            t['climbdive'], t['covariates'], t['placebo'], None, eval_cfg, data['num_concepts'],
                            data['ablation'], 'cpu', slot_to_axis=data['domain'].slot_to_axis)
    with torch.no_grad():
        outputs = model(t['input'], return_details=True)
    a_pred, concepts = outputs[0], outputs[1]
    check_decoding(model, concepts, a_pred)
    return metrics, concepts.numpy()


def stored_sample(model, run: Path, stored: Dict) -> Tuple[Dict[str, float], np.ndarray]:
    """btcvae/ivae sample their concepts even in eval: the stored test metrics and concept draws of training."""
    concepts = torch.from_numpy(np.asarray(stored['concepts'])).float()
    check_decoding(model, concepts, torch.from_numpy(np.asarray(stored['a_pred'])).float())
    return load_stored_test_metrics(run), concepts.numpy()


def task_outputs(model, cfg: Dict, run: Path, data_root: Path, stored: Dict) -> Tuple[Dict[str, float], np.ndarray]:
    """The test metrics and the test concepts of the trained weights."""
    if cfg['model']['name'] in SAMPLED_ARCHS:
        return stored_sample(model, run, stored)
    return re_encoded(model, cfg, data_root, stored)
