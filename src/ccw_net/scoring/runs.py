"""Run and score directory layout."""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[3] / 'configs'
SEEDS = tuple(range(10))
DOMAINS = {'f16': 'aircraft', 'carracing': 'carracing', 'cub200': 'cub200'}


def paper_configs() -> List[str]:
    """The ids of the configurations under configs/."""
    return sorted(path.relative_to(CONFIGS_DIR).with_suffix('').as_posix() for path in CONFIGS_DIR.rglob('*.yaml'))


def run_dir(runs: Path, config: str, seed: int) -> Path:
    return runs / config / f'seed_{seed}'


def score_file(scores: Path, config: str, seed: int) -> Path:
    return scores / config / f'seed_{seed}.json'


def load_config(run: Path) -> Dict:
    return yaml.safe_load((run / 'config.yaml').read_text())


def domain_of(cfg: Dict) -> str:
    return DOMAINS[cfg['dataset']['domain']]


def stress_test(cfg: Dict) -> Dict:
    return cfg.get('stress_test') or {}


def num_slots(cfg: Dict) -> int:
    """Concept slots of the trained model: the configured concepts plus the placebo slot."""
    return int(cfg['model']['num_concepts']) + int(bool(stress_test(cfg).get('add_placebo', False)))


def load_test_predictions(run: Path) -> Dict[str, np.ndarray]:
    with np.load(run / 'test_predictions.npz', allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def load_stored_test_metrics(run: Path) -> Dict[str, float]:
    return json.loads((run / 'final_metrics.json').read_text())['test']


def write_json(path: Path, payload: Dict) -> None:
    """Write atomically, so an interrupted run never leaves a partial score file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.replace(path)
