"""Score trained runs: one file of the paper's metrics per configuration and seed."""
import argparse
import sys
from pathlib import Path
from typing import Dict

from ccw_net.scoring.attribution import attribution
from ccw_net.scoring.disentanglement import dci
from ccw_net.scoring.independent import independent_scores
from ccw_net.scoring.models import load_model
from ccw_net.scoring.runs import (
    SEEDS,
    domain_of,
    load_config,
    load_test_predictions,
    paper_configs,
    run_dir,
    score_file,
    write_json,
)
from ccw_net.scoring.task import task_outputs


def score_run(run: Path, data_root: Path, seed: int) -> Dict:
    """Task metrics, DCI and, by domain, the CUB attribution or the independent metric of one trained run."""
    cfg = load_config(run)
    stored = load_test_predictions(run)
    model = load_model(run, cfg, int(stored['a_pred'].shape[1]))
    domain = domain_of(cfg)
    metrics, concepts = task_outputs(model, cfg, run, data_root, stored)
    scores = {'mse': metrics['mse'], 'acc_ll': metrics['acc_ll']}
    if domain != 'cub200':
        scores['acc_cd'] = metrics['acc_cd'] if domain == 'aircraft' else None
    scores['dci'] = dci(concepts, stored['leadlag'], stored['climbdive'])
    if domain == 'cub200':
        scores['attribution'] = attribution(model, concepts, stored['leadlag'], cfg['model']['name'])
    else:
        scores.update(independent_scores(model, cfg, stored, data_root, seed))
    return scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ccw_net.scoring', description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='trained runs, <runs>/<config>/seed_<n>')
    parser.add_argument('--scores', type=Path, required=True, help='output, <scores>/<config>/seed_<n>.json')
    parser.add_argument('--data-root', type=Path, required=True, help='raw data: 500K, carracing, cub200_finetuned')
    parser.add_argument('--configs', nargs='+', default=paper_configs(), help='config ids (default: all configs in configs/)')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), help='seeds (default: 0-9)')
    parser.add_argument('--force', action='store_true', help='rescore seeds that already have a score file')
    args = parser.parse_args(argv)

    jobs = [(config, seed) for config in args.configs for seed in args.seeds]
    failed = []
    for k, (config, seed) in enumerate(jobs, 1):
        out = score_file(args.scores, config, seed)
        if out.exists() and not args.force:
            print(f'[{k}/{len(jobs)}] {config} seed {seed}: already scored', flush=True)
            continue
        print(f'[{k}/{len(jobs)}] scoring {config} seed {seed}', flush=True)
        try:
            write_json(out, score_run(run_dir(args.runs, config, seed), args.data_root, seed))
        except Exception as exc:
            failed.append(f'{config} seed {seed}')
            print(f'  failed: {type(exc).__name__}: {exc}', flush=True)
    if failed:
        print(f'{len(failed)} of {len(jobs)} failed: {", ".join(failed)}', flush=True)
        return 1
    print(f'all {len(jobs)} scored into {args.scores}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
