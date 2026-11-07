import json
import os
from pathlib import Path

import pytest

from ccw_net.run import run_config

CONFIG = "carracing/width64/ccwnet_d1"
DATA_ROOT = os.environ.get("CCW_DATA_ROOT")
OVERRIDES = ("train.warmup_epochs=1", "train.flow.pretrain_epochs=1", "train.flow.refresh_every=1")


@pytest.mark.skipif(not DATA_ROOT or not Path(DATA_ROOT).is_dir(), reason="set CCW_DATA_ROOT to a data root")
def test_train_and_score_one_seed(run_module, tmp_path):
    runs, scores = tmp_path / "runs", tmp_path / "scores"
    run = run_config(CONFIG, 0, runs, DATA_ROOT, device="cpu", epochs=2, overrides=OVERRIDES)
    assert run == (runs / CONFIG / "seed_0").resolve()
    assert sorted(os.listdir(run)) == ["config.yaml", "model_final.pt", "test_predictions.npz"]
    result = run_module("ccw_net.scoring", "--runs", runs, "--scores", scores, "--data-root", DATA_ROOT,
                        "--configs", CONFIG, "--seeds", 0)
    assert result.returncode == 0, result.stdout + result.stderr
    score = json.loads((scores / CONFIG / "seed_0.json").read_text())
    assert set(score) == {"mse", "acc_ll", "acc_cd", "dci", "slots", "states", "overall", "unscored_states"}
    assert score["acc_cd"] is None and score["slots"] == ["Straight", "Left", "Right"]
