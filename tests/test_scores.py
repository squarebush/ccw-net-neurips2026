import json

import pytest
from omegaconf import OmegaConf

from ccw_net.tables.layout import CONFIGS

SEED_FILES = sorted(f"seed_{seed}.json" for seed in range(10))
DCI_KEYS = {"disentanglement", "completeness", "informativeness"}
STATE_KEYS = {"pm", "pm_active", "pm_forbidden", "active_slots"}
TOP_KEYS = {
    "aircraft": {"mse", "acc_ll", "acc_cd", "dci", "slots", "states", "overall", "unscored_states"},
    "carracing": {"mse", "acc_ll", "acc_cd", "dci", "slots", "states", "overall", "unscored_states"},
    "cub200": {"mse", "acc_ll", "dci", "attribution"},
}


def config_ids(repo):
    return sorted(path.relative_to(repo / "configs").with_suffix("").as_posix()
                  for path in (repo / "configs").rglob("*.yaml"))


def test_configs_match_tables(repo):
    ids = config_ids(repo)
    assert len(ids) == 87
    assert ids == sorted(config["id"] for config in CONFIGS)


def test_every_config_has_ten_seeds(repo):
    scored = sorted(path.parent.relative_to(repo / "scores").as_posix() for path in (repo / "scores").rglob("*.json"))
    assert sorted(set(scored)) == config_ids(repo)
    for config in config_ids(repo):
        assert sorted(path.name for path in (repo / "scores" / config).iterdir()) == SEED_FILES


@pytest.mark.parametrize("domain", sorted(TOP_KEYS))
def test_score_schema(repo, domain):
    for path in sorted((repo / "scores" / domain).rglob("*.json")):
        config = path.parent.relative_to(repo / "scores").as_posix()
        scores = json.loads(path.read_text())
        assert set(scores) == TOP_KEYS[domain], path
        assert set(scores["dci"]) == DCI_KEYS, path
        if domain == "cub200":
            assert set(scores["attribution"]) == {"pm_active", "pm_forbidden"}, path
            continue
        assert isinstance(scores["acc_cd"], float) if domain == "aircraft" else scores["acc_cd"] is None
        placebo = OmegaConf.load(repo / "configs" / f"{config}.yaml").stress_test.add_placebo
        assert set(scores["overall"]) == {"pm_active", "pm_forbidden"} | ({"pm_placebo"} if placebo else set()), path
        assert scores["states"], path
        for state in scores["states"].values():
            assert set(state) == STATE_KEYS, path
            assert len(state["pm"]) == len(scores["slots"]), path
        assert isinstance(scores["unscored_states"], list), path
