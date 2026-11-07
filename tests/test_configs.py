from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ccw_net.run import CONFIGS_DIR, build_config

CONFIG_IDS = sorted(path.relative_to(CONFIGS_DIR).with_suffix("").as_posix() for path in CONFIGS_DIR.rglob("*.yaml"))
PLACEBO_FIX = {"aircraft/width256/placebo", "aircraft/width64/placebo", "carracing/width256/placebo"}


@pytest.mark.parametrize("config", CONFIG_IDS)
def test_config_builds(config, tmp_path):
    assert OmegaConf.load(CONFIGS_DIR / f"{config}.yaml")
    for seed in (0, 3):
        cfg = build_config(config, seed, tmp_path)
        assert cfg.seed == seed
        assert "seed_overrides" not in cfg
        assert cfg.system.data_root == str(Path(tmp_path).resolve())


def test_seed_overrides_apply(tmp_path):
    assert build_config("carracing/width256/placebo", 3, tmp_path).train.flow.batch_size == ""
    assert build_config("carracing/width256/placebo", 0, tmp_path).train.flow.batch_size == 2048


def test_placebo_align_fix_configs():
    fixed = {config for config in CONFIG_IDS if OmegaConf.load(CONFIGS_DIR / f"{config}.yaml").get("train", {}).get("placebo_align_fix")}
    assert fixed == PLACEBO_FIX
    assert all(OmegaConf.load(CONFIGS_DIR / f"{config}.yaml").train.placebo_align_fix is True for config in PLACEBO_FIX)
