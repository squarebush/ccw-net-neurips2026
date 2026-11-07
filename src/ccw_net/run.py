"""Train one paper config for one seed from its stored config in configs/."""
import argparse
import os
import shutil
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DISENTANGLEMENT_MODELS = ("btcvae", "ivae")


def build_config(config, seed, data_root, device=None, epochs=None, overrides=()):
    """Stored config of `config` with seed, data root and optional overrides applied."""
    cfg = OmegaConf.load(CONFIGS_DIR / f"{config}.yaml")
    seed_overrides = cfg.pop("seed_overrides", None)
    cfg = OmegaConf.merge({"seed": seed}, cfg)
    if seed_overrides is not None and seed in seed_overrides:
        cfg = OmegaConf.merge(cfg, seed_overrides[seed])
    cfg.system.data_root = str(Path(data_root).resolve())
    if device is not None:
        cfg.system.device = device
    if epochs is not None:
        cfg.train.epochs = epochs
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def pick_device(name=None):
    """The named torch device, else CUDA, MPS or CPU, whichever is available first."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_config(config, seed, runs, data_root, device=None, epochs=None, overrides=(), force=False):
    """Train `config` at `seed` into <runs>/<config>/seed_<seed>, which it returns; a trained run is kept unless force."""
    run_dir = (Path(runs) / config / f"seed_{seed}").resolve()
    if (run_dir / "model_final.pt").exists() and not force:
        print(f"{run_dir} is already trained; pass --force to retrain it", flush=True)
        return run_dir

    cfg = build_config(config, seed, data_root, device=device, epochs=epochs, overrides=overrides)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), run_dir / "config.yaml")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cfg.model.name in DISENTANGLEMENT_MODELS:
        from ccw_net.train_disentanglement import train
    else:
        from ccw_net.train import train
    train(cfg, run_dir, pick_device(cfg.system.get("device")))
    print(f"trained {config} seed {seed}: {run_dir}", flush=True)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ccw_net.run", description=__doc__)
    parser.add_argument("--config", required=True, help="config id, e.g. aircraft/width256/ccwnet")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runs", default=os.environ.get("CCW_RUNS", str(REPO_ROOT / "runs")))
    parser.add_argument("--data-root", default=os.environ.get("CCW_DATA_ROOT", str(REPO_ROOT / "data")))
    parser.add_argument("--device", default=None, help="cpu, cuda or mps; default picks the best available")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs (smoke tests only)")
    parser.add_argument("--force", action="store_true", help="retrain a run that already has model_final.pt")
    args = parser.parse_args(argv)
    if not (CONFIGS_DIR / f"{args.config}.yaml").is_file():
        parser.error(f"unknown config {args.config}: no {CONFIGS_DIR / args.config}.yaml")
    warnings.filterwarnings("ignore")
    run_config(args.config, args.seed, args.runs, args.data_root, device=args.device, epochs=args.epochs, force=args.force)


if __name__ == "__main__":
    main()
