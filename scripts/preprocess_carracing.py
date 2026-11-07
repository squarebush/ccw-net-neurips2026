#!/usr/bin/env python3
"""Flatten the labelled CarRacing-v1 episodes into latent.npy, action.npy and concept.npy."""
import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="Source directory with X_train.pkl, real_actions.pkl, scenario*.npy")
    parser.add_argument("--dst", type=Path, default=Path("data/carracing"), help="Output directory")
    parser.add_argument("--num-labeled-episodes", type=int, default=20, help="Number of labelled episodes")
    args = parser.parse_args()

    with open(args.src / "X_train.pkl", "rb") as f:
        latent_episodes = pickle.load(f)
    with open(args.src / "real_actions.pkl", "rb") as f:
        action_episodes = pickle.load(f)

    latents, actions, concepts = [], [], []
    for i in range(args.num_labeled_episodes):
        x_ep = np.array(latent_episodes[i])
        a_ep = np.array(action_episodes[i])
        c_ep = np.load(args.src / f"scenario{i}.npy")
        length = min(len(x_ep), len(a_ep), len(c_ep))
        latents.append(x_ep[:length])
        actions.append(a_ep[:length])
        concepts.append(c_ep[:length])

    args.dst.mkdir(parents=True, exist_ok=True)
    np.save(args.dst / "latent.npy", np.concatenate(latents, axis=0).astype(np.float32))
    np.save(args.dst / "action.npy", np.concatenate(actions, axis=0).astype(np.float32))
    np.save(args.dst / "concept.npy", np.concatenate(concepts, axis=0).astype(np.int64))
    print(f"wrote latent.npy, action.npy and concept.npy to {args.dst}", flush=True)


if __name__ == "__main__":
    main()
