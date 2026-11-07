# Causal Concept Explanations for Deep Neural Models

Code for the NeurIPS 2026 paper *Causal Concept Explanations for Deep Neural Models* by Josh Rountree, Pulkit Verma,
Oswin So, Chuchu Fan, and Julie A. Shah.

CCW-Net (Causal Concept Wrapper Network) wraps a frozen pretrained backbone with a concept encoder and an attention
head. It trains them with causal alignment losses (necessity, sufficiency, independence, and presence) from
interventional mediation analysis, so that each concept's effect on the output can be measured.

## Install

```bash
conda env create -f environment.yml
conda activate ccw-net
```

Or, in an existing Python 3.10 environment on Linux:

```bash
pip install -r requirements.txt
pip install -e .
```

On macOS, use the conda environment. The PyPI SciPy 1.15 wheels for Python 3.10 fail to load on recent macOS releases.

## Print the paper's tables

`scores/` holds the per-seed scores behind every table in the paper (87 configurations, seeds 0-9).

```bash
python -m ccw_net.tables
```

This prints Tables 1, 2, 3, 8, and 9.

## Data

Put the datasets under one data root (`data/` by default). Each directory holds `.npy` arrays.

| Directory | Task | Arrays | Source |
|---|---|---|---|
| `500K/`, `5K/`, `50K/`, `5M/` | Aircraft control | `observations`, `latent`, `action`, `leadlag`, `climbdive` | Generated with the F-16 environment of So and Fan (RSS 2023), [jax-f16](https://github.com/MIT-REALM/jax-f16), which implements the F-16 model of Heidlauf et al. (ARCH 2018). Download from the [v1.0 release](https://github.com/squarebush/ccw-net-neurips2026/releases/tag/v1.0) |
| `carracing/` | CarRacing | `latent`, `action`, `concept` | From the CW-Net release of Kenny et al. (2026, CC0). Download `code_ocean_capsule/data/` from [CW-Net-Autonomous-Driving](https://github.com/EoinKenny/CW-Net-Autonomous-Driving) and run `python scripts/preprocess_carracing.py --src <that folder>` |
| `cub200_finetuned/` | CUB-200 | `latent`, `class`, `concept` | Built from CUB-200-2011 with the two commands below |

The CUB-200-2011 images are not redistributed here. To build `cub200_finetuned/`, download CUB-200-2011 and the torchvision
Inception-v3 ImageNet weights (`inception_v3_google-0cc3c7bd.pth`), then run:

```bash
python scripts/preprocess_cub.py --cub-dir <CUB_200_2011>
python scripts/train_cub_backbone.py --cub-dir <CUB_200_2011> --weights <inception_v3_google-0cc3c7bd.pth>
```

## Reproduce

Each configuration in the paper is a file `configs/<task>/<group>/<name>.yaml`, with `<task>` one of `aircraft`,
`carracing`, and `cub200`. The groups are:

- `width64/`: the models of Tables 1 and 2 and the encoder-width-64 rows of Appendix I.
- `width256/`: the other ablations of Appendix I.
- `cwnet/`: CW-Net and its variants.
- `disentanglement/`: β-TCVAE and iVAE.
- `cub200/` holds the two models of Table 3 directly.

Train and score one configuration and seed:

```bash
python -m ccw_net.run --config aircraft/width64/ccwnet --seed 0 --data-root data --runs runs
python -m ccw_net.scoring --runs runs --scores results/scores --data-root data --configs aircraft/width64/ccwnet --seeds 0
```

`bash reproduce.sh` trains and scores every configuration and seed, then prints the tables with
`python -m ccw_net.tables --scores results/scores`.

The trained runs will be posted on the [Releases page](https://github.com/squarebush/ccw-net-neurips2026/releases). Scoring them as above reproduces `scores/` exactly. Training on a
GPU is not bitwise deterministic, so retrained models reproduce the tables within seed-to-seed variation.

## Citation

```bibtex
@inproceedings{rountree2026causal,
  title     = {Causal Concept Explanations for Deep Neural Models},
  author    = {Rountree, Josh and Verma, Pulkit and So, Oswin and Fan, Chuchu and Shah, Julie A.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

The code is released under the MIT License (`LICENSE`). `src/ccw_net/models/cwnet.py` is adapted from the CW-Net code
of Kenny et al. (Apache-2.0, `LICENSES/Apache-2.0.txt`). See `NOTICE`. The CarRacing data are from the CW-Net release (CC0). The datasets keep
their own licenses.
