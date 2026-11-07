"""Rebuild a trained model from its run directory."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from ccw_net.scoring.runs import num_slots

SAMPLED_ARCHS = ('btcvae', 'ivae')
INPUT_KEYS = ('encoder.0.weight', 'classifier.0.weight', 'concept_layer.weight')
MAX_DECODE_ERROR = 1e-4


def build_trained(cfg: Dict, state: Dict, a_dim: int) -> torch.nn.Module:
    """The constructor training used: create_model, or the disentanglement-baseline classes for btcvae/ivae."""
    model_cfg, J = cfg['model'], num_slots(cfg)
    if model_cfg['name'] in SAMPLED_ARCHS:
        from ccw_net.models.disentanglement_baselines import IVAE, BetaTCVAE
        dims = dict(concept_dim=model_cfg['concept_dim'], hidden_dim=model_cfg['hidden_dim'], a_dim=a_dim,
                    num_concepts=J)
        if model_cfg['name'] == 'ivae':
            u_dim = int(state['prior_net.0.weight'].shape[1])
            return IVAE(z_dim=int(state['encoder.0.weight'].shape[1]) - u_dim, u_dim=u_dim, **dims)
        return BetaTCVAE(z_dim=int(state['encoder.0.weight'].shape[1]), **dims)
    from ccw_net.models.setup import create_model
    z_dim = next(int(state[k].shape[1]) for k in INPUT_KEYS if k in state)
    return create_model(OmegaConf.create(cfg), z_dim, a_dim, J, 'cpu')


def load_model(run: Path, cfg: Dict, a_dim: int) -> torch.nn.Module:
    """The weights of model_final.pt in the training constructor on CPU, every key required to match."""
    saved = torch.load(run / 'model_final.pt', map_location='cpu', weights_only=False)
    state = saved.get('model_state_dict', saved)
    model = build_trained(cfg, state, a_dim)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def check_decoding(model: torch.nn.Module, concepts: torch.Tensor, outputs: torch.Tensor) -> None:
    """Fail when the head applied to the concepts does not reproduce the model's outputs."""
    error = float((model.forward_from_concepts(concepts) - outputs).abs().max())
    if error > MAX_DECODE_ERROR:
        raise RuntimeError(f'forward vs forward_from_concepts max|err|={error:.5f}')


@torch.no_grad()
def encode(model: torch.nn.Module, z: np.ndarray, batch: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
    """Outputs and concepts of the inputs z through the loaded weights, in batches."""
    zt = torch.from_numpy(np.asarray(z)).float()
    actions, concepts = [], []
    for start in range(0, len(zt), batch):
        out = model(zt[start:start + batch], return_details=True)
        actions.append(out[0])
        concepts.append(out[1])
    return torch.cat(actions), torch.cat(concepts)
