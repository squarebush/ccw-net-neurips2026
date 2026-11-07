"""Constructor of the trained concept models."""
from omegaconf import OmegaConf

from ccw_net.ablations import CONCEPT_INDEX_MAP
from ccw_net.models.cbm import CBM
from ccw_net.models.cwnet import CWNet
from ccw_net.models.squash_attention import SquashAttention


def create_model(cfg, z_dim, a_dim, num_concepts, device):
    """The model `cfg.model.name` names (squash_attention, cwnet or cbm), on `device`."""
    name = cfg.model.name
    if name == "squash_attention":
        removed = OmegaConf.select(cfg, 'stress_test.remove_concept')
        model = SquashAttention(z_dim, cfg.model.concept_dim, cfg.model.hidden_dim, a_dim, num_concepts,
                                removed_slot=None if removed is None else CONCEPT_INDEX_MAP[removed])
    elif name == "cwnet":
        model = CWNet(z_dim, cfg.model.concept_dim, a_dim, num_concepts,
                      output_activation=cfg.model.get('output_activation', 'tanh'))
    elif name == "cbm":
        model = CBM(z_dim, cfg.model.concept_dim, a_dim, num_concepts)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)
