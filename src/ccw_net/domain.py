"""Concept layout and label files of the aircraft, CarRacing and CUB-200 domains."""
from dataclasses import dataclass
from typing import List


@dataclass
class DomainConfig:
    """Concept names, concept groups (axes) with their sizes, the group of each slot and the label files."""

    name: str
    concept_names: List[str]
    num_axes: int
    axis_sizes: List[int]
    slot_to_axis: List[int]
    label_files: List[str]

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names)


F16_DOMAIN = DomainConfig(
    name="aircraft",
    concept_names=["Lead", "Lag", "Climb", "Dive"],
    num_axes=2,
    axis_sizes=[2, 2],
    slot_to_axis=[0, 0, 1, 1],
    label_files=["leadlag.npy", "climbdive.npy"],
)

CARRACING_DOMAIN = DomainConfig(
    name="carracing",
    concept_names=["Straight", "Left", "Right"],
    num_axes=1,
    axis_sizes=[3],
    slot_to_axis=[0, 0, 0],
    label_files=["concept.npy"],
)

CUB200_DOMAIN = DomainConfig(
    name="cub200",
    concept_names=[f"attr_{i}" for i in range(112)],
    num_axes=112,
    axis_sizes=[2] * 112,
    slot_to_axis=list(range(112)),
    label_files=["concept.npy"],
)

_DOMAIN_REGISTRY = {
    "aircraft": F16_DOMAIN,
    "f16": F16_DOMAIN,
    "carracing": CARRACING_DOMAIN,
    "cub200": CUB200_DOMAIN,
}


def get_domain(name: str) -> DomainConfig:
    """The DomainConfig registered under `name`."""
    if name not in _DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain: {name}. Available: {list(_DOMAIN_REGISTRY.keys())}")
    return _DOMAIN_REGISTRY[name]
