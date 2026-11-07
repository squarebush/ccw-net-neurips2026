"""Adapted from the CW-Net release of Kenny et al. (Copyright 2026 Motional, Apache-2.0) and modified for this repository, see NOTICE."""
import torch
import torch.nn as nn


class CWNet(nn.Module):
    """Concept classifier and action ranker MLPs of Kenny et al. with the capsule's layer indices, 10% hidden width."""

    def __init__(self, z_dim, concept_dim, a_dim, num_concepts, output_activation='tanh'):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.output_activation = output_activation
        hidden = max(1, int(z_dim * 0.1))
        flat_concepts = num_concepts * concept_dim
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(hidden, flat_concepts),
        )
        self.ranker = nn.Sequential(
            nn.Linear(flat_concepts, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(hidden, a_dim),
        )

    def forward_from_concepts(self, v):
        """Actions from concepts [B, J, D]: tanh, or tanh steering with ReLU gas and brake for CarRacing."""
        a_pre = self.ranker(v.reshape(v.shape[0], -1))
        if self.output_activation == 'carracing':
            a_out = torch.empty_like(a_pre)
            a_out[:, 0] = torch.tanh(a_pre[:, 0])
            a_out[:, 1] = torch.relu(a_pre[:, 1])
            a_out[:, 2] = torch.relu(a_pre[:, 2])
            return a_out
        return torch.tanh(a_pre)

    def forward(self, z, return_details=False):
        """(actions, concepts), plus concept scores and raw concepts when return_details."""
        v = self.classifier(z).view(z.shape[0], self.num_concepts, self.concept_dim)
        a_hat = self.forward_from_concepts(v)
        if return_details:
            return a_hat, v, v.norm(dim=-1) if self.concept_dim > 1 else v.squeeze(-1).abs(), v
        return a_hat, v
