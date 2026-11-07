"""Concept bottleneck model of Koh et al. (2020), written for this repository without their code."""
import torch
import torch.nn as nn


class CBM(nn.Module):
    """Linear concept layer on backbone features, sigmoid, linear class head."""

    def __init__(self, z_dim, concept_dim, a_dim, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.concept_layer = nn.Linear(z_dim, num_concepts * concept_dim)
        self.classifier = nn.Linear(num_concepts * concept_dim, a_dim)

    def forward_from_concepts(self, v):
        """Class logits from concept logits [B, J, D]."""
        return self.classifier(torch.sigmoid(v.reshape(v.shape[0], -1)))

    def forward(self, z, return_details=False):
        """(class logits, concept logits), plus concept probabilities and concept logits when return_details."""
        v = self.concept_layer(z).view(z.shape[0], self.num_concepts, self.concept_dim)
        a_hat = self.forward_from_concepts(v)
        if return_details:
            probs = torch.sigmoid(v.squeeze(-1)) if self.concept_dim == 1 else torch.sigmoid(v.norm(dim=-1))
            return a_hat, v, probs, v
        return a_hat, v
