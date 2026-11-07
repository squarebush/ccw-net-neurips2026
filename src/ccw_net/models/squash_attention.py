"""CCW-Net: squashed concept slots read out by single-head attention."""
import torch
import torch.nn as nn

from ccw_net.utils import squash


class SquashAttention(nn.Module):
    """MLP encoder to J squashed D-dimensional concept slots and a single-head attention head with tanh output."""

    def __init__(self, z_dim, concept_dim, hidden_dim, a_dim, num_concepts, removed_slot=None):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.removed_slot = removed_slot
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts * concept_dim),
        )
        self.query_emb = nn.Parameter(torch.randn(a_dim, concept_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=concept_dim, num_heads=1, batch_first=True, bias=False)
        self.output_proj = nn.Parameter(torch.randn(a_dim, concept_dim) * 0.02)
        self.output_bias = nn.Parameter(torch.zeros(a_dim))

    def zero_removed(self, v):
        """The concepts with the removed slot (stress_test.remove_concept) set to zero."""
        if self.removed_slot is None:
            return v
        return v.index_fill(1, torch.tensor([self.removed_slot], device=v.device), 0.0)

    def forward_from_concepts(self, v):
        """Actions from concept slots [B, J, D]."""
        query = self.query_emb.unsqueeze(0).expand(v.shape[0], -1, -1)
        v = self.zero_removed(v)
        context, _ = self.attn(query=query, key=v, value=v)
        return torch.tanh((context * self.output_proj).sum(dim=-1) + self.output_bias)

    def forward(self, z, return_details=False):
        """(actions, concepts), plus slot norms and pre-squash slots when return_details."""
        s = self.zero_removed(self.encoder(z).view(z.shape[0], self.num_concepts, self.concept_dim))
        v = squash(s)
        a_hat = self.forward_from_concepts(v)
        if return_details:
            return a_hat, v, v.norm(dim=-1), s
        return a_hat, v
