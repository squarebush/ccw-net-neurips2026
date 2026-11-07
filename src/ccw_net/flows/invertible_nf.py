"""Per-slot conditional normalising flow that draws counterfactual concept donors."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SLOT_EMBED_DIM = 4


class ConditionerMLP(nn.Module):
    """Shift and bounded log-scale of an affine coupling from the fixed half and the context."""

    def __init__(self, input_dim, context_dim, output_dim, hidden_dim, num_layers=2, scale_bound=2.0):
        super().__init__()
        self.scale_bound = scale_bound
        layers = [nn.Linear(input_dim + context_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.net = nn.Sequential(*layers)
        self.shift_head = nn.Linear(hidden_dim, output_dim)
        self.log_scale_head = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        nn.init.zeros_(self.log_scale_head.weight)
        nn.init.zeros_(self.log_scale_head.bias)

    def forward(self, x_fixed, context):
        h = self.net(torch.cat([x_fixed, context], dim=-1))
        return self.shift_head(h), torch.tanh(self.log_scale_head(h)) * self.scale_bound


class ConditionalAffineCoupling(nn.Module):
    """Affine coupling layer that transforms the unmasked coordinates."""

    def __init__(self, dim, mask, context_dim, hidden_dim):
        super().__init__()
        self.register_buffer('mask', mask)
        fixed_dim = int(mask.sum().item())
        self.conditioner = ConditionerMLP(fixed_dim, context_dim, dim - fixed_dim, hidden_dim)

    def forward(self, x, context):
        x_fixed = x[:, self.mask.bool()]
        shift, log_scale = self.conditioner(x_fixed, context)
        y = torch.zeros_like(x)
        y[:, self.mask.bool()] = x_fixed
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(log_scale) + shift
        return y, log_scale.sum(dim=-1)

    def inverse(self, y, context):
        y_fixed = y[:, self.mask.bool()]
        shift, log_scale = self.conditioner(y_fixed, context)
        x = torch.zeros_like(y)
        x[:, self.mask.bool()] = y_fixed
        x[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - shift) * torch.exp(-log_scale)
        return x


class ConditionalAffineScalar(nn.Module):
    """Stack of context-conditioned affine maps for one-dimensional concepts."""

    def __init__(self, context_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
            for _ in range(num_layers)
        ])

    def forward(self, x, context):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            params = layer(context)
            mu, log_sigma = params[:, 0:1], params[:, 1:2]
            z = (z - mu) * torch.exp(-log_sigma)
            log_det -= log_sigma.squeeze(-1)
        return z, log_det

    def inverse(self, z, context):
        x = z
        for layer in reversed(self.layers):
            params = layer(context)
            mu, log_sigma = params[:, 0:1], params[:, 1:2]
            x = x * torch.exp(log_sigma) + mu
        return x


class ConditionalInvertibleFlow(nn.Module):
    """Affine coupling layers with alternating masks."""

    def __init__(self, dim, context_dim, hidden_dim, num_coupling_layers):
        super().__init__()
        masks = []
        for i in range(num_coupling_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1
            masks.append(mask)
        self.coupling_layers = nn.ModuleList([
            ConditionalAffineCoupling(dim, mask, context_dim, hidden_dim) for mask in masks
        ])

    def forward(self, x, context):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.coupling_layers:
            z, layer_log_det = layer.forward(z, context)
            log_det = log_det + layer_log_det
        return z, log_det

    def inverse(self, z, context):
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, context)
        return x


class PerSlotInvertibleNF(nn.Module):
    """One conditional flow per concept slot, given that slot's label block, a slot embedding and covariates."""

    def __init__(self, num_slots, concept_dim, cond_dim, hidden_dim, num_coupling_layers, slot_to_label, axis_sizes):
        super().__init__()
        self.num_slots = num_slots
        self.concept_dim = concept_dim
        self.slot_to_label = list(slot_to_label)
        self.axis_sizes = list(axis_sizes)
        self.slot_label_dim = max(self.axis_sizes)
        self.slot_embeddings = nn.Embedding(num_slots, SLOT_EMBED_DIM)
        slot_context_dim = self.slot_label_dim + cond_dim + SLOT_EMBED_DIM
        if concept_dim == 1:
            self.flows = nn.ModuleList([
                ConditionalAffineScalar(slot_context_dim, hidden_dim, num_coupling_layers) for _ in range(num_slots)
            ])
        else:
            self.flows = nn.ModuleList([
                ConditionalInvertibleFlow(concept_dim, slot_context_dim, hidden_dim, num_coupling_layers)
                for _ in range(num_slots)
            ])

    def _parse_context(self, context):
        blocks = []
        offset = 0
        for size in self.axis_sizes:
            blocks.append(context[:, offset:offset + size])
            offset += size
        return [blocks[self.slot_to_label[j]] for j in range(self.num_slots)], context[:, offset:]

    def _slot_context(self, j, label_j, cond):
        if label_j.shape[1] < self.slot_label_dim:
            label_j = F.pad(label_j, (0, self.slot_label_dim - label_j.shape[1]))
        slot_idx = torch.full((label_j.shape[0],), j, dtype=torch.long, device=label_j.device)
        return torch.cat([label_j, self.slot_embeddings(slot_idx), cond], dim=-1)

    def compute_loss(self, x, context):
        """Summed per-slot negative log-likelihood of the flattened concepts x."""
        dim = self.concept_dim
        labels, cond = self._parse_context(context)
        total_nll = 0.0
        for j in range(self.num_slots):
            z_j, log_det_j = self.flows[j].forward(x[:, j * dim:(j + 1) * dim], self._slot_context(j, labels[j], cond))
            log_p_base = -0.5 * (z_j.pow(2).sum(-1) + dim * np.log(2 * np.pi))
            total_nll = total_nll + -(log_p_base + log_det_j).mean()
        return total_nll

    @torch.no_grad()
    def sample(self, context, num_samples):
        """num_samples draws of the flattened concepts per context row, shaped [B, num_samples, J * D]."""
        labels, cond = self._parse_context(context)
        samples = []
        for _ in range(num_samples):
            slots = []
            for j in range(self.num_slots):
                z = torch.randn(context.shape[0], self.concept_dim, device=context.device)
                slots.append(self.flows[j].inverse(z, self._slot_context(j, labels[j], cond)))
            samples.append(torch.cat(slots, dim=-1))
        return torch.stack(samples, dim=1)
