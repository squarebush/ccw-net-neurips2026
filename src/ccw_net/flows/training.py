"""Fitting the donor flow on the current concepts of the labelled training rows."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ccw_net.flows.invertible_nf import PerSlotInvertibleNF

ENCODE_BATCH = 1000


def train_flow(model, train_idx, inputs, covariates, leadlag, climbdive, placebo, flow_cfg, batch_size, slot_to_label,
               axis_sizes, device):
    """Fit a PerSlotInvertibleNF to the model's concepts given labels and covariates; returns it frozen."""
    labeled = (leadlag[train_idx].cpu().numpy() >= 0) & (climbdive[train_idx].cpu().numpy() >= 0)
    if placebo is not None:
        labeled &= placebo[train_idx].cpu().numpy() >= 0
    labeled_idx = train_idx[labeled]

    model.eval()
    with torch.no_grad():
        concepts = torch.cat([model(inputs[labeled_idx[i:i + ENCODE_BATCH]], return_details=True)[1]
                              for i in range(0, len(labeled_idx), ENCODE_BATCH)])
    num_rows, num_slots, concept_dim = concepts.shape
    concepts = concepts.view(num_rows, -1)

    label_blocks = [F.one_hot(leadlag[labeled_idx].long(), axis_sizes[0]).float()]
    if len(axis_sizes) >= 2:
        label_blocks.append(F.one_hot(climbdive[labeled_idx].long(), axis_sizes[1]).float())
    if placebo is not None:
        label_blocks.append(placebo[labeled_idx].float().unsqueeze(1))
    context = torch.cat(label_blocks + [covariates[labeled_idx]], dim=1)

    flow_batch = min(flow_cfg.batch_size or batch_size, max(1, num_rows))
    num_batches = max(1, num_rows // flow_batch)
    flow = PerSlotInvertibleNF(
        num_slots=num_slots, concept_dim=concept_dim, cond_dim=covariates.shape[1], hidden_dim=flow_cfg.hidden_dim,
        num_coupling_layers=flow_cfg.num_coupling_layers, slot_to_label=slot_to_label,
        axis_sizes=list(axis_sizes) + ([1] if placebo is not None else []),
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=flow_cfg.lr)

    for epoch in range(1, flow_cfg.pretrain_epochs + 1):
        flow.train()
        perm = torch.randperm(num_rows, device=device)
        for b in tqdm(range(num_batches), desc=f"Flow epoch {epoch}/{flow_cfg.pretrain_epochs}", leave=False):
            idx = perm[b * flow_batch:(b + 1) * flow_batch]
            loss = flow.compute_loss(concepts[idx], context[idx])
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), float('inf'))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                continue
            if flow_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), flow_cfg.grad_clip)
            optimizer.step()

    flow.eval()
    flow.requires_grad_(False)
    return flow
