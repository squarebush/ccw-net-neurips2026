"""Training of the CCW-Net, CW-Net and CBM models."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ccw_net.ablations import ablation_config, apply_ablations, get_removed_concept_idx, zero_removed_concept
from ccw_net.data import load_data
from ccw_net.flows.training import train_flow
from ccw_net.losses.causal_alignment import causal_alignment_loss
from ccw_net.losses.supervised import supervised_losses
from ccw_net.models.setup import create_model


def alignment_labels(leadlag, climbdive, placebo, slot_to_axis, num_concepts):
    """Per-slot 0/1 concept labels for the alignment losses; rows missing any label are all -1."""
    labels = torch.zeros(len(leadlag), num_concepts, device=leadlag.device)
    missing = torch.zeros(len(leadlag), dtype=torch.bool, device=leadlag.device)
    for axis, axis_labels in enumerate((leadlag, climbdive)[:max(slot_to_axis) + 1]):
        slots = [j for j, a in enumerate(slot_to_axis) if a == axis]
        for class_idx, j in enumerate(slots):
            labels[:, j] = (axis_labels == class_idx).float()
        missing |= axis_labels < 0
    if placebo is not None:
        labels[:, len(slot_to_axis):] = placebo.float().unsqueeze(1)
        missing |= placebo < 0
    labels[missing] = -1.0
    return labels


def flow_refit_epoch(epoch, warmup, refresh_every):
    """Whether the donor flow is fitted before `epoch`: right after warm-up, then every refresh_every epochs."""
    if epoch == warmup + 1:
        return True
    return epoch > warmup and refresh_every > 0 and (epoch - warmup) % refresh_every == 1


def train(cfg, run_dir, device):
    """Train the configuration `cfg` on `device`; writes model_final.pt and test_predictions.npz to run_dir."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data = load_data(Path(cfg.system.data_root) / cfg.dataset.data_subdir, cfg)
    domain = data['domain']
    slot_to_axis = domain.slot_to_axis
    ablation = ablation_config(cfg, domain.num_concepts)
    leadlag_np, climbdive_np, _, placebo_np, num_concepts, _ = apply_ablations(
        data['leadlag'], data['climbdive'], data['train_labeled_idx'], data['unlabeled_idx'], ablation,
        action=data['action'])
    removed_slot = get_removed_concept_idx(ablation)
    train_idx, test_idx = data['train_idx'], data['test_idx']

    inputs = torch.tensor(data['input'], dtype=torch.float32, device=device)
    action_np = data['action']
    if action_np.ndim == 1 and np.issubdtype(action_np.dtype, np.integer):
        action = torch.tensor(action_np, dtype=torch.long, device=device)
        targets = F.one_hot(action, num_classes=int(action_np.max()) + 1).float()
    else:
        action = torch.tensor(action_np, dtype=torch.float32, device=device)
        targets = action
    leadlag = torch.tensor(leadlag_np, dtype=torch.long, device=device)
    climbdive = torch.tensor(climbdive_np, dtype=torch.long, device=device)
    covariates = torch.tensor(data['covariates'], dtype=torch.float32, device=device)
    placebo = torch.tensor(placebo_np, dtype=torch.long, device=device) if placebo_np is not None else None

    model = create_model(cfg, inputs.shape[1], targets.shape[1], num_concepts, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    lr_decay = cfg.train.get('lr_decay')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay) if lr_decay else None

    alignment = 'causal_mediation' in cfg.train.alignment_losses
    extra_slots = num_concepts - len(slot_to_axis)
    placebo_align_fix = bool(cfg.train.get('placebo_align_fix', False)) and extra_slots > 0
    if placebo_align_fix:
        slot_to_label = list(slot_to_axis) + [len(domain.axis_sizes) + k for k in range(extra_slots)]
    else:
        slot_to_label = list(slot_to_axis) + [0] * extra_slots
    alignment_placebo = placebo if placebo_align_fix else None
    flow = None

    batch_size = cfg.train.batch_size
    for epoch in range(1, cfg.train.epochs + 1):
        if alignment and flow_refit_epoch(epoch, cfg.train.warmup_epochs, cfg.train.flow.refresh_every):
            flow = train_flow(model, train_idx, inputs, covariates, leadlag, climbdive, alignment_placebo,
                              cfg.train.flow, batch_size, slot_to_label, domain.axis_sizes, device)

        model.train()
        shuffled = train_idx[np.random.permutation(len(train_idx))]
        for start in tqdm(range(0, len(shuffled), batch_size), desc=f"Epoch {epoch}/{cfg.train.epochs}"):
            batch = shuffled[start:start + batch_size]
            optimizer.zero_grad()
            outputs = model(inputs[batch], return_details=True)
            concepts = zero_removed_concept(outputs[1], ablation)
            mse, cls = supervised_losses(outputs, targets[batch], leadlag[batch], climbdive[batch], cfg.model.name,
                                         slot_to_axis, removed_slot)
            loss = cfg.train.get('mse_weight', 1.0) * mse + cfg.train.get('cls_weight', 1.0) * cls

            if flow is not None:
                labels = alignment_labels(leadlag[batch], climbdive[batch],
                                          None if alignment_placebo is None else alignment_placebo[batch],
                                          slot_to_axis, num_concepts)
                align = causal_alignment_loss(
                    concepts, labels, covariates[batch], targets[batch],
                    head=model.forward_from_concepts, flow=flow,
                    lambda_necessary=cfg.train.lambda_necessary, lambda_sufficient=cfg.train.lambda_sufficient,
                    lambda_independent=cfg.train.lambda_independent, lambda_presence=cfg.train.lambda_presence,
                    presence_beta=cfg.train.presence_beta, num_donors=cfg.train.mc_samples,
                    checkpoint=cfg.train.get('align_checkpoint', True), strict=placebo_align_fix,
                    removed_slot=removed_slot)
                loss = loss + align * cfg.train.align_weight

            if torch.isnan(loss) or torch.isinf(loss):
                if any(torch.isnan(p).any() for p in model.parameters()):
                    break
                optimizer.zero_grad()
                continue
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), run_dir / "model_final.pt")
    model.eval()
    with torch.no_grad():
        outputs = model(inputs[test_idx], return_details=True)
    np.savez_compressed(
        run_dir / "test_predictions.npz", indices=np.array(test_idx), a_true=action[test_idx].cpu().numpy(),
        a_pred=outputs[0].cpu().numpy(), concepts=outputs[1].cpu().numpy(),
        leadlag=leadlag[test_idx].cpu().numpy(), climbdive=climbdive[test_idx].cpu().numpy())
