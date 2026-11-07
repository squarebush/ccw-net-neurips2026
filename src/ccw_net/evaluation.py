"""Task error and concept accuracy of a trained model on a split."""
import torch
import torch.nn.functional as F

from ccw_net.ablations import zero_removed_concept


def axis_slots(slot_to_axis, axis):
    """Slot indices of one concept group."""
    return [j for j, a in enumerate(slot_to_axis) if a == axis]


def squash_concept_accuracy(concepts, leadlag, climbdive, placebo, num_concepts, slot_to_axis):
    """CCW-Net concept accuracy: the longest slot per concept group, or slot norm above 0.5 per attribute."""
    lengths = torch.norm(concepts, dim=-1)
    zero = torch.tensor(0.0, device=concepts.device)
    if leadlag.dim() == 2:
        acc = ((lengths > 0.5).long() == leadlag.long()).float().mean(dim=0).mean()
        return {'acc_ll': acc, 'acc_cd': zero, 'acc_placebo': zero}
    accs = []
    for axis, labels in enumerate((leadlag, climbdive)[:max(slot_to_axis) + 1]):
        pred = torch.argmax(lengths[:, torch.tensor(axis_slots(slot_to_axis, axis), device=concepts.device)], dim=-1)
        mask = labels >= 0
        accs.append((pred[mask] == labels[mask]).float().mean() if mask.any() else zero)
    acc_placebo = zero
    if num_concepts == 5 and placebo is not None and (placebo >= 0).any():
        mask = placebo >= 0
        acc_placebo = ((lengths[:, 4] > 0.5).long()[mask] == placebo[mask]).float().mean()
    return {'acc_ll': accs[0], 'acc_cd': accs[1] if len(accs) > 1 else zero, 'acc_placebo': acc_placebo}


def baseline_concept_accuracy(concepts, concept_scores, leadlag, climbdive, architecture, slot_to_axis):
    """CBM: concept probability above 0.5 per attribute; CW-Net: the largest signed score per concept group."""
    zero = torch.tensor(0.0, device=concepts.device)
    if architecture == "cbm":
        return {'acc_ll': ((concept_scores > 0.5).long() == leadlag.long()).float().mean(), 'acc_cd': zero}
    scores = concepts.squeeze(-1) if concepts.shape[-1] == 1 else concept_scores
    accs = []
    for axis, labels in enumerate((leadlag, climbdive)):
        cols = axis_slots(slot_to_axis, axis)
        mask = labels >= 0
        accs.append((torch.argmax(scores[:, cols][mask], dim=-1) == labels[mask]).float().mean()
                    if mask.any() and cols else zero)
    return {'acc_ll': accs[0], 'acc_cd': accs[1]}


def eval_on_split(model, split_idx, input_tensor, action, leadlag, climbdive, covariates, placebo, reference_bank,
                  cfg, num_concepts, ablation_config, device, verbose=False, slot_to_axis=None):
    """Task loss (cross-entropy for classes, else MSE) and concept accuracy on the first train.eval_samples rows."""
    model.eval()
    rows = split_idx[:min(cfg.train.eval_samples, len(split_idx))]
    with torch.no_grad():
        a_b, ll_b, cd_b = action[rows], leadlag[rows], climbdive[rows]
        pl_b = placebo[rows] if placebo is not None else None
        outputs = model(input_tensor[rows], return_details=True)
        concepts = zero_removed_concept(outputs[1], ablation_config)
        if a_b.dim() == 1 and a_b.dtype in (torch.long, torch.int32, torch.int64):
            mse = F.cross_entropy(outputs[0], a_b).item()
        else:
            mse = F.mse_loss(outputs[0], a_b).item()
        if cfg.model.name == "squash_attention":
            acc = squash_concept_accuracy(concepts, ll_b, cd_b, pl_b, num_concepts, slot_to_axis)
        else:
            acc = baseline_concept_accuracy(concepts, outputs[2], ll_b, cd_b, cfg.model.name, slot_to_axis)
    return {'mse': mse} | {name: value.item() for name, value in acc.items()}
