"""Task and concept losses."""
import torch
import torch.nn.functional as F


def concept_logits(raw_concepts):
    """Per-slot logits: the scalar concept itself, or the norm of a vector concept."""
    return raw_concepts.squeeze(-1) if raw_concepts.shape[2] == 1 else raw_concepts.norm(dim=-1)


def supervised_losses(outputs, targets, leadlag, climbdive, architecture, slot_to_axis, removed_slot=None):
    """Task MSE and concept loss: BCE on squash norms, cross-entropy per concept group for CW-Net, BCE on CBM logits."""
    mse = F.mse_loss(outputs[0], targets)
    concept_probs = outputs[2]
    if leadlag.dim() == 2:
        if architecture == 'cbm':
            return mse, F.binary_cross_entropy_with_logits(concept_logits(outputs[3]), leadlag.float())
        return mse, F.binary_cross_entropy(concept_probs.clamp(1e-6, 1 - 1e-6), leadlag.float())

    cls = torch.tensor(0.0, device=targets.device)
    for axis, labels in enumerate((leadlag, climbdive)[:max(slot_to_axis) + 1]):
        mask = labels >= 0
        if not mask.any():
            continue
        axis_slots = [j for j, a in enumerate(slot_to_axis) if a == axis]
        kept = [(class_idx, j) for class_idx, j in enumerate(axis_slots) if j != removed_slot]
        slot_idx = torch.tensor([j for _, j in kept], device=targets.device)
        if architecture == 'cwnet':
            cls = cls + F.cross_entropy(concept_logits(outputs[3])[mask][:, slot_idx], labels[mask].long())
        else:
            class_ids = torch.tensor([class_idx for class_idx, _ in kept], device=targets.device)
            slot_targets = (labels[mask].unsqueeze(1) == class_ids).float()
            cls = cls + F.binary_cross_entropy(concept_probs[mask][:, slot_idx].clamp(1e-6, 1 - 1e-6), slot_targets)
    return mse, cls
