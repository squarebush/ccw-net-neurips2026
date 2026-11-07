"""Causal alignment losses: necessity, sufficiency, independence and presence."""
import torch
import torch.utils.checkpoint


def sample_donors(flow, donor_labels, cond, num_donors):
    """num_donors flow draws of every slot given the donor labels and covariates, shaped [S, B, J, D]."""
    batch, slots = donor_labels.shape
    with torch.no_grad():
        donors = flow.sample(torch.cat([donor_labels.float(), cond], dim=-1), num_samples=num_donors)
    return donors.view(batch, num_donors, slots, -1).permute(1, 0, 2, 3).contiguous()


def partial_effects(concepts, concept_labels, cond, head, flow, num_donors, checkpoint):
    """Total effect against the all-zero input, each slot's effect alone (PE+) and with a donor swapped in (PE-)."""
    batch, slots, dim = concepts.shape
    donors = sample_donors(flow, 1 - concept_labels, cond, num_donors)
    if torch.isnan(donors).any():
        raise RuntimeError("Donor sampling produced NaN.")

    y_self = head(concepts)
    action_dim = y_self.shape[-1]
    y_zero = head(torch.zeros_like(concepts)).detach()
    total_effect = y_self - y_zero

    def run_head(stacked):
        if checkpoint and stacked.requires_grad:
            return torch.utils.checkpoint.checkpoint(head, stacked, use_reentrant=False)
        return head(stacked)

    alone = torch.zeros(slots, batch, slots, dim, device=concepts.device, dtype=concepts.dtype)
    for j in range(slots):
        alone[j, :, j, :] = concepts[:, j, :]
    pe_plus = run_head(alone.reshape(slots * batch, slots, dim)).reshape(slots, batch, action_dim) - y_zero.unsqueeze(0)

    swapped = concepts.unsqueeze(0).expand(num_donors, -1, -1, -1).unsqueeze(0).expand(
        slots, num_donors, batch, slots, dim).clone()
    for j in range(slots):
        swapped[j, :, :, j, :] = donors[:, :, j, :]
    pe_minus = (run_head(swapped.reshape(slots * num_donors * batch, slots, dim))
                .reshape(slots, num_donors, batch, action_dim) - y_self.detach().unsqueeze(0).unsqueeze(0))
    return total_effect, y_zero, pe_plus, pe_minus


def necessary_loss(pe_minus, concept_labels):
    """Squared effect of swapping a donor into each slot whose concept is absent."""
    forbidden = (concept_labels == 0).float().transpose(0, 1)
    return ((pe_minus ** 2).sum(dim=-1) * forbidden.unsqueeze(1)).sum(dim=0).mean()


def independent_loss(total_effect, pe_plus):
    """Squared gap between the total effect and the sum of the single-slot effects."""
    return ((total_effect - pe_plus.sum(dim=0)) ** 2).sum(dim=-1).mean()


def sufficient_loss(y_zero, pe_plus, concept_labels, actions):
    """Squared error of the action rebuilt from the effects of the present concepts."""
    active = concept_labels.float().transpose(0, 1)
    rebuilt = y_zero + (pe_plus * active.unsqueeze(-1)).sum(dim=0)
    return ((rebuilt - actions) ** 2).sum(dim=-1).mean()


def presence_loss(total_effect, pe_plus, concept_labels, beta):
    """Squared shortfall of each present concept's effect below beta times the total effect."""
    active = concept_labels.float().transpose(0, 1)
    threshold = beta * torch.norm(total_effect, dim=-1).unsqueeze(0)
    shortfall = torch.clamp(threshold - torch.norm(pe_plus, dim=-1), min=0.0)
    return ((shortfall ** 2) * active).sum(dim=0).mean()


def causal_alignment_loss(concepts, concept_labels, cond, actions, *, head, flow, lambda_necessary, lambda_sufficient,
                          lambda_independent, lambda_presence, presence_beta, num_donors, checkpoint=True,
                          strict=False, removed_slot=None):
    """Weighted sum of the four alignment losses over the fully labelled rows; zero when fewer than four."""
    device = concepts.device
    labeled = (concept_labels >= 0).all(dim=-1)
    if labeled.sum() < 4:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if not labeled.all():
        concepts, concept_labels = concepts[labeled], concept_labels[labeled]
        cond, actions = cond[labeled], actions[labeled]

    try:
        total_effect, y_zero, pe_plus, pe_minus = partial_effects(
            concepts, concept_labels, cond, head, flow, num_donors, checkpoint)
    except RuntimeError as exc:
        if strict and "out of memory" not in str(exc) and "produced NaN" not in str(exc):
            raise
        return torch.tensor(0.0, device=device, requires_grad=True)

    zero = torch.tensor(0.0, device=device)
    necessary = necessary_loss(pe_minus, concept_labels) if lambda_necessary > 0 else zero
    sufficient = sufficient_loss(y_zero, pe_plus, concept_labels, actions) if lambda_sufficient > 0 else zero
    independent = independent_loss(total_effect, pe_plus) if lambda_independent > 0 else zero
    present = concept_labels if removed_slot is None else concept_labels.index_fill(
        1, torch.tensor([removed_slot], device=device), 0.0)
    presence = presence_loss(total_effect, pe_plus, present, presence_beta) if lambda_presence > 0 else zero
    return (lambda_necessary * necessary + lambda_sufficient * sufficient + lambda_independent * independent
            + lambda_presence * presence)
