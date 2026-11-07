"""Training of the beta-TCVAE and iVAE models."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ccw_net.data import load_data
from ccw_net.models.disentanglement_baselines import IVAE, BetaTCVAE

EVAL_BATCH = 1024


def encode_u(leadlag, climbdive, num_axes, num_classes):
    """iVAE auxiliary variable: one-hot concept labels, zero for unlabelled rows."""
    if num_axes >= 2:
        u = torch.cat([F.one_hot(leadlag.clamp(min=0).long(), num_classes=2).float(),
                       F.one_hot(climbdive.clamp(min=0).long(), num_classes=2).float()], dim=-1)
        return u * (leadlag >= 0).float().unsqueeze(-1) * (climbdive >= 0).float().unsqueeze(-1)
    one_hot = F.one_hot(leadlag.clamp(min=0).long(), num_classes=num_classes).float()
    return one_hot * (leadlag >= 0).float().unsqueeze(-1)


def evaluate_split(model, z, a, ll, cd, is_ivae, num_axes, u_dim, device):
    """Sampled actions and concepts of every row, their action MSE and concept accuracy from slot lengths."""
    model.eval()
    a_preds, concepts = [], []
    with torch.no_grad():
        for i in range(0, len(z), EVAL_BATCH):
            zb = z[i:i + EVAL_BATCH].to(device)
            if is_ivae:
                a_hat, v = model(zb, u=encode_u(ll[i:i + EVAL_BATCH], cd[i:i + EVAL_BATCH], num_axes, u_dim).to(device))
            else:
                a_hat, v = model(zb)
            a_preds.append(a_hat.cpu())
            concepts.append(v.cpu())
    a_pred, v_concepts = torch.cat(a_preds), torch.cat(concepts)
    lengths = v_concepts.norm(dim=-1)
    if num_axes >= 2:
        acc_ll = ((lengths[:, 1] > lengths[:, 0]).long() == ll.cpu()).float().mean().item()
        acc_cd = ((lengths[:, 3] > lengths[:, 2]).long() == cd.cpu()).float().mean().item()
    else:
        acc_ll = (lengths.argmax(dim=-1) == ll.cpu()).float().mean().item()
        acc_cd = float("nan")
    metrics = {"mse": F.mse_loss(a_pred, a.detach().cpu()).item(), "acc_ll": acc_ll, "acc_cd": acc_cd}
    return metrics, a_pred, v_concepts


def train(cfg, run_dir, device):
    """Train the beta-TCVAE or iVAE config `cfg`; writes model_final.pt, test_predictions.npz and final_metrics.json."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    data = load_data(Path(cfg.system.data_root) / cfg.dataset.data_subdir, cfg)
    num_axes = data["domain"].num_axes
    leadlag = torch.as_tensor(data["leadlag"]).long().to(device)
    climbdive = torch.as_tensor(data["climbdive"]).long().to(device)
    actions = torch.as_tensor(data["action"]).to(device)
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    if not torch.is_floating_point(actions):
        actions = actions.float()
    inputs = torch.as_tensor(data["input"]).float().to(device)

    is_ivae = cfg.model.name == "ivae"
    dims = dict(z_dim=inputs.shape[1], concept_dim=cfg.model.concept_dim, hidden_dim=cfg.model.hidden_dim,
                a_dim=actions.shape[1], num_concepts=cfg.model.num_concepts)
    if is_ivae:
        u_dim = 4 if num_axes >= 2 else max(int(leadlag.max().item()) + 1, 1)
        model = IVAE(u_dim=u_dim, **dims).to(device)
    else:
        u_dim = 0
        model = BetaTCVAE(beta=cfg.model.get("beta", 6.0), **dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    train_idx = torch.as_tensor(data["train_labeled_idx"], dtype=torch.long, device=device)
    test_idx = torch.as_tensor(data["test_idx"], dtype=torch.long, device=device)
    batch_size = cfg.train.batch_size
    num_steps = (len(train_idx) + batch_size - 1) // batch_size
    rng = np.random.RandomState(cfg.seed)
    for epoch in range(cfg.train.epochs):
        model.train()
        perm = rng.permutation(len(train_idx))
        for step in tqdm(range(num_steps), desc=f"Epoch {epoch + 1}/{cfg.train.epochs}"):
            idx = train_idx[perm[step * batch_size:(step + 1) * batch_size]]
            if is_ivae:
                loss = model.loss(inputs[idx], encode_u(leadlag[idx], climbdive[idx], num_axes, u_dim).to(device),
                                  actions[idx])
            else:
                loss = model.loss(inputs[idx], actions[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

    torch.save({"model_state_dict": model.state_dict()}, run_dir / "model_final.pt")
    metrics, a_pred, concepts = evaluate_split(model, inputs[test_idx], actions[test_idx], leadlag[test_idx],
                                               climbdive[test_idx], is_ivae, num_axes, u_dim, device)
    np.savez_compressed(
        run_dir / "test_predictions.npz", indices=test_idx.cpu().numpy(), a_true=actions[test_idx].cpu().numpy(),
        a_pred=a_pred.numpy(), concepts=concepts.numpy(), leadlag=leadlag[test_idx].cpu().numpy(),
        climbdive=climbdive[test_idx].cpu().numpy())
    (run_dir / "final_metrics.json").write_text(json.dumps({"test": metrics}, indent=2))
