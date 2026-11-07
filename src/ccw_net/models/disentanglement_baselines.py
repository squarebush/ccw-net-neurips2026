"""Beta-TCVAE and iVAE baselines with a linear tanh action head on their sampled latent slots."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden_dim, out_dim, n_hidden=2):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


def _log_normal(x, mu, logvar):
    return -0.5 * (logvar + math.log(2 * math.pi) + (x - mu).pow(2) / logvar.exp())


class BetaTCVAE(nn.Module):
    """Beta-TCVAE whose sampled latent is J concept slots of size D; it samples in eval mode too."""

    def __init__(self, z_dim, concept_dim=16, hidden_dim=256, a_dim=2, num_concepts=4, beta=6.0):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.beta = beta
        flat_dim = num_concepts * concept_dim
        self.encoder = _mlp(z_dim, hidden_dim, 2 * flat_dim)
        self.decoder = _mlp(flat_dim, hidden_dim, z_dim)
        self.W = nn.Parameter(torch.randn(num_concepts, concept_dim, a_dim) * 0.02)
        self.action_bias = nn.Parameter(torch.zeros(a_dim))

    def forward_from_concepts(self, v):
        """Actions from concept slots [B, J, D]."""
        return torch.tanh(torch.einsum('bjd,jda->ba', v, self.W) + self.action_bias)

    def forward(self, z, return_details=False):
        """(actions, sampled slots), plus posterior parameters, flat sample and reconstruction when return_details."""
        mu, logvar = self.encoder(z).chunk(2, dim=-1)
        latent = _reparameterize(mu, logvar)
        v = latent.view(-1, self.num_concepts, self.concept_dim)
        a_hat = self.forward_from_concepts(v)
        if return_details:
            return a_hat, v, mu, logvar, latent, self.decoder(latent)
        return a_hat, v

    def loss(self, z, action_target):
        """Reconstruction, mutual information, beta-weighted total correlation, dimension-wise KL and action MSE."""
        a_hat, _, mu, logvar, latent, z_hat = self(z, return_details=True)
        batch = z.shape[0]
        recon = F.mse_loss(z_hat, z, reduction='sum') / batch
        log_q_zx = _log_normal(latent, mu, logvar).sum(dim=-1)
        log_q_z_xall = _log_normal(latent.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))
        log_q_z = torch.logsumexp(log_q_z_xall.sum(dim=-1), dim=-1) - math.log(batch)
        log_q_zd_prod = (torch.logsumexp(log_q_z_xall, dim=1) - math.log(batch)).sum(dim=-1)
        log_p_z = _log_normal(latent, torch.zeros_like(latent), torch.zeros_like(latent)).sum(dim=-1)
        mi = (log_q_zx - log_q_z).mean()
        tc = (log_q_z - log_q_zd_prod).mean()
        dim_kl = (log_q_zd_prod - log_p_z).mean()
        return recon + mi + self.beta * tc + dim_kl + F.mse_loss(a_hat, action_target)


class IVAE(nn.Module):
    """iVAE with a label-conditioned prior, whose sampled latent is J concept slots of size D."""

    def __init__(self, z_dim, concept_dim=16, hidden_dim=256, a_dim=2, num_concepts=4, u_dim=2):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.u_dim = u_dim
        flat_dim = num_concepts * concept_dim
        self.encoder = _mlp(z_dim + u_dim, hidden_dim, 2 * flat_dim)
        self.decoder = _mlp(flat_dim, hidden_dim, z_dim)
        self.prior_net = _mlp(u_dim, hidden_dim, 2 * flat_dim, n_hidden=1)
        self.W = nn.Parameter(torch.randn(num_concepts, concept_dim, a_dim) * 0.02)
        self.action_bias = nn.Parameter(torch.zeros(a_dim))

    def forward_from_concepts(self, v):
        """Actions from concept slots [B, J, D]."""
        return torch.tanh(torch.einsum('bjd,jda->ba', v, self.W) + self.action_bias)

    def forward(self, z, u=None, return_details=False):
        """(actions, sampled slots) given labels u (zeros by default), plus posterior, prior and reconstruction."""
        if u is None:
            u = torch.zeros(z.shape[0], self.u_dim, device=z.device, dtype=z.dtype)
        mu_q, logvar_q = self.encoder(torch.cat([z, u], dim=-1)).chunk(2, dim=-1)
        latent = _reparameterize(mu_q, logvar_q)
        v = latent.view(-1, self.num_concepts, self.concept_dim)
        a_hat = self.forward_from_concepts(v)
        if return_details:
            mu_p, logvar_p = self.prior_net(u).chunk(2, dim=-1)
            return a_hat, v, mu_q, logvar_q, mu_p, logvar_p, latent, self.decoder(latent)
        return a_hat, v

    def loss(self, z, u, action_target):
        """Reconstruction, KL to the conditional prior and action MSE."""
        a_hat, _, mu_q, logvar_q, mu_p, logvar_p, _, z_hat = self(z, u, return_details=True)
        recon = F.mse_loss(z_hat, z, reduction='sum') / z.shape[0]
        var_p = logvar_p.exp().clamp(min=1e-8)
        kl = 0.5 * ((logvar_p - logvar_q) + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / var_p - 1.0)
        return recon + kl.sum(dim=-1).mean() + F.mse_loss(a_hat, action_target)
