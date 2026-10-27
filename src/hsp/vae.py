"""A minimal fully-connected variational autoencoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VAE(nn.Module):
    """A VAE for fixed-width vector inputs."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the latent distribution mean and log-variance."""
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample a latent vector using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct an input from a latent vector."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return reconstruction, latent mean, and latent log-variance."""
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @staticmethod
    def loss(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """Return mean reconstruction error plus KL divergence."""
        reconstruction_loss = F.mse_loss(reconstruction, target, reduction="mean")
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
        return reconstruction_loss + kl_divergence
