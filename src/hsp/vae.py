"""A minimal fully-connected variational autoencoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VAE(nn.Module):
    """A VAE for fixed-width vector inputs."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32, num_layers: int = 1) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder
        encoder_layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        decoder_layers = []
        current_dim = latent_dim

        for _ in range(num_layers):
            decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

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
    def loss(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return mean reconstruction error plus KL divergence."""
        reconstruction_loss = F.binary_cross_entropy_with_logits(
            reconstruction,
            target,
            reduction="none"
        ).sum(dim=1).mean()

        # KL Divergence
        kl_divergence = -0.5 * torch.sum(
            1 + logvar - mu.square() - logvar.exp(),
            dim=1
        ).mean()

        # Total Loss
        total_loss = reconstruction_loss + kl_divergence

        return total_loss, reconstruction_loss, kl_divergence