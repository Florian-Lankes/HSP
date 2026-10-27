from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GAN(nn.Module):

    def __init__(self, noise_dim: int, data_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def generate(self, noise: Tensor) -> Tensor:
        return self.generator(noise)

    def discriminate(self, samples: Tensor) -> Tensor:
        return self.discriminator(samples)

    @staticmethod
    def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return real_loss + fake_loss

    @staticmethod
    def generator_loss(fake_logits: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

    def train_step(
        self,
        real_samples: Tensor,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
    ) -> tuple[Tensor, Tensor]:
        noise = torch.randn(real_samples.size(0), self.noise_dim, device=real_samples.device)

        discriminator_optimizer.zero_grad()
        fake_samples = self.generate(noise).detach()
        discriminator_loss = self.discriminator_loss(
            self.discriminate(real_samples), self.discriminate(fake_samples)
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # maybe more discriminator training steps then generator training steps later
        generator_optimizer.zero_grad()
        fake_samples = self.generate(noise)
        generator_loss = self.generator_loss(self.discriminate(fake_samples))
        generator_loss.backward()
        generator_optimizer.step()

        return generator_loss.detach(), discriminator_loss.detach()
