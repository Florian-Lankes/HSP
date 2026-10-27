# HSP

A minimal PyTorch variational autoencoder baseline.

## Setup

```bash
uv sync
```

## CPU smoke check

```bash
uv run python -c 'import torch; from hsp.vae import VAE; x = torch.randn(4, 2); model = VAE(input_dim=2, latent_dim=1); reconstruction, mu, logvar = model(x); loss = model.loss(reconstruction, x, mu, logvar); assert loss.ndim == 0 and torch.isfinite(loss); loss.backward()'
```

Implemented code lives in `src/hsp/vae.py`. The root-level `data/`, `docs/`, `experiments/`, `notebooks/`, and `scripts/` directories are intentionally empty placeholders for future work.
