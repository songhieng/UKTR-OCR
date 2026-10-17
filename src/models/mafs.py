"""Modality-Aware Adaptive Feature Selector (MAFS), Stage 2.

Paper (arXiv:2603.00702): after the visual encoder, a router maps the average-pooled encoder
output z to probabilities r over n modality sources; n adapters (projection dim p=128) are applied
to both the 2D features G (Transformer path) and the height-pooled G1d (CTC path); outputs are the
r-weighted sums. The paper trains the router only through the recognition loss.

Choices made here (assumptions, see docs/STAGE2_MAFS.md):
* adapters are residual bottlenecks x + up(gelu(down(LN(x)))) with `up` zero-initialised, so a freshly
  attached MAFS is an exact identity and Stage-2 training starts from the Stage-1 model;
* the same adapter weights are used on G and G1d;
* z is the mean of G1d over valid (unpadded) columns;
* the router logits are returned so a supervised routing loss can be added (route_loss_weight > 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, d: int, p: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, p)
        self.up = nn.Linear(p, d)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.drop(torch.nn.functional.gelu(self.down(self.norm(x)))))


class MAFS(nn.Module):
    def __init__(self, d: int, n_sources: int = 5, adapter_dim: int = 128, router_hidden: int = 256,
                 dropout: float = 0.1, modalities=None):
        super().__init__()
        self.n_sources, self.modalities = n_sources, list(modalities or [])
        self.router = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, router_hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(router_hidden, n_sources))
        self.adapters = nn.ModuleList(Adapter(d, adapter_dim, dropout) for _ in range(n_sources))

    def forward(self, G, G1d, col_mask):
        valid = (~col_mask).to(G1d.dtype)[..., None]                     # B,W',1
        z = (G1d * valid).sum(1) / valid.sum(1).clamp(min=1)              # B,d
        logits = self.router(z)
        r = logits.float().softmax(-1).to(G.dtype)                        # B,n
        H = torch.einsum("bn,nbhwd->bhwd", r, torch.stack([a(G) for a in self.adapters]))
        U1d = torch.einsum("bn,nbwd->bwd", r, torch.stack([a(G1d) for a in self.adapters]))
        return H, U1d, {"route_logits": logits}
