import math
from typing import Optional

import torch
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings of size `dim` for integer timesteps `t`."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * -(math.log(10000.0) / max(1, half - 1))
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    def __init__(self, emb_dim: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.proj(t_emb)
