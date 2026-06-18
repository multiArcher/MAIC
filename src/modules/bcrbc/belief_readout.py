import torch
import torch.nn as nn


class BeliefReadout(nn.Module):
    """Gather per-agent query tokens from joint block-causal latents."""

    def __init__(self, d_model: int, belief_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, belief_dim),
            nn.ReLU(inplace=True),
            nn.Linear(belief_dim, belief_dim),
        )

    def forward(self, latents: torch.Tensor, query_indices: torch.Tensor) -> torch.Tensor:
        # latents [B, T, S, D] -> gather per-agent query tokens [B, T, n, D],
        # then add the explicit per-agent vector axis -> [B, T, n, 1, belief_dim]
        # so downstream heads/losses operate with [b, t, n, 1, *] natively.
        beliefs = latents.index_select(dim=2, index=query_indices)
        beliefs = self.proj(beliefs)
        return beliefs.unsqueeze(-2)
