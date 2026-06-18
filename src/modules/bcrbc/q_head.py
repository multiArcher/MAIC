import torch
import torch.nn as nn


class BCRBCQHead(nn.Module):
    """Map compensated beliefs to local action values."""

    def __init__(self, belief_dim: int, n_actions: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or belief_dim
        self.net = nn.Sequential(
            nn.LayerNorm(belief_dim),
            nn.Linear(belief_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, beliefs: torch.Tensor) -> torch.Tensor:
        return self.net(beliefs)
