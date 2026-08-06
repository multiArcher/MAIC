import torch
import torch.nn as nn


class BCRBCQHead(nn.Module):
    """Map agent-token outputs to local action values."""

    def __init__(self, agent_output_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(agent_output_dim),
            nn.Linear(agent_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, agent_outputs: torch.Tensor) -> torch.Tensor:
        return self.net(agent_outputs)
