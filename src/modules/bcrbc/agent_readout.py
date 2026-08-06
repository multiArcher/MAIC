import torch.nn as nn


class AgentReadout(nn.Module):
    def __init__(self, model_hidden_dim, agent_output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(model_hidden_dim),
            nn.Linear(model_hidden_dim, agent_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(agent_output_dim, agent_output_dim),
        )

    def forward(self, transformer_outputs, agent_slice):
        return self.projection(transformer_outputs[..., agent_slice, :])
