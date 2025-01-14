import re
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.th_utils import get_parameters_num
from utils.custom_logging import PyMARLLogger


class FiLMAgent(nn.Module):
    def __init__(self, input_shape, args: SimpleNamespace):
        super(FiLMAgent, self).__init__()
        self.args = args
        self.device = getattr(args, "device", torch.device("cpu"))
        self.unit_types, self.unit_type_slice = self._initialize_unit_type_slice()

        self.obs_encoding = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.q_projection = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.n_actions),
            nn.LayerNorm(args.n_actions),
        )

        self.FiLM_layer = FiLMLayer(self.unit_types, args.n_actions)

        PyMARLLogger("main").get_child_logger("FiLMAgent").info(f"FiLMAgent Size: {get_parameters_num(self.parameters())}")

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.rnn_hidden_dim, device=self.device)

    def forward(self, inputs, hidden_state):
        batch_size, n_agents, input_dim = inputs.size()
        inputs = inputs.view(-1, input_dim)

        x = self.obs_encoding(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        q = self.q_projection(hh)

        q = self.FiLM_layer(q, torch.argmax(inputs[:, self.unit_type_slice], dim=1).detach())

        return q.view(batch_size, n_agents, -1), hh.view(batch_size, n_agents, -1)

    def _initialize_unit_type_slice(self) -> tuple[int, slice]:
        """Calculate the slice of the unit type feature in the observation."""
        obs_feature_names = self.args.env_info["obs_feature_names"]

        pattern = r"^own_unit_type"
        unit_type_slice = [
            index
            for index, feature_name in enumerate(obs_feature_names)
            if re.match(pattern, feature_name)
        ]

        return len(unit_type_slice), slice(unit_type_slice[0], unit_type_slice[-1] + 1, 1)


class FiLMLayer(nn.Module):
    """
    A network module for feature-wise linear modulation (FiLM).
    """
    def __init__(self, num_categories, hidden_dim):
        """
        Args:
            num_categories (int): Number of distinct categories.
            hidden_dim (int): Dimension of the features to be modulated.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, hidden_dim)
        self.gamma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, features: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        Perform feature modulation based on category input.

        Args:
            features (torch.Tensor): Tensor of shape (batch_size, hidden_dim) containing features to be modulated.
            category (torch.Tensor): Tensor of shape (batch_size,) containing category indices.

        Returns:
            torch.Tensor: Modulated features of shape (batch_size, hidden_dim).
        """
        return self.out_proj(
            self.gamma(self.embedding(category)) * features + self.beta(self.embedding(category))
        )
