import re
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.custom_logging import PyMARLLogger


class FiLMAgent(nn.Module):
    """A network module for a FiLM agent.

    FiLM agent is a modified version of the QMIX agent that uses FiLM layer
    to modulate the features of the agent's observation or q or other inputs.

    The FiLM layer is a feature-wise linear modulation layer that takes a category
    as input and modulates the features of the input based on the category.

    This structure targets to learn a differential policy/q for different kinds of units.

    Args:
        input_shape (int): Shape of the input observation.
        args (SimpleNamespace): Arguments for the agent.
    """

    def __init__(self, input_shape, args: SimpleNamespace, device=None, dtype=None):
        """Initialize the FiLMAgent."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.args = args
        self.device = getattr(args, "device", torch.device("cpu"))
        self.unit_types, self.unit_type_slice = self._initialize_unit_type_slice()

        self.obs_encoding = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
        )
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim, **factory_kwargs)

        self.FiLM_layer = FiLMLayer(self.unit_types, args.rnn_hidden_dim, **factory_kwargs)

        self.q_projection = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.n_actions, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Linear(args.n_actions, args.n_actions, **factory_kwargs),
        )

        PyMARLLogger("main").get_child_logger("FiLMAgent").info(f"FiLMAgent Size: {self.size}")

    @property
    def size(self):
        return str(sum(p.numel() for p in self.parameters()) / 1000) + 'K'

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.args.rnn_hidden_dim, device=self.device)

    def forward(self, inputs, hidden_state):
        batch_size, n_agents, input_dim = inputs.size()
        inputs = inputs.view(-1, input_dim)
        unit_types = self._unit_type_from_obs(inputs)

        x = self.obs_encoding(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        q_in = self.FiLM_layer(hh, unit_types)

        q = self.q_projection(q_in)

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

    def _unit_type_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract the unit type feature from the observation."""
        return torch.argmax(obs[:, self.unit_type_slice], dim=1).detach()


class FiLMLayer(nn.Module):
    """A network module for feature-wise linear modulation (FiLM).

    Args:
    num_categories (int): Number of distinct categories.
    hidden_dim (int): Dimension of the features to be modulated.
    """
    def __init__(self, num_categories, hidden_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(num_categories, hidden_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.FiLM_modulation1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim, **factory_kwargs),
        )

        self.FiLM_modulation2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, **factory_kwargs),
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, **factory_kwargs),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim, **factory_kwargs),
        )

    def forward(self, x: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        Perform feature modulation based on category input.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, hidden_dim) containing features to be modulated.
            category (torch.Tensor): Tensor of shape (batch_size,) containing category indices.

        Returns:
            torch.Tensor: Modulated features of shape (batch_size, hidden_dim).
        """
        # First Modulation
        alpha, beta1, gamma1 = self.FiLM_modulation1(self.embedding(category)).chunk(3, dim=-1)
        x = x + alpha * self.feed_forward(self.norm1(x) * (1 + gamma1) + beta1)

        # Output Modulation
        beta2, gamma2 = self.FiLM_modulation2(self.embedding(category)).chunk(2, dim=-1)
        x = self.norm2(x) * (1 + gamma2) + beta2

        return x
