import re
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_, get_parameters_num
from utils.custom_logging import PyMARLLogger


class FiLMAgent(nn.Module):
    def __init__(self, input_shape, args: SimpleNamespace):
        super(FiLMAgent, self).__init__()
        self.args = args
        self.unit_types, self.unit_type_slice = self._initialize_unit_type_slice()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.FiLM1 = FiLMLayer(self.unit_types, args.rnn_hidden_dim)
        self.FiLM2 = FiLMLayer(self.unit_types, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

        PyMARLLogger("main").get_child_logger("FiLMAgent").info(f"FiLMAgent Size: {get_parameters_num(self.parameters())}")

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        x = self.FiLM1(x, torch.argmax(inputs[:, self.unit_type_slice], dim=1).detach())
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            hh = self.layer_norm(hh)

        q = self.fc2(hh)
        q = self.FiLM2(q, torch.argmax(inputs[:, self.unit_type_slice], dim=1).detach())

        return q.view(b, a, -1), hh.view(b, a, -1)

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
        super(FiLMLayer, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, hidden_dim)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        Perform feature modulation based on category input.

        Args:
            features (torch.Tensor): Tensor of shape (batch_size, hidden_dim) containing features to be modulated.
            category (torch.Tensor): Tensor of shape (batch_size,) containing category indices.

        Returns:
            torch.Tensor: Modulated features of shape (batch_size, hidden_dim).
        """
        gamma = self.gamma(self.category_embedding(category))
        beta = self.beta(self.category_embedding(category))

        modulated_features = gamma * features + beta
        return modulated_features
