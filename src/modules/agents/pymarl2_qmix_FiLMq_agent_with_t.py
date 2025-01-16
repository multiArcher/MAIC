import re
from types import SimpleNamespace
from functools import partial

import torch
import torch.nn as nn
import math

from utils.custom_logging import PyMARLLogger


class FiLMTAgent(nn.Module):
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
        self.hidden_dim = args.rnn_hidden_dim
        self.number_of_unit_types, self.unit_type_slice = self._initialize_unit_type_slice()

        # obs encoding part
        self.obs_encoding = nn.Sequential(
            nn.Linear(input_shape + 1, self.hidden_dim, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
        )
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim, **factory_kwargs)

        # out normal q
        self.q_projection = nn.Linear(self.hidden_dim, args.n_actions, **factory_kwargs)

        # modulation.
        self.type_embedding = nn.Embedding(self.number_of_unit_types, self.hidden_dim, **factory_kwargs)
        self.timestep_embedding = TimestepEmbedder(self.hidden_dim, frequency_embedding_size=self.hidden_dim)

        self.modulation = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.n_actions * (args.n_actions + 1), **factory_kwargs),
        )

        # merging.
        self.merge = nn.Linear(self.hidden_dim, 3, **factory_kwargs)

        PyMARLLogger("main").get_child_logger("FiLMAgent").info(f"FiLMAgent Size: {self.size}")

    @property
    def size(self):
        return str(sum(p.numel() for p in self.parameters()) / 1000) + 'K'

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.hidden_dim, device=self.device)

    def forward(self, inputs, hidden_state):
        obs, timestep = inputs.values()
        batch_size, n_agents, input_dim = obs.size()
        obs = torch.cat([obs, timestep], dim=-1)
        input_dim = input_dim + 1

        obs = obs.reshape(-1, input_dim)
        timestep = timestep.reshape(-1, 1)
        unit_types = obs[:, self.unit_type_slice].detach()

        # obs encoding forward.
        x = self.obs_encoding(obs)
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        hh = self.rnn(x, h_in)

        # q forward.
        q = self.q_projection(hh)

        # modulation forward.
        type_embedding = self.type_embedding(torch.argmax(unit_types, dim=-1))
        timestep_embedding = self.timestep_embedding(timestep)
        embedding = timestep_embedding + type_embedding

        modulation_weights, modulation_bias = (
            self.modulation(embedding)
            .reshape(-1, self.args.n_actions + 1, self.args.n_actions)
            .split([self.args.n_actions, 1], dim=-2)
        )
        q_modulated = (q.unsqueeze(-2) @ modulation_weights + modulation_bias).squeeze(-2)

        # merging forward.
        alpha, beta, gamma = self.merge(embedding).chunk(3, dim=-1)
        q = q + alpha * ((1 + gamma) * q_modulated + beta)

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
        # self.embedding = nn.Embedding(num_categories, hidden_dim, **factory_kwargs)
        self.embedding = partial(nn.functional.one_hot, num_classes=num_categories)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.FiLM_modulation = nn.Sequential(
            nn.Linear(num_categories, 3 * hidden_dim + 3, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3 * hidden_dim + 3, 3 * hidden_dim + 3, **factory_kwargs),
            nn.LeakyReLU(inplace=True),
            nn.Linear(3 * hidden_dim + 3, 3 * hidden_dim + 3, **factory_kwargs),
        )
        self.modulation_split = [hidden_dim, hidden_dim, hidden_dim, 1, 1, 1]

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, **factory_kwargs),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim, **factory_kwargs),
        )

        self.out_projection = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

    def forward(self, x: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        """
        Perform feature modulation based on category input.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, hidden_dim) containing features to be modulated.
            category (torch.Tensor): Tensor of shape (batch_size,) containing category indices.

        Returns:
            torch.Tensor: Modulated features of shape (batch_size, hidden_dim).
        """
        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = (
            self.FiLM_modulation(self.embedding(category))
            .split(self.modulation_split, dim=-1)
        )

        # First Modulation
        x = x + alpha1 * self.feed_forward(self.norm(x) * (1 + gamma1) + beta1)

        # Output Modulation
        x = alpha2 * ((1 + gamma2) * x + beta2)

        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=200):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t.float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb