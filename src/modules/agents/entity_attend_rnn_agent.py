from types import SimpleNamespace

import torch.nn
import torch.nn as nn

from modules.layers import EntityAttnLayer
from utils.rms_norm import RMSNorm
from utils.custom_logging import PyMARLLogger
from .agent import Agent


class EntityAttnRNNAgent(Agent):
    def __init__(self, input_scheme, args: SimpleNamespace):
        super().__init__(input_scheme, args)

        self.args = args
        self.device = args.device
        self.n_agents: int = getattr(args, "n_agents")
        self.n_heads: int = getattr(args, "agent_attn_heads")   # Attention heads.
        self.hidden_dim: int = getattr(args, "agent_hidden_dim")    # Dimension of embedding andRNN.
        self.attn_dim: int = getattr(args, "agent_attn_dim")    # Dimension of full attention layer
        self.gru_layers: int = getattr(args, "agent_gru_layers")    # Number of GRU layers.

        # Ensure attention dimension is divisible by the number of heads
        if self.attn_dim % self.n_heads != 0:
            PyMARLLogger("main").get_child_logger(f"{self.__class__.__name__}").Fatal(
                f"Attention dimension must be divisible by number of heads. Current values: dim: {self.attn_dim}, heads: {self.n_heads}"
            )
            raise ValueError("Attention dimension must be divisible by number of heads.")

        # Embedding layers: scheme -> hidden_dim
        self.embedding_layers = nn.ModuleList()
        for feat_name, feat_shape in input_scheme[0].items():
            self.embedding_layers.append(
                nn.Linear(feat_shape[1], self.hidden_dim, bias=False, )
            )

        # Embedding layers: 1 -> hidden_dim
        for feat_name, feat_shape in input_scheme[1].items():
            self.embedding_layers.append(
                nn.Embedding(feat_shape[1], self.hidden_dim, )
            )

        # Encoding layers: hidden_dim -> attn_dim
        self.encoding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim, ),
            nn.LeakyReLU(inplace=True),
        )

        self.attn = EntityAttnLayer(args, self.attn_dim, self.n_heads)
        self.feedforward = nn.Linear(self.attn_dim, self.attn_dim, bias=False)
        self.norm = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True)

        # Output layers: attn_dim -> hidden_dim -> n_actions q
        self.rnn_proj = nn.Linear(self.attn_dim, self.hidden_dim, )
        # TODO: RNN might not be advanced. Transformer decoder seems to work here.
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=self.gru_layers, batch_first=True)
        self.decoding = nn.Linear(self.hidden_dim, args.n_actions, )

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.gru_layers, self.args.agent_hidden_dim, device=self.device)
        # return self.own_embed.weight.new(1, self.args.agent_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # Head
        x = [
            layer(feature_input) for feature_input, layer in zip(inputs, self.embedding_layers)
        ]

        x = torch.cat(x, dim=-2)

        # Encoding: hidden_dim -> attn_dim
        # A single transformer encoder.
        # TODO: Test multiple structure of attention.
        x = self.encoding(x)  # TODO: Maybe not useful because all information has already been embedded.
        x = self.norm(x[..., 0, :] + self.attn(x))
        x = self.norm(x + self.feedforward(x))


        # TODO: After the first entity attention layer, the rest should be self attention layer. Not implemented.

        # Output:  attn_dim -> n_actions
        batch_size, time_size, n_agents, _ = x.shape
        x = self.rnn_proj(x)     # attn_dim -> hidden_dim

        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.hidden_dim)  # b * t * n * d -> b * n * t * d -> (b * n) * t * d
        h = hidden_state.reshape(self.gru_layers, batch_size * n_agents, self.hidden_dim)

        x, h = self.rnn(x, h)
        x = x.reshape(batch_size, n_agents, time_size, self.hidden_dim).transpose(1, 2)
        h = h.reshape(self.gru_layers, batch_size, n_agents, self.hidden_dim)

        q = self.decoding(x)

        return q, h
