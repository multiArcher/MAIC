from types import SimpleNamespace

import torch.nn
import torch.nn as nn

from .agent import Agent
from modules.layers import EntityAttnLayer
from modules.layers.rms_norm import RMSNorm


class EntityAttnRNNAgent(Agent):
    def __init__(self, input_scheme, args: SimpleNamespace):
        super().__init__(input_scheme, args)

        self.args = args
        self.device: torch.device = torch.device(getattr(args, "device", "cpu"))
        self.n_heads: int = getattr(args, "agent_attn_heads")   # Attention heads.
        self.hidden_dim: int = getattr(args, "agent_hidden_dim")    # Dimension of embedding andRNN.
        self.attn_dim: int = getattr(args, "agent_attn_dim")    # Dimension of full attention layer
        self.gru_layers: int = getattr(args, "agent_gru_layers")    # Number of GRU layers.

        # Embedding layers: scheme -> hidden_dim
        self.embedding_layers = nn.ModuleList()
        self.own_feature_index: int = 0
        self_feature_count = 0
        for feat_name, feat_shape in input_scheme[0].items():
            self.embedding_layers.append(
                nn.Linear(feat_shape[1], self.hidden_dim, bias=False, device=self.device)
            )

            if feat_name == "own_feats_size":
                # prepare own_feats_slice for attention.
                self.own_feature_index = self_feature_count
            self_feature_count += feat_shape[0] # count the number of other features before self features.


        # Embedding layers: 1 -> hidden_dim
        for feat_name, feat_shape in input_scheme[1].items():
            self.embedding_layers.append(
                nn.Embedding(feat_shape[1], self.hidden_dim, device=self.device)
            )

        # Encoding layers: hidden_dim -> attn_dim
        self.encoding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim, device=self.device),
            nn.LeakyReLU(inplace=True),
        )

        self.attn = EntityAttnLayer(self.attn_dim, self.n_heads, self.own_feature_index, device=self.device)
        self.norm1 = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True).to(self.device)
        self.feedforward = nn.Linear(self.attn_dim, self.attn_dim, bias=False, device=self.device)
        self.norm2 = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True).to(self.device)

        # Output layers: attn_dim -> hidden_dim -> n_actions q
        self.rnn_proj = nn.Linear(self.attn_dim, self.hidden_dim, device=self.device)
        # TODO: RNN might not be advanced. Transformer encoder seems to work here.
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=self.gru_layers, batch_first=True, device=self.device)
        self.decoding = nn.Linear(self.hidden_dim, args.n_actions, device=self.device)

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.gru_layers, self.args.agent_hidden_dim, device=self.device)
        # return self.own_embed.weight.new(1, self.args.agent_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # Embedding: scheme -> hidden_dim
        entities = torch.cat(
            [layer(feature_input) for feature_input, layer in zip(inputs, self.embedding_layers)],
            dim=-2
        )   # batch * time * n_agents * n_entities * hidden_dim

        # Encoding: hidden_dim -> attn_dim
        # A single transformer encoder.
        # TODO: Test multiple structure of attention.
        entities = self.encoding(entities)
        attn = self.norm1(
            entities[..., self.own_feature_index, :] + self.attn(entities)
        )
        attn = self.norm2(attn + self.feedforward(attn))    # batch * time * n_agents * attn_dim

        # TODO: After the first entity attention layer, the rest should be self attention layer. Not implemented.

        batch_size, time_size, n_agents, _ = attn.shape

        # Output:  attn_dim -> n_actions
        # attn_dim -> hidden_dim
        x = self.rnn_proj(attn)    # batch * time * n_agents * hidden_dim

        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.hidden_dim)  # b * t * n * d -> b * n * t * d -> (b * n) * t * d
        h = hidden_state.reshape(self.gru_layers, batch_size * n_agents, self.hidden_dim)   # layers * (batch * n_agents) * hidden_dim

        x, h = self.rnn(x, h)   # GRU forward.

        x = x.reshape(batch_size, n_agents, time_size, self.hidden_dim).transpose(1, 2)     # (b * n) * t * d -> b * n * t * d -> b * t * n * d
        h = h.reshape(self.gru_layers, batch_size, n_agents, self.hidden_dim)

        q = self.decoding(x)

        return q, h
