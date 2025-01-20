import re
from types import SimpleNamespace

import torch.nn
import torch.nn as nn

from .agent import Agent
from modules.layers import EntityAttnLayer
from modules.layers.rms_norm import RMSNorm
from utils.custom_logging import PyMARLLogger


class EntityFiLMAgent(Agent):
    def __init__(self, input_scheme, args: SimpleNamespace):
        super().__init__(input_scheme, args)

        self.args = args
        self.device: torch.device = torch.device(getattr(args, "device", "cpu"))
        self.n_heads: int = getattr(args, "agent_attn_heads")   # Attention heads.
        self.hidden_dim: int = getattr(args, "agent_hidden_dim")    # Dimension of embedding andRNN.
        self.attn_dim: int = getattr(args, "agent_attn_dim")    # Dimension of full attention layer
        self.gru_layers: int = getattr(args, "agent_gru_layers")    # Number of GRU layers.

        self.number_of_unit_types, self.unit_type_slice = self._initialize_unit_type_slice()

        # Entity_encoding.
        # Embedding layers: scheme -> hidden_dim
        self.embedding_layers = nn.ModuleList()
        self.own_feature_index_in_entities: int = 0  # Index in encoded entities features.
        self.own_feature_index_in_inputs = 0   # Index in agent inputs.
        self_feature_count = 0
        input_part_count = 0

        for feat_name, feat_shape in input_scheme[0].items():
            self.embedding_layers.append(
                nn.Linear(feat_shape[1], self.hidden_dim, bias=False, device=self.device)
            )

            # prepare own_feats_slice for entity attention.
            if feat_name == "own_feats_size":
                self.own_feature_index_in_entities = self_feature_count
                self.own_feature_index_in_inputs = input_part_count
            self_feature_count += feat_shape[0] # count the number of other features before self features.
            input_part_count += 1   # count the number of input parts.

        # Embedding layers: 1 -> hidden_dim
        for feat_name, feat_shape in input_scheme[1].items():
            self.embedding_layers.append(
                nn.Embedding(feat_shape[1], self.hidden_dim, device=self.device)
            )

        # Encoding layers: hidden_dim -> attn_dim
        self.encoding = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.attn_dim, device=self.device),
        )

        # Attention layers: attn_dim -> attn_dim
        # TODO: Extract entity attention block to a separate module.
        self.attn = EntityAttnLayer(self.attn_dim, self.n_heads, self.own_feature_index_in_entities, device=self.device)
        self.norm1 = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True).to(self.device)
        self.feedforward = nn.Linear(self.attn_dim, self.attn_dim, bias=False, device=self.device)
        self.norm2 = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True).to(self.device)

        # Q projection layers: attn_dim -> hidden_dim -> n_actions q
        self.rnn_projection = nn.Linear(self.attn_dim, self.hidden_dim, device=self.device)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=self.gru_layers, batch_first=True, device=self.device)
        self.q_projection = nn.Linear(self.hidden_dim, args.n_actions, device=self.device)

        # modulation layers
        self.modulation = nn.Sequential(
            nn.Linear(self.number_of_unit_types, self.hidden_dim, device=self.device),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.n_actions * (args.n_actions + 1), device=self.device),
        )
        # merging.
        self.merge_factor = nn.Linear(self.attn_dim, 3, device=self.device)


        PyMARLLogger("main").get_child_logger(f"Agent").info(f"{self.__class__.__name__} Size: {self.size}")


    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.gru_layers, self.args.agent_hidden_dim, device=self.device)
        # return self.own_embed.weight.new(1, self.args.agent_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        unit_types = inputs[self.own_feature_index_in_inputs][..., self.unit_type_slice]   # batch * time * n_agents * n_unit_types

        # Embedding: scheme -> hidden_dim
        entities = torch.cat(
            [layer(feature_input) for feature_input, layer in zip(inputs, self.embedding_layers)],
            dim=-2
        )   # batch * time * n_agents * n_entities * hidden_dim

        # Encoding: hidden_dim -> attn_dim
        # A single transformer encoder.
        entities = self.encoding(entities)
        # TODO: After the first entity attention layer, the rest should be self attention layer. Not implemented.
        attn = self.norm1(
            entities[..., self.own_feature_index_in_entities, :] + self.attn(entities)
        )
        attn = self.norm2(attn + self.feedforward(attn))    # batch * time * n_agents * attn_dim


        batch_size, time_size, n_agents, _ = attn.shape

        # Output:  attn_dim -> n_actions
        # attn_dim -> hidden_dim
        x = self.rnn_projection(attn)    # batch * time * n_agents * hidden_dim

        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.hidden_dim)  # b * t * n * d -> b * n * t * d -> (b * n) * t * d
        h = hidden_state.reshape(self.gru_layers, batch_size * n_agents, self.hidden_dim)   # layers * (batch * n_agents) * hidden_dim

        x, h = self.rnn(x, h)   # GRU forward.

        x = x.reshape(batch_size, n_agents, time_size, self.hidden_dim).transpose(1, 2)     # (b * n) * t * d -> b * n * t * d -> b * t * n * d
        h = h.reshape(self.gru_layers, batch_size, n_agents, self.hidden_dim)

        q = self.q_projection(x)

        # q modulation.
        modulation_weights, modulation_bias =(
            self.modulation(unit_types)
            .reshape(*q.shape[:-1], self.args.n_actions + 1, self.args.n_actions)
            .split([self.args.n_actions, 1], dim=-2)
        )
        q_modulated = (q.unsqueeze(-2) @ modulation_weights + modulation_bias).squeeze(-2)

        # Merge
        alpha, beta, gamma = self.merge_factor(entities.sum(dim=-2)).chunk(3, dim=-1)
        q = q + alpha * (gamma * q_modulated + beta)

        return q, h

    def _initialize_unit_type_slice(self) -> tuple[int, slice]:
        """Calculate the slice of the unit type feature in the observation."""
        obs_feature_names = self.args.env_info["obs_feature_names"]

        # Slice self features.
        pattern = re.compile("^own_")
        own_feature_slice = [
            feature_name
            for index, feature_name in enumerate(obs_feature_names)
            if re.match(pattern, feature_name)
        ]

        # Slice unit type feature.
        pattern = re.compile("^own_unit_type")
        own_type_slice = [
            index
            for index, feature_name in enumerate(own_feature_slice)
            if re.match(pattern, feature_name)
        ]

        return len(own_type_slice), slice(own_type_slice[0], own_type_slice[-1] + 1, 1)
