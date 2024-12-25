from types import SimpleNamespace

import torch as th
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import EntityAttentionLayer
from utils.rms_norm import RMSNorm
from utils.custom_logging import PyMARLLogger
from .agent import Agent


class ImagineEntityAttnRNNAgentICM(Agent):
    def __init__(self, input_shape, args: SimpleNamespace):
        super(EntityAttentionLayer, self).__init__(input_shape, args)
        self.args = args
        self.n_agents: int = getattr(args, "n_agents")
        self.n_enemies: int = getattr(args, "env_info")["n_enemies"]
        self.n_actions: int = getattr(args, "n_actions")
        self.n_heads: int = getattr(args, "agent_attn_heads")
        self.hidden_dim: int = getattr(args, "agent_hidden_dim")
        self.attn_dim: int = getattr(args, "agent_attn_dim")    # Dimension of full attention layer
        self.pooling_type: str = getattr(args, "pooling_type", None)

        self.n_entities: int = self.n_agents + self.n_enemies
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim, self.last_action_dim, self.agent_id_dim = input_shape

        if self.attn_dim % self.n_heads != 0:
            PyMARLLogger("main").get_child_logger(f"{self.__class__.__name__}").Fatal(
                f"Attention dimension must be divisible by number of heads. Current values: dim: {self.attn_dim}, heads: {self.n_heads}"
            )
            raise ValueError("Attention dimension must be divisible by number of heads.")

        self.head_dim = self.attn_dim // self.n_heads   # Dimension of each attention head

        # Embedding layers
        self.own_embed = nn.Linear(self.own_feats_dim, self.hidden_dim)
        self.enemy_embed = nn.Linear(self.enemy_feats_dim, self.hidden_dim)
        self.ally_embed = nn.Linear(self.ally_feats_dim, self.hidden_dim)
        self.agent_id_embed = None
        self.last_action_embed = None

        if self.last_action_dim > 0:
            self.agent_id_embed = torch.nn.Embedding(self.n_agents, self.hidden_dim)

        if self.agent_id_dim > 0:
            self.last_action_embed = torch.nn.Embedding(self.n_actions, self.hidden_dim)

        # Encoding layers
        self.encoding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim),
            nn.LeakyReLU(),
        )   # hidden_dim -> attn_dim

        self.attn = EntityAttentionLayer(args, self.attn_dim, self.n_heads)
        self.feedforward = nn.Linear(self.attn_dim, self.attn_dim)
        self.norm = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True)

        # Output layers
        self.rnn_proj = nn.Linear(self.attn_dim, self.hidden_dim)   # attn_dim -> hidden_dim
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.decoding = nn.Linear(self.hidden_dim, args.n_actions)  # h -> q

    def init_hidden(self):
        # make hidden states on same device as model
        return self.own_embed.weight.new(1, self.args.agent_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ret_attn_logits=None, msg=None, ret_attn_weights=False):
        # Head
        own_feats, ally_feats, enemy_feats, last_actions, agent_id = inputs
        batch_size, time_size, _, _ = own_feats.shape

        # Feature embedding. (own_feats_dim, enemy_feats_dim, ally_feats_dim, ...) -> hidden_dim(entity_dim)
        own_embedding = self.own_embed(own_feats)
        ally_embedding = self.ally_embed(ally_feats)
        enemy_embedding = self.enemy_embed(enemy_feats)

        # TODO: pymarl3 use sum to concreate three own embeddings. Maybe a learnable weight is better.
        if self.agent_id_embed is not None:
            agent_id_embedding = self.agent_id_embed(agent_id)
            own_embedding = own_embedding + agent_id_embedding

        if self.last_action_embed is not None:
            last_action_embedding = self.last_action_embed(last_actions)
            own_embedding = own_embedding + last_action_embedding

        entities = th.cat([own_embedding.unsqueeze(-2), ally_embedding, enemy_embedding], dim=-2)

        # Encoding  hidden_dim -> attn_dim

        # A single transformer encoder.
        # TODO: Test multiple structure of attention.
        x = self.encoding(entities)  # TODO: Maybe not useful because all information has already been embedded.
        x = self.norm(x[..., 0, :] + self.attn(x))
        x = self.norm(x + self.feedforward(x))

        # TODO: After the first entity attention layer, the rest should be self attention layer. Not implemented.

        # Output.   attn_dim -> n_actions
        # TODO: RNN might not be advanced. Transformer decoder seems to work here.
        x = self.rnn_proj(x)     # attn_dim -> hidden_dim
        h = hidden_state.reshape(-1, self.hidden_dim)
        hs = []
        for t in range(time_size):
            curr_x = x[:, t].reshape(-1, self.hidden_dim)
            h = self.rnn(curr_x, h)
            hs.append(h.reshape(batch_size, self.n_agents, self.hidden_dim))
        hs = torch.stack(hs, dim=1)
        q = self.decoding(hs)

        return q, hs
