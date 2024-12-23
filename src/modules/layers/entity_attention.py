import math
from types import SimpleNamespace

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils.rms_norm import RMSNorm


class EntityAttnLayer(nn.Module):
    """
    Attention Layer for entity scheme multi-agent systems..
    Args:
        args: args initialized in pymarl.
        embed_dim: size of embedding feature.
        num_heads: number of attention heads.
    """
    def __init__(
            self,
            args: SimpleNamespace,
            embed_dim: int,
            num_heads: int
    ):
        super(EntityAttnLayer, self).__init__()
        self.args = args

        # Attn arguments.
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # networks
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        for name, module in self.named_modules():
            module.full_name = name

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, heads={self.num_heads}"

    def forward(
            self,
            entities: torch.Tensor,

            agent_mask: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            entities: Entity representations.
                [batch size, time size, n_agents, n_entities, embedding dimension]

            agent_mask: Which agents/entities are not available. Zero out their outputs tow
                prevent gradients from flowing back. Shape of 2nd dim determines
                whether to compute queries for all entities or just agents.
                batch size, # of agents or entities]

        Return:
            tuple(attn, attn_weights):
            - attn: attentioned entities state., [batch size, n_agents, embedding dimension]
            - attn_weights: attention weights from agent to entities. [batch size, n_agents, embedding dimension]
        """
        batch_size, time_size, n_agents, n_entities, embed_dim = entities.shape

        # calculate q, k, v.
        q = self.q_proj(entities[..., 0, :].unsqueeze(-2))  # batch * time * agents * self(1) * embed_dim
        k = self.k_proj(entities)  # batch * time * agents * n_entities * embed_dim
        v = self.v_proj(entities)  # batch * time * agents * n_entities * embed_dim

        q = q.reshape(*q.shape[:-1], self.num_heads, -1)  # batch * time * agents * self(1) * heads * head_dim
        q = q * self.scaling
        k = k.reshape(*k.shape[:-1], self.num_heads, -1)  # batch * time * agents * n_entities * heads * head_dim
        v = v.reshape(*v.shape[:-1], self.num_heads, -1)  # batch * time * agents * n_entities * heads * head_dim

        q = q.transpose(-2, -3)  # batch * time * agents * heads * self(1)  * head_dim
        k = k.transpose(-2, -3)  # batch * time * agents * heads * n_entities * head_dim
        v = v.transpose(-2, -3)  # batch * time * agents * heads * n_entities * head_dim

        # calculate attention weights.
        # why not batch * time * agents * heads * n_entities  * head_dim
        attn_weights = q @ k.transpose(-2, -1)  # batch * time * agents * heads * 1  * head_dim

        # attention change -> if refil use agent mask
        if agent_mask is not None and self.args.name == "refil":
            agent_mask_repeat = agent_mask.repeat_interleave(self.num_heads, dim=0)
            attn_weights = attn_weights.masked_fill(agent_mask_repeat[:, :, :n_entities], -float('Inf'))

        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32
        ).type_as(attn_weights)

        attn = attn_weights @ v     # batch * time * agents * heads * 1  * head_dim
        attn = attn.transpose(-2, -3).reshape(*attn.shape[:-3], self.num_heads * self.head_dim)

        attn = self.out_proj(attn)  # batch * agents * embedding_dim

        return attn

    @staticmethod
    def _lambda_inti_fn(depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)
