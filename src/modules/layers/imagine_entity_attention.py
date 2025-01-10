import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

from utils.custom_logging import PyMARLLogger


class ImagineEntityAttnLayer(nn.Module):
    """
    Attention Layer for entity scheme multi-agent systems.
    Args:
        args: args initialized in pymarl.
        embed_dim: size of embedding feature.
        num_heads: number of attention heads.
    """
    def __init__(
            self,
            args: SimpleNamespace,
            embed_dim: int,
            num_heads: int,
            device: Optional[torch.device] = None
    ):
        super(ImagineEntityAttnLayer, self).__init__()
        self.args = args

        # Attn arguments.
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Ensure attention dimension is divisible by the number of heads
        if self.embed_dim % self.num_heads != 0:
            PyMARLLogger("main").get_child_logger(f"{self.__class__.__name__}").Fatal(
                f"Attention dimension must be divisible by number of heads. Current values: dim: {self.embed_dim}, heads: {self.num_heads}"
            )
            raise ValueError("Attention dimension must be divisible by number of heads.")

        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # networks
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, heads={self.num_heads}"

    def forward(
            self, entities: torch.Tensor, pre_mask=None, post_mask=None, ret_attn_logits=None
    ) -> torch.Tensor:
        """
        Args:
            entities (torch.Tensor): Entity representations.
                [batch size, time size, n_agents, n_entities, embedding dimension]

        Return:
            attn (torch.Tensor): attend entities state., [batch size, n_agents, embedding dimension]
        """
        # calculate q, k, v.
        q = self.q_proj(entities[..., 0, :].unsqueeze(-2))  # batch * time * agents * self(1) * embed_dim
        k = self.k_proj(entities)  # batch * time * agents * n_entities * embed_dim
        v = self.v_proj(entities)  # batch * time * agents * n_entities * embed_dim

        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)  # batch * time * agents * self(1) * heads * head_dim
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim)  # batch * time * agents * n_entities * heads * head_dim
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim)  # batch * time * agents * n_entities * heads * head_dim

        q = q.transpose(-2, -3)  # batch * time * agents * heads * self(1)  * head_dim
        k = k.transpose(-2, -3)  # batch * time * agents * heads * n_entities * head_dim
        v = v.transpose(-2, -3)  # batch * time * agents * heads * n_entities * head_dim

        # calculate attention weights.
        attn_weights = q @ k.transpose(-2, -1)  # batch * time * agents * heads * 1  * n_entities

        batch_size, time_size, n_agents, n_entities, _ = entities.shape

        attn_logits = attn_weights * self.scaling
        # attention masked -> if refil use agent mask
        if pre_mask is not None:
            agent_mask_repeat = pre_mask.unsqueeze(-2).unsqueeze(-2).repeat(1, 1, 1, self.num_heads, 1, 1).to(attn_weights.device)
            # attn_logits = attn_weights.masked_fill(agent_mask_repeat[:, :, :n_agents, :, : ,:].bool(), -float('Inf'))
            attn_logits = attn_weights.masked_fill(agent_mask_repeat[:, :, :n_agents, :, :, :].bool(), -1e8)
        attn_weights = functional.softmax(attn_logits, dim=-1)
        attn = attn_weights @ v     # batch * time * agents * heads * 1  * head_dim
        attn = attn.transpose(-2, -3).reshape(*attn.shape[:-3], self.num_heads * self.head_dim)

        attn = self.out_proj(attn)  # batch * agents * embedding_dim

        if post_mask is not None:
            attn_outs = attn.masked_fill(post_mask.unsqueeze(2).bool(), 0)
        if ret_attn_logits is not None:
            # bs * n_heads, nq, ne
            attn_logits = attn_logits.reshape(batch_size, self.num_heads,
                                              n_agents, n_entities)
            if ret_attn_logits == 'max':
                attn_logits = attn_logits.max(dim=1)[0]
            elif ret_attn_logits == 'mean':
                attn_logits = attn_logits.mean(dim=1)
            elif ret_attn_logits == 'norm':
                attn_logits = attn_logits.mean(dim=1)
            return attn_outs, attn_logits

        return attn