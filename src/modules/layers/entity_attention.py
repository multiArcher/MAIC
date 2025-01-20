import torch
import torch.nn as nn

from utils.custom_logging import PyMARLLogger


class EntityAttnLayer(nn.Module):
    """Attention Layer for entity scheme multi-agent systems.

    Args:
        embed_dim: size of embedding feature.
        num_heads: number of attention heads.
        own_feats_index: index of own features in entity representation.
        keep_dim: whether to keep the attention dimension or not.
        device: device to use.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            own_feats_index: int = 0,
            keep_dim: bool = False,
            device: torch.device = None,
    ):
        default_factory = {"device": device}
        super(EntityAttnLayer, self).__init__()

        # Attn arguments.
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.keep_dim = keep_dim
        self.own_feats_slice = slice(own_feats_index, own_feats_index+1)

        # Ensure attention dimension is divisible by the number of heads
        if self.embed_dim % self.num_heads != 0:
            PyMARLLogger("main").get_child_logger(f"{self.__class__.__name__}").Fatal(
                f"Attention dimension must be divisible by number of heads. Current values: dim: {self.embed_dim}, heads: {self.num_heads}"
            )
            raise ValueError("Attention dimension must be divisible by number of heads.")

        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # networks
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False, **default_factory)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False, **default_factory)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False, **default_factory)
        self.score_function = nn.functional.scaled_dot_product_attention
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **default_factory)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, heads={self.num_heads}"

    def forward(
            self,
            entities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            entities (torch.Tensor): Entity representations.
                [batch size, time size, n_agents, n_entities, embedding dimension]

        Return:
            attn (torch.Tensor): attend entities state., [batch size, n_agents, embedding dimension]
        """
        # calculate q, k, v.
        q = self.q_proj(entities[..., self.own_feats_slice, :])  # batch * time * agents * self(1) * embed_dim
        k = self.k_proj(entities)  # batch * time * agents * n_entities * embed_dim
        v = self.v_proj(entities)  # batch * time * agents * n_entities * embed_dim

        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)  # batch * time * agents * self(1) * heads * head_dim
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim)  # batch * time * agents * n_entities * heads * head_dim
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim)  # batch * time * agents * n_entities * heads * head_dim

        q = q.transpose(-2, -3)  # batch * time * agents * heads * self(1)  * head_dim
        k = k.transpose(-2, -3)  # batch * time * agents * heads * n_entities * head_dim
        v = v.transpose(-2, -3)  # batch * time * agents * heads * n_entities * head_dim

        # calculate attention weights.
        attn = self.score_function(q, k, v)  # batch * time * agents * heads * 1  * head_dim
        # Equivalent implementation, use torch.nn.functional.scaled_dot_product_attention instead.
        # attn_weights = q @ k.transpose(-2, -1)  # batch * time * agents * heads * 1  * n_entities
        # attn_weights = attn_weights * self.scaling
        # attn_weights = functional.softmax(attn_weights, dim=-1)
        # attn = attn_weights @ v     # batch * time * agents * heads * 1  * head_dim

        attn = attn.reshape(
            *attn.shape[:-3],
            *(1,) * self.keep_dim,
            self.num_heads * self.head_dim
        )  # batch * time * agents * (1 if keep_dim)  * (heads * head_dim)

        attn = self.out_proj(attn)  # batch * agents * (1 if keep_dim) * embedding_dim

        return attn
