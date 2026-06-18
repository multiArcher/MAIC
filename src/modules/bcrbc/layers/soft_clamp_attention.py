from typing import overload, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .functions import softclamp
from .multi_head_rms_norm import MultiHeadRMSNorm
from .rotary_embedding import apply_rotary_pos_emb


class SoftClampAttention(nn.Module):
    def __init__(
        self,
        dim_attn: int,
        dim_head: int = 64,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        pre_rmsnorm: bool = True,
        q_norm: bool = False,
        k_norm: bool = True,
        softclamp_value: float = 50.,
        device = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.softclamp_value = softclamp_value

        self.heads = heads
        self.kv_heads = heads if kv_heads is None else kv_heads
        self.dim_head = dim_head
        if self.heads % self.kv_heads != 0:
            raise ValueError(f"Heads must be divisible by kv_heads.")
        self.groups = self.heads // self.kv_heads

        self.norm = nn.RMSNorm(dim_attn, device=device) if pre_rmsnorm else nn.Identity()
        dim_q_inner = dim_head * heads
        dim_kv_inner = dim_head * self.kv_heads

        self.to_q = nn.Linear(dim_attn, dim_q_inner, bias=False, device=device)
        self.to_k = nn.Linear(dim_attn, dim_kv_inner, bias=False, device=device)
        self.to_v = nn.Linear(dim_attn, dim_kv_inner, bias=False, device=device)
        self.to_out = nn.Linear(dim_q_inner, dim_attn, bias=False, device=device)

        self.q_norm = MultiHeadRMSNorm(dim_head, heads, device=device) if q_norm else nn.Identity()
        self.k_norm = MultiHeadRMSNorm(dim_head, self.kv_heads, device=device) if k_norm else nn.Identity()

    @overload
    def forward(
        self, x: Tensor, mask=None, rotary_pos_emb=None, kv_cache=None, *, return_cache: Literal[False] = False
    ) -> Tensor: ...

    @overload
    def forward(
        self, x: Tensor, mask=None, rotary_pos_emb=None, kv_cache=None, *, return_cache: Literal[True]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]: ...

    def forward(
        self,
        x: Tensor,
        mask=None,
        rotary_pos_emb=None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        *,
        return_cache: bool = False
    ):
        x = self.norm(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        *batch_dims, seq_len, _ = q.shape
        q = q.reshape(*batch_dims, seq_len, self.heads, self.dim_head).transpose(-2, -3)
        k = k.reshape(*batch_dims, seq_len, self.kv_heads, self.dim_head).transpose(-2, -3)
        v = v.reshape(*batch_dims, seq_len, self.kv_heads, self.dim_head).transpose(-2, -3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(q, *rotary_pos_emb)
            k = apply_rotary_pos_emb(k, *rotary_pos_emb)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=-2)
            v = torch.cat((v_cache, v), dim=-2)

        if self.groups > 1:
            q = q.reshape(*batch_dims, self.kv_heads, self.groups, seq_len, self.dim_head)
            k = k.unsqueeze(-3)
            v = v.unsqueeze(-3)

        similarity = (q * self.scale) @ k.transpose(-1, -2)
        similarity = softclamp(similarity, self.softclamp_value)

        if mask is not None:
            mask_value = -torch.finfo(similarity.dtype).max
            similarity = similarity.masked_fill(~mask, mask_value)

        attn = similarity.softmax(dim=-1) @ v

        if self.groups > 1:
            attn = attn.reshape(*batch_dims, self.heads, seq_len, self.dim_head)
            k = k.squeeze(-3)
            v = v.squeeze(-3)

        attn = attn.transpose(-2, -3)
        attn = attn.reshape(*batch_dims, seq_len, self.heads * self.dim_head)
        attn = self.to_out(attn)

        if return_cache:
            return attn, (k, v)
        return attn
