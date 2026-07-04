import math

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    r"""Rotary Position Embedding Module.

    Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing 568 (2024): 127063.
    https://arxiv.org/abs/2104.09864
    """
    # TODO: Implement through torch.complex for clear math representations.
    inv_freq: torch.Tensor   # Buffer for inverse frequencies
    cos_cache: torch.Tensor  # Cache for cosine values
    sin_cache: torch.Tensor  # Cache for sine values

    def __init__(self, dim, theta=10000., max_seq_len = 512, device = None):
        super().__init__()
        self.device = device
        self.theta = theta
        self.max_seq_len = max_seq_len

        # inv_freq = 1 / (theta ** (2i / dim))
        #          = theta ** (-2i / dim)
        #          = e ** (-ln(theta) * (2i / dim))
        exponent = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        inv_freq = torch.exp(-exponent * math.log(theta))

        self.register_buffer("inv_freq", inv_freq)

        self._set_cache(max_seq_len)    # Precompute caches

    @property
    def cache_len(self) -> int:
        return self.cos_cache.shape[0]

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rotary position embeddings for a given sequence length and offset."""
        total_len = seq_len + offset

        if total_len > self.max_seq_len:
            self._set_cache(total_len)  # Recompute caches if needed

        return (
            self.cos_cache[offset:offset + seq_len],
            self.sin_cache[offset:offset + seq_len]
        )

    def _set_cache(self, seq_len: int):
        """Calculate and set the cosine and sine caches up to seq_len."""
        self.max_seq_len = seq_len
        # t: [0, 1, ..., seq_len-1]
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # freqs: (seq_len, dim/2)
        freqs = torch.einsum("i, j -> ij", t, self.inv_freq)    #

        # (seq_len, dim): [θ0, θ1, ..., θ0, θ1, ...]
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def __repr__(self) -> str:
        return f"RotaryPositionEmbedding(dim={self.inv_freq.shape[0]*2}, theta={self.theta})"

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Calculates the rotation of the last dimension by half."""
    # x1, x2, x3, x4 -> -x3, -x4, x1, x2
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(seq: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # t: (..., len, dim)
    # cos, sin: (len, dim)
    return (seq * cos) + (_rotate_half(seq) * sin)
