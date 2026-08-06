"""BlockCausalTransformer wrapper for axial space-time transformer."""

from typing import Optional

import torch
import torch.nn as nn

from .layers import BlockCasualTransformer


class BlockCausalTransformer(nn.Module):
    """Causal space-time block transformer wrapper.

    Wraps the axial space-time transformer with a clean interface compatible
    with dynamics_core.py.
    """

    def __init__(
        self,
        model_hidden_dim: int,
        num_transformer_layers: int,
        num_attention_heads: int,
        dropout: float,
        agent_slice: Optional[slice],
        context_window: Optional[int] = None,
        time_block_every: int = 4,
        attn_softclamp_value: float = 50.,
        feed_forward_expansion_factor: int = 4,
    ):
        """Initialize the transformer wrapper.

        Args:
            model_hidden_dim: Model dimension.
            num_transformer_layers: Number of transformer blocks.
            num_attention_heads: Number of attention heads.
            dropout: Dropout rate (unused in axial transformer).
            agent_slice: Slice for causal-confusion masking (the query-token span).
            context_window: Optional temporal context window. When set (> 0), the causal
                time attention is banded to the previous `context_window` steps. None / 0
                => unbounded causal attention (legacy).
            time_block_every: Apply temporal attention every N layers (architecture
                constant, not a per-run hyperparameter).
            attn_softclamp_value: Softclamp value for attention logits (architecture
                constant).
            feed_forward_expansion_factor: FFN expansion factor (architecture constant).
        """
        super().__init__()

        attention_head_dim = model_hidden_dim // num_attention_heads

        self.model_hidden_dim = model_hidden_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.transformer = BlockCasualTransformer(
            dim=model_hidden_dim,
            depth=num_transformer_layers,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            q_norm=False,
            k_norm=True,
            attn_softclamp_value=attn_softclamp_value,
            time_block_every=time_block_every,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            num_special_tokens=0,
            is_decoder=False,
            is_dynamics=(agent_slice is not None),
            agent_slice=agent_slice,
            context_window=context_window,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        kv_cache=None,
        use_kv_cache: bool = False,
        rope_offset=None,
    ):
        """Forward pass through the axial space-time transformer.

        Args:
            tokens: [B, T, N, A, D], where N is agent and A is the local-token
                axis. Spatial attention operates over A independently for every
                (B,T,N); temporal attention operates over T independently for
                every (B,N,A).
            kv_cache: Optional list of cached (k, v) tuples for the time layers
                (incremental decoding across time). When supplied with
                ``use_kv_cache=True``, ``tokens`` is the *new* time slice only and
                each time layer attends it against the cached past.
            use_kv_cache: When True, return the updated per-time-layer KV caches
                alongside the output so the caller can persist them.

        Returns:
            output: Tensor of the same shape [B, T, N, A, D].
            If ``use_kv_cache`` is True, also returns the updated kv cache list.
        """
        if use_kv_cache:
            return self.transformer(tokens, kv_cache=kv_cache, use_kv_cache=True,
                                    rope_offset=rope_offset)
        return self.transformer(tokens)
