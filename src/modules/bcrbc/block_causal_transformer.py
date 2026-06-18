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
        d_model: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
        kv_heads: Optional[int] = None,
        dim_head: Optional[int] = None,
        time_block_every: int = 4,
        attn_softclamp_value: float = 50.,
        feed_forward_expansion_factor: int = 4,
        agent_slice: Optional[slice] = None,
    ):
        """Initialize the transformer wrapper.

        Args:
            d_model: Model dimension.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            dropout: Dropout rate (unused in axial transformer).
            kv_heads: Number of key/value heads for GQA. Defaults to heads.
            dim_head: Dimension per head. Defaults to d_model // heads.
            time_block_every: Apply temporal attention every N layers.
            attn_softclamp_value: Softclamp value for attention logits.
            feed_forward_expansion_factor: FFN expansion factor.
            agent_slice: Optional slice for causal-confusion masking.
        """
        super().__init__()

        if dim_head is None:
            assert d_model % heads == 0, f"d_model must be divisible by heads"
            dim_head = d_model // heads

        self.d_model = d_model
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head

        self.transformer = BlockCasualTransformer(
            dim=d_model,
            depth=depth,
            heads=heads,
            kv_heads=kv_heads,
            dim_head=dim_head,
            q_norm=False,
            k_norm=True,
            attn_softclamp_value=attn_softclamp_value,
            time_block_every=time_block_every,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            num_special_tokens=0,
            is_decoder=False,
            is_dynamics=(agent_slice is not None),
            agent_slice=agent_slice,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the axial space-time transformer.

        Args:
            tokens: Input tensor of shape [..., T, S, D] where the leading dims
                are an arbitrary batch prefix (typically just B), T is time, S is
                the joint agent/modality token-sequence the spatial attention
                attends over, and D is the model dim. The underlying axial core
                operates on ``*batch_dims`` so no flattening of the batch prefix
                is required.

        Returns:
            output: Tensor of the same shape [..., T, S, D].
        """
        assert tokens.ndim >= 3, f"Expected at least [T, S, D], got {tokens.shape}"
        assert tokens.shape[-1] == self.d_model, (
            f"Input dim {tokens.shape[-1]} != model dim {self.d_model}"
        )

        output = self.transformer(tokens)
        return output
