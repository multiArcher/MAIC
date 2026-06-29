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
        block_group_ids: Optional[torch.Tensor] = None,
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
            block_group_ids: Optional per-token agent-group id over the space axis
                (length S). When set, the dynamics spatial mask is block-diagonal per
                agent (CTDE): cross-agent info flows only through message tokens.
            time_block_every: Apply temporal attention every N layers (architecture
                constant, not a per-run hyperparameter).
            attn_softclamp_value: Softclamp value for attention logits (architecture
                constant).
            feed_forward_expansion_factor: FFN expansion factor (architecture constant).
        """
        super().__init__()

        # Per-head dimension is fully determined by the model dim and head count;
        # a non-divisible config is a setup error and must fail loudly.
        assert model_hidden_dim % num_attention_heads == 0, (
            f"model_hidden_dim {model_hidden_dim} not divisible by "
            f"num_attention_heads {num_attention_heads}"
        )
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
            block_group_ids=block_group_ids,
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
        assert tokens.shape[-1] == self.model_hidden_dim, (
            f"Input dim {tokens.shape[-1]} != model dim {self.model_hidden_dim}"
        )

        output = self.transformer(tokens)
        return output
