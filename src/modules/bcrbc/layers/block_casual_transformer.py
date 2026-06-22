from typing import Literal, overload, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .soft_clamp_attention import SoftClampAttention
from .rotary_embedding import RotaryPositionEmbedding
from .siwglu_feed_forward import SwiGLUFeedForward


class AxialTransformerBlock(nn.Module):
    """Attention Block for Axial Transformer."""
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        q_norm: bool = False,
        k_norm: bool = True,
        attn_softclamp_value: float = 50.,
        feed_forward_expansion_factor: int = 4,
        dropout: float = 0.0,  # Currently not used.
        device=None
    ):
        super().__init__()

        self.attention = SoftClampAttention(
            dim_attn=dim,
            dim_head=dim_head,
            heads=heads,
            kv_heads=kv_heads,
            q_norm=q_norm,
            k_norm=k_norm,
            softclamp_value=attn_softclamp_value,
            device=device
        )

        self.feed_forward = SwiGLUFeedForward(
            dim=dim,
            expansion_factor=feed_forward_expansion_factor,
            device=device
        )

    @overload
    def forward(
        self, x: Tensor, mask=None, rotary_pos_emb=None, kv_cache=None, *, return_cache: Literal[False] = False
    ) -> Tensor: ...    # Returns only output

    @overload
    def forward(
        self, x: Tensor, mask=None, rotary_pos_emb=None, kv_cache=None, *, return_cache: Literal[True]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]: ...    # Returns output and cache

    def forward(
        self, x: Tensor, mask=None, rotary_pos_emb=None, kv_cache=None, *, return_cache: bool = False,
    ):
        """
        Args:
            x: Input tensor of shape (..., seq_len, dim).
            mask: Attention mask broadcastable to (..., heads, seq_len, seq_len).
            rotary_pos_emb: Tuple of (cos, sin) tensors for RoPE application.
            kv_cache: Tuple of cached keys and values for inference.
            return_cache: Whether to return updated cache.

        Returns:
            If return_cache is False:
                output: Tensor of shape (..., seq_len, dim).
            If return_cache is True:
                output: Tensor of shape (..., seq_len, dim).
                new_cache: Tuple of updated (k_cache, v_cache).
        """
        new_cache = None

        attention_out = self.attention(
            x,
            mask=mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_cache=kv_cache,
            return_cache=return_cache
        )
        if return_cache:
            attn, new_cache = attention_out
        else:
            attn = attention_out

        x = x + attn
        x = x + self.feed_forward(x)

        if return_cache:
            return x, new_cache
        else:
            return x

class BlockCasualTransformer(nn.Module):
    """Casual Space-Time Transformer Block from Dreamer V4.

    Implements the Causal Space-Time Axial Transformer based on the DreamerV4 architecture.

    This module decomposes the full spatiotemporal attention into two steps:
    1. **Temporal Attention (Causal):** Applies self-attention along the time axis to capture motion dynamics and temporal dependencies.
    2. **Spatial Attention (Bidirectional):** Applies self-attention along the spatial axis to capture intra-frame visual details.

    Reference:
        Hafner et al., "Training agents inside of scalable world models." (2025).
        https://arxiv.org/pdf/2509.24527
    """
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        kv_heads: Optional[int] = None,
        dim_head = 64,
        q_norm = False,
        k_norm = True,
        attn_softclamp_value = 50.,
        time_block_every = 4,
        feed_forward_expansion_factor = 4,
        num_special_tokens = 0,
        is_decoder = False,
        is_dynamics = False,
        agent_slice: Optional[slice] = None,
        dropout = 0.0,
        device=None
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_special_tokens = num_special_tokens
        self.decoding = is_decoder
        self.device = device
        self.is_dynamics = is_dynamics
        self.agent_slice = agent_slice

        self.layers = nn.ModuleList([])
        self.is_time_layer = []

        self.rotary = RotaryPositionEmbedding(dim_head, device=device)

        for i in range(depth):
            is_time = ((i + 1) % time_block_every == 0)  # Every N layers is a Time Layer
            self.is_time_layer.append(is_time)

            block = AxialTransformerBlock(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                q_norm=q_norm,
                k_norm=k_norm,
                dim_head=dim_head,
                attn_softclamp_value=attn_softclamp_value,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                dropout=dropout,
                device=device
            )
            self.layers.append(block)

        self.final_norm = nn.RMSNorm(dim, device=device)

    def _build_space_mask(self, dim: int, device=None) -> Optional[Tensor]:
        """Build the spatial attention mask considering special tokens and dynamics logic.

        During Encoding:
            Latent tokens can attend to all tokens, while patch tokens cannot attend to latent tokens.
            e.g.,   | patch1 | patch2 | latent
            -----------------------------------
            patch1  |  True  |  True  | False |
            patch2  |  True  |  True  | False |
            latent  |  True  |  True  |  True |

        During Decoding:
            Latent tokens can only attend to latent tokens, while patch tokens can attend to all tokens.
            e.g.,   | patch1 | patch2 | latent
            -----------------------------------
            patch1  |  True  |  True  |  True |
            patch2  |  True  |  True  |  True |
            latent  | False  | False  |  True |

        During Dynamics (World Model):
            Agent tokens can attend to all other modalities, but no other modalities can attend back
            to agent tokens. This prevents causal confusion (future predictions cannot be directly
            influenced by the current task). All non-agent modalities fully attend to each other.
            e.g.,   | spatial | action | agent
            -----------------------------------
            spatial |  True   |  True  | False |
            action  |  True   |  True  | False |
            agent   |  True   |  True  |  True |
        """
        if self.is_dynamics:
            # In the dynamics (world-model) mode, ordinary tokens within the same
            # time step attend to each other fully (bidirectional).
            mask = torch.ones((dim, dim), device=device, dtype=torch.bool)

            if self.agent_slice is not None:
                # Follow the paper's section 3.3 rule to prevent causal confusion.
                # Rule 1: no other modality may attend back to the agent tokens
                # (i.e. the columns at agent_slice are set to False).
                mask[:, self.agent_slice] = False

                # Rule 2: agent tokens may attend to themselves and to all other
                # modalities (i.e. the rows at agent_slice are set to True, which
                # also restores agent-token self-attention disabled by rule 1).
                mask[self.agent_slice, :] = True

            return mask

        if self.num_special_tokens == 0:
            return None

        mask = torch.ones((dim, dim), device=device, dtype=torch.bool)
        sep = dim - self.num_special_tokens
        mask[:sep, sep:] = False

        if self.decoding:
            mask = mask.T

        return mask

    @overload
    def forward(self, x: Tensor, kv_cache=None, *, use_kv_cache: Literal[False] = False) -> Tensor: ...    # Returns only output

    @overload
    def forward(
        self, x: Tensor, kv_cache=None, *, use_kv_cache: Literal[True]
    ) -> Tuple[Tensor, list]: ...    # Returns output and list of caches

    def forward(
        self,
        x: Tensor, #TODO：Support source from different inputs.
        kv_cache = None,
        use_kv_cache: bool = False
    ):
        """Forward pass of the Block Casual Transformer.

        Args:
            x: Input tensor of shape (..., time_steps, sequence_len, dim).
            kv_cache: List of cached (k_cache, v_cache) tuples for time layers during inference.
            use_kv_cache: Whether to use and return KV caches for time layers.
        Returns:
            If use_kv_cache is False:
                output: Tensor of shape (..., time_steps, sequence_len, dim).
            If use_kv_cache is True:
                output: Tensor of shape (..., time_steps, sequence_len, dim).
                next_kv_caches: List of updated (k_cache, v_cache) tuples for time layers.
        """
        time_steps, sequence_len = x.shape[-3:-1]   # (..., T, S, D)

        space_mask = self._build_space_mask(sequence_len, device=x.device)  # (S, S) or None

        if use_kv_cache:
            next_kv_caches = []
            if kv_cache is not None:
                cache_iter = iter(kv_cache)  # Cache for time layers.
            else:
                cache_iter = iter([None] * sum(self.is_time_layer))

        for i, layer in enumerate(self.layers):  # AxialTransformerBlock
            is_time_layer = self.is_time_layer[i]

            # 1. Variable Preparation
            time_mask = None
            freqs = None
            layer_cache = None

            if is_time_layer:
                if use_kv_cache:
                    layer_cache = next(cache_iter, None)  # Cache when inference frame by frame.

                x = x.transpose(-2, -3)  # (..., S, T, D)

                # RoPE
                offset = layer_cache[0].shape[-2] if layer_cache is not None else 0 # Offset for cached keys
                freqs = self.rotary(time_steps, offset=offset)

                # Time Mask (Causal)
                if layer_cache is None and not use_kv_cache:
                    time_mask = torch.ones((time_steps, time_steps), device=x.device, dtype=torch.bool).tril()

            # 2. Block Computation
            # Transformer accross dim -2 (Time or Space).
            layer_out = layer(
                x,
                mask=time_mask if is_time_layer else space_mask,
                rotary_pos_emb=freqs, # Space Layer is None
                kv_cache=layer_cache, # Space Layer is None
                return_cache=use_kv_cache,
            )

            if use_kv_cache:
                x, new_cache = layer_out
                if is_time_layer:
                    next_kv_caches.append(new_cache)
            else:
                x = layer_out

            # 3. Transpose back to time-major if this was a time layer.
            if is_time_layer:
                x = x.transpose(-2, -3)  # (..., T, S, D)

        x = self.final_norm(x)

        if use_kv_cache:
            return x, next_kv_caches

        return x
