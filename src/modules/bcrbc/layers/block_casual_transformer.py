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
        block_group_ids: Optional[Tensor] = None,
        context_window: Optional[int] = None,
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
        # Temporal context window: when set (> 0), the causal time mask is *banded* so
        # each step attends only the previous `context_window` steps (instead of the
        # whole causal past). Bounds receptive field + cost on long episodes and keeps
        # train consistent with the windowed eval rollout. None / 0 => unbounded (exact
        # legacy lower-triangular causal mask).
        self.context_window = context_window if (context_window and context_window > 0) else None
        # Per-token agent-group id over the space axis (length S). When present in
        # dynamics mode, the spatial mask is block-diagonal per agent (CTDE): an
        # agent's tokens attend only within its own block, so cross-agent information
        # flows solely through the message tokens placed inside each receiver block.
        if block_group_ids is not None:
            self.register_buffer("block_group_ids", block_group_ids.long(), persistent=False)
        else:
            self.block_group_ids = None

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

            if self.block_group_ids is not None:
                # CTDE: restrict the base attention to a block-diagonal pattern, so an
                # agent's tokens attend only within its own per-agent block (own obs
                # latents, own action, the messages it received, own query). Direct
                # cross-agent observation access is removed; the only cross-agent
                # channel is the message tokens (sender obs) sitting inside each
                # receiver's block. With comm off (no message tokens) agents become
                # fully independent, i.e. decentralizable.
                g = self.block_group_ids.to(device)
                mask = (g[:, None] == g[None, :])

            if self.agent_slice is not None:
                # Follow the paper's section 3.3 rule to prevent causal confusion.
                # Rule 1: no other modality may attend back to the agent (query)
                # tokens (i.e. the columns at agent_slice are set to False).
                mask[:, self.agent_slice] = False

                # Rule 2: restore the agent (query) token ROWS. Without block ids this
                # is full attention (legacy all-ones behaviour); with block ids the
                # rows are restored to the block-diagonal pattern only, so a query
                # attends its own block (own content + self) and NOT other agents.
                if self.block_group_ids is not None:
                    g = self.block_group_ids.to(device)
                    block = (g[:, None] == g[None, :])
                    mask[self.agent_slice] = block[self.agent_slice]
                else:
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
        use_kv_cache: bool = False,
        rope_offset: Optional[int] = None,
    ):
        """Forward pass of the Block Casual Transformer.

        Args:
            x: Input tensor of shape (..., time_steps, sequence_len, dim).
            kv_cache: List of cached (k_cache, v_cache) tuples for time layers during inference.
            use_kv_cache: Whether to use and return KV caches for time layers.
            rope_offset: Absolute time position of the first token in ``x``. When None
                (default) the offset is inferred from the cached-key length — correct only
                while the cache is never evicted (within-build append). With an evicted /
                sliding cache, cache length no longer equals absolute position, so the
                caller MUST pass the true absolute offset. RoPE is shift-invariant, so
                supplying the absolute offset is bit-identical to the dense forward.
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

                # RoPE: position of the new query/key tokens. Prefer the explicit absolute
                # offset (required once the cache is evicted/slid); fall back to cached-key
                # length for the non-evicting append path.
                if rope_offset is not None:
                    offset = rope_offset
                else:
                    offset = layer_cache[0].shape[-2] if layer_cache is not None else 0
                freqs = self.rotary(time_steps, offset=offset)

                # Time Mask (Causal)
                if layer_cache is None and not use_kv_cache:
                    time_mask = torch.ones((time_steps, time_steps), device=x.device, dtype=torch.bool).tril()
                    if self.context_window is not None:
                        # Band the causal mask: each step attends only the previous
                        # `context_window` steps (including itself) -> tril & triu(-(W-1)).
                        time_mask = time_mask & torch.ones(
                            (time_steps, time_steps), device=x.device, dtype=torch.bool
                        ).triu(diagonal=-(self.context_window - 1))
                elif use_kv_cache and self.context_window is not None:
                    # Cached path with a sliding window: keys are [cached.. , new..] and the
                    # query tokens sit at absolute positions [offset, offset+time_steps).
                    # The cache may retain MORE than W entries (so a corrected tail slot can
                    # still see its full W-window), so we cannot rely on plain causality —
                    # build an explicit [time_steps, n_cached+time_steps] band so each query
                    # attends exactly its own W-window (q-W < k <= q) over the joined keys.
                    n_cached = layer_cache[0].shape[-2] if layer_cache is not None else 0
                    total_kv = n_cached + time_steps
                    q_abs = torch.arange(offset, offset + time_steps, device=x.device).view(time_steps, 1)
                    # absolute position of each key: cached keys end just before the new ones.
                    k_abs = torch.arange(offset - n_cached, offset + time_steps, device=x.device).view(1, total_kv)
                    rel = q_abs - k_abs
                    time_mask = (rel >= 0) & (rel < self.context_window)

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
