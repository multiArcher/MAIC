"""Block builder for assembling token blocks from per-modality tensors.

This module provides an abstraction for assembling per-modality token tensors
(observations, actions, messages, queries) into the ordered [B, T, S, D] layout
used by the block-causal transformer.

The token ordering places all content tokens grouped per agent first, then
query tokens contiguous at the tail:
  [ agent0_obs, agent0_action, agent0_msg0, ...,
    agent1_obs, agent1_action, agent1_msg0, ...,
    ...,
    query0, query1, ..., queryN ]
"""

import torch
import torch.nn as nn


class BlockBuilder(nn.Module):
    """Assembles token blocks from per-modality tensors.

    Wraps the DelayTokenizer and provides a clean interface for building
    the [B, T, S, D] token tensor with explicit query-token positioning.
    """

    def __init__(self, tokenizer: nn.Module, window_len: int | None = None):
        """Initialize the block builder.

        Args:
            tokenizer: DelayTokenizer instance (assumed to be already built)
            window_len: Rolling window length for token blocks (None = full sequence).
                        Currently not implemented; defaults to full sequence.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.window_len = window_len
        if window_len is not None:
            raise NotImplementedError("Rolling window tokenization is deferred to Phase F")

    @property
    def query_indices(self) -> torch.Tensor:
        """Return indices of query tokens in the flattened space axis."""
        return self.tokenizer.query_indices

    @property
    def agent_slice(self) -> slice:
        """Return the slice for query tokens in the space dimension."""
        return self.tokenizer.agent_slice

    def forward(
        self,
        obs: torch.Tensor,
        last_actions: torch.Tensor,
        start_t: int = 0,
        obs_delay: torch.Tensor | None = None,
        obs_gen_t: torch.Tensor | None = None,
        obs_fresh_mask: torch.Tensor | None = None,
        messages: torch.Tensor | None = None,
        msg_gen_t: torch.Tensor | None = None,
        msg_arrive_t: torch.Tensor | None = None,
        msg_delay: torch.Tensor | None = None,
        msg_fresh_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble token block from observations, actions, and messages.

        Args:
            obs: [B, T, n_agents, obs_dim]
            last_actions: [B, T, n_agents, n_actions]
            start_t: starting time index
            obs_delay: [B, T, n_agents, 1] or None
            obs_gen_t: [B, T, n_agents, 1] or None
            obs_fresh_mask: [B, T, n_agents, 1] or None
            messages: [B, T, n_agents, n_agents-1, d_msg] or None
            msg_gen_t: [B, T, n_agents, n_agents-1, 1] or None
            msg_arrive_t: [B, T, n_agents, n_agents-1, 1] or None
            msg_delay: [B, T, n_agents, n_agents-1, 1] or None
            msg_fresh_mask: [B, T, n_agents, n_agents-1, 1] or None

        Returns:
            tokens: [B, T, S, D] assembled token tensor
        """
        return self.tokenizer(
            obs,
            last_actions,
            start_t=start_t,
            obs_delay=obs_delay,
            obs_gen_t=obs_gen_t,
            obs_fresh_mask=obs_fresh_mask,
            messages=messages,
            msg_gen_t=msg_gen_t,
            msg_arrive_t=msg_arrive_t,
            msg_delay=msg_delay,
            msg_fresh_mask=msg_fresh_mask,
        )
