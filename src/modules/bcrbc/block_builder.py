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

    def __init__(self, tokenizer: nn.Module):
        """Initialize the block builder.

        Args:
            tokenizer: DelayTokenizer instance (assumed to be already built).
        """
        super().__init__()
        self.tokenizer = tokenizer

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
        messages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble token block from observation latents, actions, and messages.

        Args:
            obs: [B, T, n_agents, num_latent_tokens, latent_dim]
            last_actions: [B, T, n_agents, n_actions]
            start_t: starting time index
            messages: [B, T, n_agents, n_agents-1, message_dim] or None

        Returns:
            tokens: [B, T, S, D] assembled token tensor
        """
        return self.tokenizer(obs, last_actions, start_t=start_t, messages=messages)
