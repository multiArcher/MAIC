"""Communication-delay pathway for BC-RBC.

A thin wrapper over the unified :class:`components.delay_model.DelayModel`. It maps
inter-agent communication onto the shared arrival-time model: each receiver gets the
n-1 senders' observations, every sender's message produced at step ``s`` arrives at
``s + delay`` and the receiver sees the **freshest** message that has arrived by the
query step. Unarrived slots are zero-filled.

At training time delay is zero (the replay buffer is always no-delay), so every
receiver sees each sender's current-step observation. At evaluation a Gaussian delay
is sampled, so a receiver gets a stale copy. Delay is never exposed to the model as a
token: the staleness lives entirely in the message *content* (which generation step's
observation is delivered). The payload is the sender's raw observation (dim
``message_dim = obs_dim``), keeping message reconstruction well defined.
"""

import torch
import torch.nn as nn

from components.delay_model import DelayModel


class CommDelay(nn.Module):
    """Builds delayed inter-agent messages via the unified arrival-time model.

    Returns the message payload tensor ``[B, T, n_agents, n_agents - 1, obs_dim]``
    expected by :class:`DelayTokenizer`. Staleness is encoded in the content only;
    no per-message delay metadata is produced.
    """

    def __init__(self, n_agents: int, delay_mean: float = 1.0, delay_std: float = 1.0, max_delay: int = 16):
        super().__init__()
        self.n_agents = n_agents
        self.num_senders = max(n_agents - 1, 0)
        # sender_idx[i] = ordered list of agents j != i (the n-1 senders for receiver i)
        sender_idx = torch.tensor(
            [[j for j in range(n_agents) if j != i] for i in range(n_agents)],
            dtype=torch.long,
        ).reshape(n_agents, self.num_senders)
        self.register_buffer("sender_idx", sender_idx, persistent=False)
        self.delay_model = DelayModel(
            delay_type="gaussian",
            delay_mean=delay_mean,
            delay_std=delay_std,
            max_delay=max_delay,
            delay_per_source=True,
        )

    def forward(self, obs: torch.Tensor, start_t: int = 0, training: bool = True) -> torch.Tensor:
        """Build delayed messages from an observation window.

        Args:
            obs: [B, T, n_agents, obs_dim] observation history for this window.
            start_t: absolute time index of obs[:, 0].
            training: if True, delay is 0 (buffer rollouts are always no-delay);
                if False (eval), Gaussian delay is sampled per receiver-sender slot.

        Returns:
            messages [B, T, n_agents, n_agents - 1, obs_dim]; unarrived slots zero.
        """
        batch_size, time_steps, num_agents, obs_dim = obs.shape
        if self.num_senders == 0:
            return obs.new_zeros(batch_size, time_steps, num_agents, 0, obs_dim)

        # Per-(receiver, sender) message payload at each sent step: the sender's obs.
        # sender_idx selects, for every receiver, its n-1 senders along the agent axis.
        sender_j = self.sender_idx.to(obs.device).view(1, 1, num_agents, self.num_senders)
        sender_j = sender_j.expand(batch_size, time_steps, num_agents, self.num_senders)
        payload = torch.gather(
            obs.unsqueeze(2).expand(batch_size, time_steps, num_agents, num_agents, obs_dim),
            3,
            sender_j.unsqueeze(-1).expand(batch_size, time_steps, num_agents, self.num_senders, obs_dim),
        )  # [B, T, n, n-1, obs_dim]

        # Arrival-time delay over source dims (receiver, sender); feat = obs_dim.
        messages, _, _, _ = self.delay_model(payload, start_t=start_t, training=training, feat_ndims=1)
        return messages
