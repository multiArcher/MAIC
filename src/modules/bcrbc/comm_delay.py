"""Communication-delay pathway for BC-RBC.

At training time the replay buffer only ever contains no-delay rollouts, so
communication delay is zero and every receiver sees each sender's current-step
observation. At evaluation time a Gaussian delay (same parameterization as the
CoDe communication model) is sampled, so a receiver gets a *stale* copy of each
sender's observation, taken from generation step ``t - d``. When ``t - d < 0``
no message has been produced yet, so the slot is zero-filled.

Delay is never exposed to the model as a token: the staleness lives entirely in
the message *content* (which generation step's observation is delivered). The
message payload is the sender's raw observation (dim ``message_dim = obs_dim``),
which keeps message reconstruction well defined and avoids any dependence on
model weights for cached content.
"""

import torch
import torch.nn as nn


class CommDelay(nn.Module):
    """Builds delayed inter-agent messages from an observation history.

    Returns the message payload tensor ``[B, T, n_agents, n_agents - 1, obs_dim]``
    expected by :class:`DelayTokenizer`. Staleness is encoded in the content only;
    no per-message delay metadata is produced.
    """

    def __init__(self, n_agents: int, delay_mean: float = 1.0, delay_std: float = 1.0, max_delay: int = 16):
        super().__init__()
        self.n_agents = n_agents
        self.delay_mean = float(delay_mean)
        self.delay_std = max(0.0, float(delay_std))
        self.max_delay = int(max_delay)
        # sender_idx[i] = ordered list of agents j != i (the n-1 senders for receiver i)
        sender_idx = torch.tensor(
            [[j for j in range(n_agents) if j != i] for i in range(n_agents)],
            dtype=torch.long,
        ).reshape(n_agents, max(n_agents - 1, 0))
        self.register_buffer("sender_idx", sender_idx, persistent=False)

    def forward(self, obs: torch.Tensor, start_t: int = 0, training: bool = True) -> torch.Tensor:
        """Build delayed messages from an observation window.

        Args:
            obs: [B, T, n_agents, obs_dim] observation history for this window.
            start_t: absolute time index of obs[:, 0] (unused; kept for call symmetry).
            training: if True, delay is 0 (buffer rollouts are always no-delay);
                if False (eval), Gaussian delay is sampled per receiver-sender slot.

        Returns:
            messages [B, T, n_agents, n_agents - 1, obs_dim]; unarrived slots zero.
        """
        batch_size, time_steps, num_agents, obs_dim = obs.shape
        num_senders = max(num_agents - 1, 0)
        device = obs.device
        if num_senders == 0:
            return obs.new_zeros(batch_size, time_steps, num_agents, 0, obs_dim)

        query_step = torch.arange(time_steps, device=device).view(1, time_steps, 1, 1)  # relative query step
        if training or (self.delay_mean == 0.0 and self.delay_std == 0.0):
            delay = torch.zeros(batch_size, time_steps, num_agents, num_senders, device=device)
        else:
            delay = torch.normal(
                self.delay_mean, self.delay_std, size=(batch_size, time_steps, num_agents, num_senders), device=device
            ).clamp(min=0.0).ceil()
        delay = delay.clamp(max=float(self.max_delay)).long()

        gen_step_raw = query_step - delay  # [B,T,n,num_senders] relative generation index
        valid = (gen_step_raw >= 0)
        gen_step = gen_step_raw.clamp(min=0, max=time_steps - 1)

        batch_index = torch.arange(batch_size, device=device).view(batch_size, 1, 1, 1)
        sender_j = self.sender_idx.to(device).view(1, 1, num_agents, num_senders)  # [1,1,n,num_senders]
        # advanced index: out[b,t,i,k] = obs[b, gen_step[b,t,i,k], sender_j[i,k]]
        messages = obs[batch_index, gen_step, sender_j]  # [B,T,n,num_senders,obs_dim]
        messages = messages * valid.unsqueeze(-1).to(messages.dtype)
        return messages

