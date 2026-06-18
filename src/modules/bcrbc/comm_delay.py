"""Communication-delay pathway for BC-RBC.

At training time the replay buffer only ever contains no-delay rollouts, so
communication delay is zero and every receiver sees each sender's current-step
observation. At evaluation time a Gaussian delay (same parameterization as the
CoDe communication model) is sampled, so a receiver gets a *stale* copy of each
sender's observation, taken from generation step ``t - d``. When ``t - d < 0``
no message has been produced yet, so the slot is zero-filled with freshness 0
(per the project rule: zero-fill missing info rather than a learnable token).

The message payload is the sender's raw observation (dim ``d_msg = obs_dim``),
which keeps message reconstruction (L_rec) well defined and avoids any
dependence on model weights for cached content.
"""

import torch
import torch.nn as nn


class CommDelay(nn.Module):
    """Builds delayed inter-agent messages from an observation history.

    Returns message payloads plus per-message delay metadata in the 5-D layout
    ``[B, T, n_agents, n_agents - 1, *]`` expected by :class:`DelayTokenizer`.
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

    def forward(self, obs: torch.Tensor, start_t: int = 0, training: bool = True):
        """Build delayed messages from an observation window.

        Args:
            obs: [B, T, n_agents, obs_dim] observation history for this window.
            start_t: absolute time index of obs[:, 0].
            training: if True, delay is 0 (buffer rollouts are always no-delay);
                if False (eval), Gaussian delay is sampled per receiver-sender slot.

        Returns:
            dict with messages [B,T,n,n-1,obs_dim] and msg_gen_t / msg_arrive_t /
            msg_delay / msg_fresh_mask, each [B,T,n,n-1,1].
        """
        B, T, n, D = obs.shape
        n_send = max(n - 1, 0)
        device = obs.device
        if n_send == 0:
            empty = obs.new_zeros(B, T, n, 0, D)
            empty_meta = obs.new_zeros(B, T, n, 0, 1)
            return {
                "messages": empty,
                "msg_gen_t": empty_meta.long(),
                "msg_arrive_t": empty_meta.long(),
                "msg_delay": empty_meta.long(),
                "msg_fresh_mask": empty_meta,
            }

        t_idx = torch.arange(T, device=device).view(1, T, 1, 1)  # relative query step
        if training or (self.delay_mean == 0.0 and self.delay_std == 0.0):
            delay = torch.zeros(B, T, n, n_send, device=device)
        else:
            delay = torch.normal(
                self.delay_mean, self.delay_std, size=(B, T, n, n_send), device=device
            ).clamp(min=0.0).ceil()
        delay = delay.clamp(max=float(self.max_delay)).long()

        g_rel_raw = t_idx - delay  # [B,T,n,n_send] relative generation index
        valid = (g_rel_raw >= 0)
        g_rel = g_rel_raw.clamp(min=0, max=T - 1)

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
        sender_j = self.sender_idx.to(device).view(1, 1, n, n_send)  # [1,1,n,n_send]
        # advanced index: out[b,t,i,k] = obs[b, g_rel[b,t,i,k], sender_j[i,k]]
        messages = obs[b_idx, g_rel, sender_j]  # [B,T,n,n_send,D]
        messages = messages * valid.unsqueeze(-1).to(messages.dtype)

        gen_abs = (start_t + g_rel).long().unsqueeze(-1)
        arrive_abs = (start_t + t_idx.expand(B, T, n, n_send)).long().unsqueeze(-1)
        delay_out = delay.unsqueeze(-1)
        fresh = (valid & (delay == 0)).to(obs.dtype).unsqueeze(-1)
        return {
            "messages": messages,
            "msg_gen_t": gen_abs,
            "msg_arrive_t": arrive_abs,
            "msg_delay": delay_out,
            "msg_fresh_mask": fresh,
        }
