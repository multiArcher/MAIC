import torch


class RetroReplay:
    """Batch-level retroactive belief compensation via a zero-delay teacher path.

    The teacher recomputes beliefs from a *patched* history in which every
    delayed observation (and, when communication is enabled, every delayed
    message) is written back to its generation step and presented with
    zero-delay / fresh metadata. The corrected (delayed) beliefs from the online
    pass are then aligned to these full-information teacher beliefs.

    Unarrived evidence (generation step outside the replay window) is left as a
    zero-filled, not-fresh slot rather than a learnable token, matching the
    project rule that delay only appears at evaluation.
    """

    def __init__(self, max_replay_len: int, use_comm: bool, n_agents: int):
        self.max_replay_len = max_replay_len
        self.use_comm = use_comm
        self.n_agents = n_agents

    def compute(self, mac, batch, t: slice, delayed_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_delay = batch["obs_delay"][:, t]
        # Design choice: cap the teacher replay to the most recent max_replay_len steps
        # (0 means replay the full window). Truncating shrinks the patched history the
        # zero-delay teacher recomputes over.
        if self.max_replay_len > 0 and obs_delay.size(1) > self.max_replay_len:
            start = obs_delay.size(1) - self.max_replay_len
            obs_delay = obs_delay[:, start:]
            corrected_beliefs = delayed_out["beliefs"][:, start:]
            t = slice((t.start or 0) + start, t.stop)
            time_slice = slice(start, None)
        else:
            corrected_beliefs = delayed_out["beliefs"]
            time_slice = slice(None)

        start_t = t.start or 0
        obs_gen_t = batch["obs_gen_t"][:, t]
        raw_obs = batch["obs"][:, t]
        patched_obs = self.patch_observations(raw_obs, obs_gen_t, start_t=start_t)
        clean_metadata = self._zero_delay_metadata(batch, t)

        # Design choice: when comm is on, also rebuild full-information teacher messages.
        if self.use_comm:
            # Full-information teacher messages: each sender's patched (gen-time
            # aligned) observation, delivered with zero delay and full freshness.
            clean_metadata.update(self._zero_delay_messages(patched_obs, start_t=start_t))

        with torch.no_grad():
            clean_out = mac.forward(batch, t, obs_override=patched_obs, test_mode=False, **clean_metadata)

        in_window = obs_gen_t >= start_t
        retro_mask = ((obs_delay > 0) & in_window).to(corrected_beliefs.dtype)
        return {
            "corrected_beliefs": corrected_beliefs,
            "target_beliefs": clean_out["beliefs"],
            "retro_mask": retro_mask,
            "time_slice": time_slice,
        }

    @staticmethod
    def _zero_delay_metadata(batch, t: slice) -> dict[str, torch.Tensor]:
        obs = batch["obs"][:, t]
        batch_size, time_size, n_agents = obs.shape[:3]
        device = obs.device
        start_t = t.start or 0
        time_ids = torch.arange(start_t, start_t + time_size, device=device).view(1, time_size, 1, 1)
        time_ids = time_ids.expand(batch_size, -1, n_agents, -1)
        return {
            "obs_delay": torch.zeros(batch_size, time_size, n_agents, 1, dtype=torch.long, device=device),
            "obs_gen_t": time_ids.long(),
            "obs_fresh_mask": torch.ones(batch_size, time_size, n_agents, 1, dtype=torch.float32, device=device),
        }

    @staticmethod
    def _zero_delay_messages(obs: torch.Tensor, start_t: int = 0) -> dict[str, torch.Tensor]:
        """Build full-information (zero-delay, fresh) messages from an obs window.

        messages[b,t,i,k] = obs[b,t, sender_k(i)] where sender_k enumerates j != i.
        """
        batch_size, time_size, n_agents, obs_dim = obs.shape
        device = obs.device
        n_send = max(n_agents - 1, 0)
        if n_send == 0:
            empty = obs.new_zeros(batch_size, time_size, n_agents, 0, 1)
            return {
                "messages": obs.new_zeros(batch_size, time_size, n_agents, 0, obs_dim),
                "msg_gen_t": empty.long(),
                "msg_arrive_t": empty.long(),
                "msg_delay": empty.long(),
                "msg_fresh_mask": empty,
            }
        sender_idx = torch.tensor(
            [[j for j in range(n_agents) if j != i] for i in range(n_agents)],
            dtype=torch.long, device=device,
        ).reshape(n_agents, n_send)
        # messages[b,t,i,k] = obs[b,t,sender_idx[i,k]]
        messages = obs[:, :, sender_idx]  # [B,T,n,n_send,obs_dim]
        t_ids = torch.arange(start_t, start_t + time_size, device=device).view(1, time_size, 1, 1, 1)
        t_ids = t_ids.expand(batch_size, time_size, n_agents, n_send, 1).long()
        return {
            "messages": messages,
            "msg_gen_t": t_ids,
            "msg_arrive_t": t_ids,
            "msg_delay": torch.zeros_like(t_ids),
            "msg_fresh_mask": torch.ones(batch_size, time_size, n_agents, n_send, 1, device=device),
        }

    @staticmethod
    def patch_observations(obs: torch.Tensor, obs_gen_t: torch.Tensor, start_t: int = 0) -> torch.Tensor:
        """Write each delayed observation back to its generation step (vectorized).

        For agent i at arrival step t with generation step tau, the observation
        is moved from local index t to local index (tau - start_t). Later arrivals
        overwrite earlier writes to the same slot (scatter semantics), so the
        freshest copy of each generation step wins.
        """
        batch_size, time_size, n_agents = obs.shape[:3]
        local_gen_t = (obs_gen_t.long().squeeze(-1) - start_t).clamp(min=0, max=time_size - 1)
        patched = torch.zeros_like(obs)
        # scatter along the time axis per (batch, agent)
        index = local_gen_t.unsqueeze(-1).expand(-1, -1, -1, obs.shape[-1])
        patched.scatter_(1, index, obs)
        # slots that never received a write stay zero-filled (unarrived placeholder)
        return patched
