import torch


class RetroReplay:
    """Align delayed agent outputs with a patched zero-delay teacher pass."""

    def __init__(self, max_replay_len: int):
        self.max_replay_len = max_replay_len

    def compute(self, mac, batch, t: slice, delayed_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs_delay = batch["obs_delay"][:, t]
        # Design choice: cap the teacher replay to the most recent max_replay_len steps
        # (0 means replay the full window). Truncating shrinks the patched history the
        # zero-delay teacher recomputes over.
        if self.max_replay_len > 0 and obs_delay.size(1) > self.max_replay_len:
            start = obs_delay.size(1) - self.max_replay_len
            obs_delay = obs_delay[:, start:]
            corrected_agent_outputs = delayed_out["agent_outputs"][:, start:]
            t = slice((t.start or 0) + start, t.stop)
            time_slice = slice(start, None)
        else:
            corrected_agent_outputs = delayed_out["agent_outputs"]
            time_slice = slice(None)

        start_t = t.start or 0
        obs_gen_t = batch["obs_gen_t"][:, t]
        raw_obs = batch["obs"][:, t]
        patched_obs = self.patch_observations(raw_obs, obs_gen_t, start_t=start_t)

        with torch.no_grad():
            clean_out = mac.forward(batch, t, obs_override=patched_obs, test_mode=False)

        in_window = obs_gen_t >= start_t
        retro_mask = ((obs_delay > 0) & in_window).to(corrected_agent_outputs.dtype)
        return {
            "corrected_agent_outputs": corrected_agent_outputs,
            "target_agent_outputs": clean_out["agent_outputs"],
            "retro_mask": retro_mask,
            "time_slice": time_slice,
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
