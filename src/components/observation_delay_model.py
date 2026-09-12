from types import SimpleNamespace as SN

import torch


class ObservationDelayModel:
    """
    Applies Gaussian observation delay at the agent input boundary.

    The replay buffer keeps true observations. This model only changes which
    historical observation is fed to the agent for a requested decision time.
    """

    def __init__(self, args: SN):
        self.args = args
        self.device: torch.device | str = args.device
        self.n_agents: int = args.n_agents

        self.enabled: bool = getattr(args, "obs_delay_enabled", False)
        self.apply_train: bool = getattr(args, "obs_delay_apply_train", False)
        self.apply_test: bool = getattr(args, "obs_delay_apply_test", False)
        self.delay_mean: float = getattr(args, "obs_gaussian_delay_mean", 0.0)
        self.delay_std: float = max(0.0, getattr(args, "obs_gaussian_delay_std", 0.0))
        self.discretization: str = getattr(args, "obs_delay_discretization", "round")
        self.max_delay: int | None = getattr(args, "obs_delay_max", None)
        if self.max_delay is not None:
            self.max_delay = int(self.max_delay)
            if self.max_delay < 0:
                raise ValueError("obs_delay_max must be non-negative or None")

        self.last_delays: torch.Tensor | None = None
        self.last_source_times: torch.Tensor | None = None

    def apply(
            self,
            observations: torch.Tensor,
            t: slice,
            training: bool,
        ) -> torch.Tensor:
        """
        Return observations for time slice t after per-agent Gaussian delay.

        Args:
            observations: [B, T_total, N, obs_dim], true observations.
            t: requested current time slice.
            training: whether the owning MAC is in training mode.

        Returns:
            delayed observations with shape [B, len(t), N, obs_dim].
        """
        delayed, _, _ = self.apply_with_metadata(observations, t, training)
        return delayed

    def apply_with_metadata(
            self,
            observations: torch.Tensor,
            t: slice,
            training: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply delay and return ``(observation, effective_delay, source_time)``.

        Effective delay is measured after clamping the source time at the start of
        an episode. This keeps the metadata consistent with the observation that was
        actually gathered.
        """
        if not self._should_apply(training):
            obs = observations[:, t]
            self.last_delays = torch.zeros(
                obs.shape[:-1], dtype=torch.long, device=observations.device
            )
            time_indices = torch.arange(t.start, t.stop, device=observations.device, dtype=torch.long)
            self.last_source_times = time_indices.reshape(1, -1, 1).expand(obs.shape[:-1])
            return obs, self.last_delays, self.last_source_times

        batch_size = observations.shape[0]
        time_size = t.stop - t.start
        obs_dim = observations.shape[-1]

        delays = self._sample_delays(batch_size, time_size, observations.device)
        time_indices = torch.arange(t.start, t.stop, device=observations.device, dtype=torch.long)
        current_times = time_indices.reshape(1, time_size, 1).expand(
            batch_size, time_size, self.n_agents
        )
        source_times = torch.clamp(current_times - delays, min=0, max=observations.shape[1] - 1)
        effective_delays = current_times - source_times

        gather_idx = source_times.unsqueeze(-1).expand(batch_size, time_size, self.n_agents, obs_dim)
        delayed_observations = torch.gather(observations, dim=1, index=gather_idx)

        self.last_delays = effective_delays
        self.last_source_times = source_times
        return delayed_observations, effective_delays, source_times

    def _should_apply(self, training: bool) -> bool:
        if not self.enabled:
            return False
        return self.apply_train if training else self.apply_test

    def _sample_delays(self, batch_size: int, time_size: int, device: torch.device | str) -> torch.Tensor:
        if self.delay_std == 0.0:
            delay_sample = torch.full(
                (batch_size, time_size, self.n_agents),
                self.delay_mean,
                dtype=torch.float,
                device=device,
            )
        else:
            delay_sample = torch.normal(
                self.delay_mean,
                self.delay_std,
                size=(batch_size, time_size, self.n_agents),
                device=device,
            )

        delay_sample = delay_sample.clamp(min=0.0)
        if self.discretization == "ceil":
            delay_sample = delay_sample.ceil()
        elif self.discretization == "floor":
            delay_sample = delay_sample.floor()
        elif self.discretization == "round":
            delay_sample = delay_sample.round()
        else:
            raise ValueError(
                "obs_delay_discretization must be one of: round, ceil, floor. "
                f"Got {self.discretization!r}."
            )

        delay_sample = delay_sample.long()
        if self.max_delay is not None:
            delay_sample = delay_sample.clamp(max=self.max_delay)
        return delay_sample
