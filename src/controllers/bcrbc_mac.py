from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from components.action_selectors.action_selector import ActionSelector
from components.episode_buffer import EpisodeBatch
from modules.bcrbc.bcrbc_model import BCRBCModel
from modules.bcrbc.evaluation_diagnostics import EvaluationDiagnostics
from utils.maker import ActionSelectorMaker


class BCRBCMAC(MAC):
    """Joint block-causal MAC for no-delay BC-RBC training."""

    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super().__init__(scheme, groups, args)
        self.args = args
        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.agent_output_type = args.agent_output_type
        self.obs_shape = scheme["obs"]["vshape"]
        self.action_selector: ActionSelector = ActionSelectorMaker.make(args.action_selector, args)
        self.agent = BCRBCModel(self.obs_shape, self.n_actions, self.n_agents, args)
        # Missing raw observations use learned MASK tokens. Late arrivals replace
        # those tokens and trigger recomputation of the mutable history tail.
        # Persistent across-step eval cache (freeze-on-commit). 0/False => recompute the
        # whole window each step (still freezes committed latents); True => also persist
        # the committed-prefix KV so only the mutable tail [t-D, t] is recomputed.
        self.use_kv_cache = args.bcrbc_kv_cache
        self.hidden_states = None
        # Episode-local transformer history for ordinary one-step action sampling.
        # Independent from the correction-aware evaluation state below.
        self._online_kv_cache = None
        self._online_encoder_cache = None
        # Corrected history buffer, rebuilt lazily and reset in init_hidden.
        self._eval_state = None
        self.evaluation_diagnostics = EvaluationDiagnostics(
            output_path=None, batch_size=args.batch_size_run,
        )

    def select_actions(
        self,
        ep_batch: EpisodeBatch,
        t_ep: int,
        t_env: int,
        bs=slice(None),
        test_mode: bool = False,
    ) -> Any:
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        mac_out = self.forward(
            ep_batch, t_ep, test_mode=test_mode, incremental=True
        )
        q_values = mac_out["q_values"].squeeze(1).squeeze(-2)  # [b, n, a]
        self.decision_q_values = q_values.detach()
        actions = self.action_selector.select_action(
            q_values[bs],
            avail_actions[bs],
            t_env,
            test_mode=test_mode,
        )
        return actions

    def forward(
        self,
        ep_batch: EpisodeBatch,
        t: int | slice,
        test_mode=False,
        incremental=False,
        **kwargs,
    ):
        if isinstance(t, int):
            t = slice(t, t + 1)

        missing_mask = kwargs.pop("missing_mask", None)
        completion_noise = kwargs.pop("completion_noise", None)
        compute_aux = kwargs.pop("compute_aux", True)
        masked_eval = (
            test_mode
            and "obs_override" not in kwargs
            and "z_override" not in kwargs
        )

        # At evaluation, encode the available history with MASK at missing slots.
        # An explicit obs/z override (used by retro replay and diagnostics)
        # bypasses it and runs the plain pass.
        if masked_eval:
            return self._masked_forward(ep_batch, t)

        # This override evaluates reconstruction against the original observation
        # while the delayed trajectory remains unchanged.
        obs_override = kwargs.pop("obs_override", None)
        obs, last_actions = self._build_inputs(ep_batch, t, obs_override=obs_override)

        if incremental:
            z, encoder_cache = self.agent.encode_observations(
                obs, kv_cache=self._online_encoder_cache,
                use_kv_cache=True, rope_offset=t.start or 0,
            )
            out = self.agent(
                obs, last_actions, start_t=t.start or 0,
                z_override=z, kv_cache=self._online_kv_cache,
                use_kv_cache=True, rope_offset=t.start or 0,
                reconstruct=False, **kwargs,
            )
            self._online_encoder_cache = self._trim_cache(encoder_cache)
            self._online_kv_cache = self._trim_cache(out["kv_cache"])
        elif not test_mode and not kwargs:
            # One probability per sequence/agent, then independent block masks.
            if missing_mask is None:
                missing_probability = torch.rand(
                    obs.shape[0], 1, self.n_agents, 1, 1, device=obs.device
                ) * self.args.bcrbc_mask_probability_max
                missing_mask = torch.rand(
                    *obs.shape[:3], 1, 1, device=obs.device
                ) < missing_probability
            out = self.agent.forward_training(
                obs, last_actions, missing_mask, start_t=t.start or 0,
                completion_noise=completion_noise, compute_aux=compute_aux,
            )
        else:
            out = self.agent(
                obs, last_actions, start_t=t.start or 0,
                **kwargs,
            )
        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        out["q_values"] = out["q_values"].masked_fill(avail_actions == 0, -1e7)
        return out

    def _trim_cache(self, cache):
        window = self.agent.context_window
        if window:
            cache = [(key[..., -window:, :], value[..., -window:, :])
                     for key, value in cache]
        return [(key.detach(), value.detach()) for key, value in cache]

    @staticmethod
    def _align_observations(observations, generation_times):
        """Align raw packets before temporal encoding; invalid packets never write."""
        length = observations.shape[1]
        query = torch.arange(length, device=observations.device)[None, :, None, None]
        valid = (generation_times >= 0) & (generation_times < length)
        sources = torch.where(valid, query, -1).expand_as(generation_times)
        latest = torch.full_like(generation_times, -1)
        latest.scatter_reduce_(
            1, generation_times.clamp(0, length - 1).long(),
            sources.long(), reduce="amax",
        )
        missing = latest < 0
        index = latest.clamp_min(0).expand_as(observations)
        aligned = observations.gather(1, index)
        aligned = torch.where(missing, 0.0, aligned)
        return aligned, missing[..., None]

    @torch.no_grad()
    def _masked_forward(self, ep_batch: EpisodeBatch, t: slice):
        """Correct raw arrivals and regenerate the mutable missing history.

        Encoder and dynamics have separate causal caches. Only the
        prefix older than the maximum delay is committed. Every mutable tail is
        recomputed after arrivals. Each missing block uses the same initial noise
        on every replay, so a correction does not randomly resample the history.
        """
        output, self._eval_state = self.history_forward(
            ep_batch, t, self._eval_state, generate=True,
        )
        return output

    @torch.no_grad()
    def history_forward(self, ep_batch, t, previous, generate):
        """Replay available history with independent state for each policy path."""
        end = t.stop
        observations, previous_actions = self._build_inputs(ep_batch, slice(0, end))
        aligned, missing = self._align_observations(
            observations, ep_batch["obs_gen_t"][:, :end].to(self.device)
        )
        commit_end = max(0, end - 1 - int(self.args.env_args["max_delay"]))
        if previous is None:
            commit_end = 0
        cache_start = previous["commit_end"] if previous is not None else 0
        if not self.use_kv_cache:
            cache_start = 0
        encoder_cache = previous["encoder_cache"] if cache_start else None
        dynamics_cache = previous["kv"] if cache_start else None

        # Advance encoder's newly committed prefix, then query its mutable tail.
        encoded_parts = [previous["encoded_z"][:, :cache_start]] if cache_start else []
        if cache_start < commit_end:
            encoded, encoder_cache = self.agent.encode_observations(
                aligned[:, cache_start:commit_end],
                missing_mask=missing[:, cache_start:commit_end],
                kv_cache=encoder_cache, use_kv_cache=True, rope_offset=cache_start,
            )
            encoded_parts.append(encoded)
        committed_encoder_cache = (
            self._trim_cache(encoder_cache) if encoder_cache is not None else None
        )
        encoded_tail, _ = self.agent.encode_observations(
            aligned[:, commit_end:end], missing_mask=missing[:, commit_end:end],
            kv_cache=encoder_cache, use_kv_cache=True, rope_offset=commit_end,
        )
        encoded_parts.append(encoded_tail)
        encoded_z = torch.cat(encoded_parts, dim=1)

        # Reuse each generation slot's noise when late arrivals trigger replay.
        noise = previous["noise"] if previous is not None else encoded_z[:, :0]
        if generate and noise.shape[1] < end:
            new_noise = torch.randn_like(encoded_z[:, noise.shape[1]:])
            noise = torch.cat([noise, new_noise], dim=1)
        z_parts = [previous["z"][:, :cache_start]] if cache_start else []
        committed_dynamics_cache = dynamics_cache
        for step in range(cache_start, end):
            z = encoded_z[:, step:step + 1]
            if generate:
                output = self.agent.complete_step(
                    z, previous_actions[:, step:step + 1],
                    missing[:, step:step + 1], noise[:, step:step + 1],
                    start_t=step, kv_cache=dynamics_cache,
                )
            else:
                output = self.agent.estimate_clean_z(
                    z, previous_actions[:, step:step + 1],
                    torch.ones_like(z[..., :1, :1]), start_t=step,
                    kv_cache=dynamics_cache, use_kv_cache=True, rope_offset=step,
                )
                output["z"] = z
            dynamics_cache = self._trim_cache(output["kv_cache"])
            z_parts.append(output["z"])
            if step + 1 == commit_end:
                committed_dynamics_cache = dynamics_cache
        trajectory_z = torch.cat(z_parts, dim=1)

        state = {
            "base": 0, "commit_end": commit_end,
            "z": trajectory_z.detach(), "encoded_z": encoded_z.detach(),
            "encoder_cache": committed_encoder_cache,
            "kv": committed_dynamics_cache,
            "missing": missing,
            "noise": noise,
        }
        output.update({
            "z": trajectory_z[:, -1:],
        })
        avail_actions = ep_batch["avail_actions"][:, end - 1:end].unsqueeze(-2)
        output["q_values"] = output["q_values"].masked_fill(avail_actions == 0, -1e7)
        return output, state

    def init_hidden(self, batch_size):
        self.hidden_states = None
        self._online_kv_cache = None
        self._online_encoder_cache = None
        # New episodes discard the aligned history and committed caches.
        self._eval_state = None

    def save_models(self, path):
        torch.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        state = torch.load(
            f"{path}/agent.th",
            map_location=lambda storage, loc: storage,
        )
        self.agent.load_state_dict(state)

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def _build_agents(self, input_shape):
        self.agent = BCRBCModel(
            self.obs_shape,
            self.n_actions,
            self.n_agents,
            self.args,
        )

    def _get_last_actions(self, batch, t: slice, batch_size: int, n_agents: int):
        if t.start == 0:
            zeros = torch.zeros(batch_size, 1, n_agents, 1, self.n_actions, device=self.device)
            sliced_actions = batch["actions"][:, slice(0, t.stop - 1)]
            sliced_actions = one_hot(sliced_actions.long(), num_classes=self.n_actions)
            return torch.cat([zeros, sliced_actions], dim=1).float()
        last_actions = batch["actions"][:, slice(t.start - 1, t.stop - 1)]
        return one_hot(last_actions.long(), num_classes=self.n_actions).float()

    def _get_input_shape(self, scheme):
        return scheme["obs"]["vshape"]

    def _build_inputs(self, batch, t: slice, obs_override: torch.Tensor | None = None):
        obs_source = batch["obs"][:, t] if obs_override is None else obs_override
        batch_size, _, n_agents = obs_source.shape[:3]
        last_actions = self._get_last_actions(batch, t, batch_size, n_agents)
        return obs_source.float(), last_actions.squeeze(-2).float()
