from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from components.action_selectors.action_selector import ActionSelector
from components.episode_buffer import EpisodeBatch
from modules.bcrbc.bcrbc_model import BCRBCModel
from modules.bcrbc.evaluation_diagnostics import EvaluationDiagnostics
from modules.bcrbc.comm_delay import CommDelay
from modules.bcrbc.retro_replay import RetroReplay
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
        self.use_comm = args.bcrbc_use_comm
        if self.use_comm:
            self.comm_delay = CommDelay(
                self.n_agents,
                delay_mean=args.comm_gaussian_delay_mean,
                delay_std=args.comm_gaussian_delay_std,
                max_delay=args.bcrbc_max_delay,
            )
        else:
            self.comm_delay = None
        # Generative autoregressive imputation at eval (Phase 3). When enabled and
        # test_mode is on, not-yet-arrived observations are replaced by latents the
        # world model generates from history; late arrivals are written back to
        # their generation slot and the rollout is re-rolled (correction-on-arrival).
        self.use_generative_eval = args.bcrbc_generative_eval
        # Persistent across-step eval cache (freeze-on-commit). 0/False => recompute the
        # whole window each step (still freezes committed latents); True => also persist
        # the committed-prefix KV so only the mutable tail [t-D, t] is recomputed.
        self.use_kv_cache = args.bcrbc_kv_cache
        self.hidden_states = None
        # Episode-local transformer history for ordinary one-step action sampling.
        # Independent from the correction-aware generative-eval state below.
        self._online_kv_cache = None
        self._online_encoder_cache = None
        self._online_decoder_cache = None
        # Messages delivered by the stateful online communication channel.
        self._online_message_history = None
        self._online_message_t = -1
        # Stateful generative-eval buffer, rebuilt lazily and reset in init_hidden.
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
        # Snapshot the latent actually used for this decision, before late arrivals.
        self.decision_z = mac_out["z"].detach()
        self.decision_reconstructed_observations = mac_out[
            "reconstructed_observations"
        ].detach()
        q_values = mac_out["q_values"].squeeze(1).squeeze(-2)  # [b, n, a]
        return self.action_selector.select_action(
            q_values[bs],
            avail_actions[bs],
            t_env,
            test_mode=test_mode,
        )

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

        messages = kwargs.pop("messages", None)
        generative_eval = (
            self.use_generative_eval
            and test_mode
            and "obs_override" not in kwargs
            and "z_override" not in kwargs
        )

        # Online communication samples every packet delay once and preserves its
        # arrival across later environment steps. Batched learner forwards keep the
        # stateless zero-delay path below.
        if self.comm_delay is not None and (incremental or generative_eval):
            messages = self._update_online_messages(
                ep_batch,
                t,
                test_mode,
                messages,
            )

        # At evaluation, the generative path fills not-yet-arrived observations.
        # An explicit obs/z override (used by retro replay and diagnostics)
        # bypasses it and runs the plain pass.
        if generative_eval:
            return self._generative_forward(ep_batch, t)

        # This override evaluates reconstruction against the original observation
        # while the delayed trajectory remains unchanged.
        obs_override = kwargs.pop("obs_override", None)
        obs, last_actions = self._build_inputs(ep_batch, t, obs_override=obs_override)

        # Build delayed messages unless a caller already supplied them.
        # Delay lives in the message content only.
        if self.comm_delay is not None and messages is None:
            raw_obs = cast(torch.Tensor, ep_batch["obs"][:, t]).to(self.device)
            messages = self.comm_delay(raw_obs, start_t=t.start or 0, training=not test_mode)

        if incremental:
            z, encoder_cache = self.agent.encode_observations(
                obs, kv_cache=self._online_encoder_cache,
                use_kv_cache=True, rope_offset=t.start or 0,
            )
            out = self.agent(
                obs, last_actions, start_t=t.start or 0, messages=messages,
                z_override=z, kv_cache=self._online_kv_cache,
                use_kv_cache=True, rope_offset=t.start or 0,
                reconstruct=False, **kwargs,
            )
            reconstructed, decoder_cache = self.agent.decode_observations(
                z, kv_cache=self._online_decoder_cache,
                use_kv_cache=True, rope_offset=t.start or 0,
            )
            out["reconstructed_observations"] = reconstructed
            self._online_encoder_cache = self._trim_cache(encoder_cache)
            self._online_decoder_cache = self._trim_cache(decoder_cache)
            self._online_kv_cache = self._trim_cache(out["kv_cache"])
        elif not test_mode and not kwargs:
            # One probability per sequence/agent, then independent block masks.
            missing_probability = torch.rand(
                obs.shape[0], 1, self.n_agents, 1, 1, device=obs.device
            ) * self.args.bcrbc_mask_probability_max
            missing_mask = torch.rand(
                *obs.shape[:3], 1, 1, device=obs.device
            ) < missing_probability
            out = self.agent.forward_training(
                obs, last_actions, messages, missing_mask, start_t=t.start or 0
            )
        else:
            out = self.agent(
                obs, last_actions, start_t=t.start or 0,
                messages=messages, **kwargs,
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
    def _generative_forward(self, ep_batch: EpisodeBatch, t: slice):
        """Correct raw arrivals, encode masked history, then sample missing z.

        Encoder, dynamics and decoder have separate causal caches. Only the
        prefix older than the maximum delay is committed. Every mutable tail is
        recomputed after arrivals; generated z never enters the encoder cache.
        """
        end = t.stop
        observations, previous_actions = self._build_inputs(ep_batch, slice(0, end))
        aligned, missing = self._align_observations(
            observations, ep_batch["obs_gen_t"][:, :end].to(self.device)
        )
        messages = None
        if self.comm_delay is not None:
            messages = self._online_message_history[:, :end]

        commit_end = max(0, end - 1 - int(self.args.bcrbc_max_delay))
        previous = self._eval_state
        if previous is None:
            commit_end = 0
        cache_start = previous["commit_end"] if previous is not None else 0
        if not self.use_kv_cache:
            cache_start = 0
        encoder_cache = previous["encoder_cache"] if cache_start else None
        dynamics_cache = previous["kv"] if cache_start else None
        decoder_cache = previous["decoder_cache"] if cache_start else None

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

        # Frozen generated prefix remains part of the dynamics trajectory.
        z_parts = [previous["z"][:, :cache_start]] if cache_start else []
        committed_dynamics_cache = dynamics_cache
        for step in range(cache_start, end):
            z = encoded_z[:, step:step + 1]
            step_missing = missing[:, step:step + 1]
            step_messages = None if messages is None else messages[:, step:step + 1]
            if step < commit_end:
                z = torch.where(step_missing, previous["z"][:, step:step + 1], z)
            elif step_missing.any():
                generated_z = self.agent.sample_z(
                    previous_actions[:, step:step + 1], step_messages,
                    steps=self.args.bcrbc_flow_steps, start_t=step,
                    kv_cache=dynamics_cache, rope_offset=step,
                )
                z = torch.where(step_missing, generated_z, z)
            output = self.agent.estimate_clean_z(
                z, previous_actions[:, step:step + 1], step_messages,
                torch.ones_like(z[..., :1, :1]), start_t=step,
                kv_cache=dynamics_cache, use_kv_cache=True, rope_offset=step,
            )
            dynamics_cache = self._trim_cache(output["kv_cache"])
            z_parts.append(z)
            if step + 1 == commit_end:
                committed_dynamics_cache = dynamics_cache
        trajectory_z = torch.cat(z_parts, dim=1)

        if cache_start < commit_end:
            _, decoder_cache = self.agent.decode_observations(
                trajectory_z[:, cache_start:commit_end],
                kv_cache=decoder_cache, use_kv_cache=True, rope_offset=cache_start,
            )
        committed_decoder_cache = (
            self._trim_cache(decoder_cache) if decoder_cache is not None else None
        )
        reconstructed, _ = self.agent.decode_observations(
            trajectory_z[:, commit_end:end], kv_cache=decoder_cache,
            use_kv_cache=True, rope_offset=commit_end,
        )
        self._eval_state = {
            "base": 0, "commit_end": commit_end,
            "z": trajectory_z.detach(), "encoded_z": encoded_z.detach(),
            "encoder_cache": committed_encoder_cache,
            "kv": committed_dynamics_cache,
            "decoder_cache": committed_decoder_cache,
            "missing": missing,
        }
        output.update({
            "z": trajectory_z[:, -1:],
            "reconstructed_observations": reconstructed[:, -1:],
            "reconstructed_messages": self.agent.message_decoder(output["agent_outputs"]),
        })
        avail_actions = ep_batch["avail_actions"][:, end - 1:end].unsqueeze(-2)
        output["q_values"] = output["q_values"].masked_fill(avail_actions == 0, -1e7)
        return output

    def _rollout_latent_buffer(self, ep_batch, end, lo=0):
        """Replay arrivals causally for offline diagnostics without altering control."""
        saved_state = self._eval_state
        saved_messages = self._online_message_history
        self._eval_state = None
        if self.comm_delay is not None and saved_messages is None:
            self._online_message_history = self.comm_delay(
                ep_batch["obs"][:, :end].to(self.device), start_t=0, training=False
            )
        for step in range(end):
            self._generative_forward(ep_batch, slice(step, step + 1))
        z = self._eval_state["z"][:, lo:end]
        is_real = ~self._eval_state["missing"][:, lo:end, :, 0, 0]
        observations, actions = self._build_inputs(ep_batch, slice(lo, end))
        self._eval_state = saved_state
        self._online_message_history = saved_messages
        return z, is_real.to(z.dtype), observations, actions

    @torch.no_grad()
    def z_diagnostics(self, ep_batch: EpisodeBatch, full_obs: torch.Tensor) -> dict:
        """Generated-z diagnostics against full-information observations.

        Measures whether generative imputation + correction-on-arrival actually
        rebuilds timely z by comparing against full-information z
        encoded from the un-delayed observations ``full_obs`` (available only for
        analysis / in simulation, never used for control).

        Returns dict of scalars:
            - diag/z_reconstruction_error: mean ||z_imputed - z_full||^2 over generated
              (not-yet-arrived) slots — how well the WM completes missing latents.
            - diag/correction_improvement: on slots later corrected by a delayed
              arrival, how much closer the corrected latent is to the full-info
              latent than the pure-generated estimate (positive = correction helps).
            - diag/msg_recovery_error: message reconstruction error.
        """
        device = self.device
        time_steps = full_obs.size(1)
        num_agents = self.n_agents
        batch_size = ep_batch.batch_size
        observations_full, last_actions = self._build_inputs(
            ep_batch,
            slice(0, time_steps),
            obs_override=full_obs.to(device),
        )
        observations_full = observations_full.to(device)
        last_actions = last_actions.to(device)
        z_full = self.agent.encode_observations(observations_full)

        z_corr, is_real, _, _ = self._rollout_latent_buffer(ep_batch, time_steps)

        # masks carry the token axis (broadcast over it): [B, T, n, 1, 1].
        gen_mask = (is_real < 1).view(batch_size, time_steps, num_agents, 1, 1).float()
        denom = gen_mask.sum().clamp_min(1.0)
        z_reconstruction_error = (
            ((z_corr - z_full) ** 2).mean(-1, keepdim=True) * gen_mask
        ).sum() / denom

        messages = None
        if self.comm_delay is not None:
            messages = self.comm_delay(
                full_obs.to(device),
                start_t=0,
                training=True,
            )
        out = self.agent(
            observations_full,
            last_actions,
            start_t=0,
            z_override=z_corr,
            messages=messages,
        )
        gen_est = z_corr.clone()
        cache = None
        for step in range(time_steps - 1):
            step_messages = None if messages is None else messages[:, step:step + 1]
            history_output = self.agent(
                observations_full[:, step:step + 1],
                last_actions[:, step:step + 1],
                start_t=step,
                z_override=z_corr[:, step:step + 1],
                messages=step_messages,
                kv_cache=cache,
                use_kv_cache=True,
                rope_offset=step,
            )
            cache = history_output["kv_cache"]
            next_messages = None if messages is None else messages[:, step + 1:step + 2]
            gen_est[:, step + 1:step + 2] = self.agent.sample_z(
                last_actions[:, step + 1:step + 2],
                next_messages,
                steps=self.args.bcrbc_flow_steps,
                start_t=step + 1,
                kv_cache=cache,
                rope_offset=step + 1,
            )
        delay = ep_batch["obs_delay"][:, :time_steps].to(device).squeeze(-1)
        delayed_real = ((is_real > 0).float() * (delay > 0).float()).view(
            batch_size,
            time_steps,
            num_agents,
            1,
            1,
        )
        d_denom = delayed_real.sum().clamp_min(1.0)
        err_gen = (((gen_est - z_full) ** 2).mean(-1, keepdim=True) * delayed_real).sum() / d_denom
        err_corr = (((z_corr - z_full) ** 2).mean(-1, keepdim=True) * delayed_real).sum() / d_denom
        correction_improvement = err_gen - err_corr

        diag = {
            "diag/z_reconstruction_error": z_reconstruction_error.item(),
            "diag/correction_improvement": correction_improvement.item(),
        }
        if self.comm_delay is not None:
            teacher = self.comm_delay(full_obs.to(device), start_t=0, training=True)
            target_messages = teacher.mean(dim=3, keepdim=True)
            diag["diag/msg_recovery_error"] = (
                (out["reconstructed_messages"] - target_messages) ** 2
            ).mean().item()
        return diag

    def init_hidden(self, batch_size):
        self.hidden_states = None
        self._online_kv_cache = None
        self._online_encoder_cache = None
        self._online_decoder_cache = None
        self._online_message_history = None
        self._online_message_t = -1
        if self.comm_delay is not None:
            self.comm_delay.reset()
        # New episode(s): drop the stateful generative-eval buffer (frozen z + committed KV).
        self._eval_state = None

    def _update_online_messages(
        self,
        ep_batch: EpisodeBatch,
        t: slice,
        test_mode: bool,
        messages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance the online communication channel through the requested step."""
        start_t = t.start or 0
        end_t = (t.stop or start_t + 1) - 1
        if self._online_message_history is None:
            obs = cast(torch.Tensor, ep_batch["obs"][:, start_t]).to(self.device)
            self._online_message_history = obs.new_zeros(
                ep_batch.batch_size,
                self.args.env_info["episode_limit"] + 1,
                self.n_agents,
                self.n_agents - 1,
                obs.shape[-1],
            )

        for step in range(self._online_message_t + 1, end_t + 1):
            obs = cast(torch.Tensor, ep_batch["obs"][:, step]).to(self.device)
            delivered_messages = self.comm_delay.push_and_query(
                obs,
                step,
                training=not test_mode,
                max_t=self.args.env_info["episode_limit"] + 1,
            )
            self._online_message_history[:, step] = delivered_messages

        if messages is not None:
            stop_t = start_t + messages.shape[1]
            self._online_message_history[:, start_t:stop_t] = messages

        self._online_message_t = max(self._online_message_t, end_t)
        return self._online_message_history[:, start_t:end_t + 1]

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
