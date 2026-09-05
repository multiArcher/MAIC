from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from components.action_selectors.action_selector import ActionSelector
from components.episode_buffer import EpisodeBatch
from modules.bcrbc.bcrbc_model import BCRBCModel
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
        # Messages delivered by the stateful online communication channel.
        self._online_message_history = None
        self._online_message_t = -1
        # Stateful generative-eval buffer, rebuilt lazily and reset in init_hidden.
        self._eval_state = None

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
            # One-step incremental KV cache for ordinary action sampling.
            # The caller is responsible for persisting the cache across env steps.
            out = self.agent(
                obs,
                last_actions,
                start_t=t.start or 0,
                messages=messages,
                kv_cache=self._online_kv_cache,
                use_kv_cache=True,
                rope_offset=t.start or 0,
                **kwargs,
            )

            context_window = self.agent.context_window
            online_kv_cache = out["kv_cache"]

            if context_window:
                online_kv_cache = [
                    (
                        key[..., -context_window:, :],
                        value[..., -context_window:, :],
                    )
                    for key, value in online_kv_cache
                ]
            self._online_kv_cache = [
                (key.detach(), value.detach())
                for key, value in online_kv_cache
            ]

        else:   # batched training forward
            if (
                not test_mode
                and "z_override" not in kwargs
                and "noisy_z" not in kwargs
            ):
                z = self.agent.encode_observations(obs)
                signal_levels = torch.rand(
                    *z.shape[:3], 1, 1, device=z.device
                )
                noise = torch.randn_like(z)
                # Reuse the clean encoding already computed for flow matching.
                kwargs["z_override"] = z
                kwargs["noisy_z"] = (
                    (1.0 - signal_levels) * noise
                    + signal_levels * z
                )
                kwargs["signal_levels"] = signal_levels
            out = self.agent(
                obs,
                last_actions,
                start_t=t.start or 0,
                messages=messages,
                **kwargs,
            )
        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        out["q_values"] = out["q_values"].masked_fill(avail_actions == 0, -1e7)
        return out

    def _rollout_latent_buffer(self, ep_batch: EpisodeBatch, end: int, lo: int = 0):
        """Reconstruct the per-(slot, agent) latent buffer over window [lo, end).

        ``lo`` is the absolute start of the context window (0 = full history). Slots
        whose generation step falls before ``lo`` (slid out of the window) are dropped
        and left for the world model to generate; this is what bounds eval cost on long
        episodes. Returns the corrected/imputed latent buffer ``z_cur``
        [B, end-lo, n, num_z_tokens, z_dim], the ``is_real`` mask [B, end-lo, n] (1
        where a real observation occupies the slot), plus observations/actions (over the
        same window) needed to run the world model over the buffer. Shared by the eval
        forward and diagnostics.
        """
        device = self.device
        win = slice(lo, end)
        length = end - lo
        batch_size = ep_batch.batch_size
        num_agents = self.n_agents
        num_z_tokens = self.agent.num_z_tokens
        z_dim = self.agent.z_dim

        observations, last_actions = self._build_inputs(ep_batch, win)
        observations = observations.to(device)
        last_actions = last_actions.to(device)
        z_delivered = self.agent.encode_observations(observations)
        messages = None
        if self.comm_delay is not None:
            messages = self.comm_delay(
                cast(torch.Tensor, ep_batch["obs"][:, win]).to(device),
                start_t=lo,
                training=False,
            )

        gen_t = ep_batch["obs_gen_t"][:, win].to(device).long().squeeze(-1)   # [B, length, n]

        # Scatter each delivered latent to its *generation* slot (relative to the window
        # start ``lo``); mark which slots hold a real (received) latent. A delayed obs
        # still provides a real observation for its generation slot, so every delivered
        # packet counts regardless of freshness. Later arrivals overwrite earlier copies.
        # An obs is dropped from this window when ``gen_t < lo`` — either unarrived
        # (sentinel gen_t = -1) or generated before the window slid past it; such slots
        # have no real latent here and are left for the world model to generate.
        z_buf = torch.zeros_like(z_delivered)
        is_real = torch.zeros(batch_size, length, num_agents, device=device)
        arrived = gen_t >= lo                                      # [B, length, n]
        local_gen = (gen_t - lo).clamp(min=0, max=length - 1)
        idx = local_gen.view(batch_size, length, num_agents, 1, 1).expand(
            -1, -1, -1, num_z_tokens, z_dim
        )
        # Only scatter latents/real-flags for in-window arrived packets; dropped query
        # steps contribute nothing to any generation slot.
        real_mask = arrived.view(batch_size, length, num_agents, 1, 1).to(
            z_delivered.dtype
        )
        z_src = z_delivered * real_mask
        z_buf.scatter_(1, idx, z_src)
        is_real.scatter_(1, local_gen, arrived.to(is_real.dtype))

        # Fill generated slots left-to-right, re-rolling from the latest real history.
        # Absolute time of local slot 0 is ``lo`` (start_t below), so the time embedding
        # / banded attention see correct absolute positions.
        #
        # Incremental KV-cache pass: instead of re-running the transformer over the whole
        # growing prefix [0, tau) for every generated slot (O(length^2) per build), we
        # walk the window left-to-right ONCE, threading the per-time-layer KV cache. At
        # each position we (a) finalize that slot's latent — keeping a real arrival or
        # substituting a world-model sample drawn from causal history — then
        # (b) forward just that one step against the cached past (append-only). This is
        # provably identical to the full-prefix forward (the time layer attends each token
        # to its own causal history; appending one step reproduces the dense result), and
        # collapses the rollout to O(length).
        z_cur = z_buf.clone()
        cache = None
        for tau in range(length):
            if (is_real[:, tau] < 1).any():
                step_messages = None if messages is None else messages[:, tau:tau + 1]
                gen_z = self.agent.sample_z(
                    last_actions[:, tau:tau + 1],
                    step_messages,
                    steps=self.args.bcrbc_flow_steps,
                    start_t=lo + tau,
                    kv_cache=cache,
                    rope_offset=lo + tau,
                )[:, 0]
                real_mask = is_real[:, tau].view(batch_size, num_agents, 1, 1)
                z_cur[:, tau] = real_mask * z_cur[:, tau] + (1.0 - real_mask) * gen_z
            step_messages = None if messages is None else messages[:, tau:tau + 1]
            out = self.agent(
                observations[:, tau:tau + 1], last_actions[:, tau:tau + 1],
                start_t=lo + tau, z_override=z_cur[:, tau:tau + 1],
                messages=step_messages,
                kv_cache=cache, use_kv_cache=True,
                rope_offset=lo + tau,
            )
            cache = out["kv_cache"]

        return z_cur, is_real, observations, last_actions

    def _generative_forward(self, ep_batch: EpisodeBatch, t: slice):
        """Eval-time autoregressive imputation + correction-on-arrival (freeze-on-commit).

        Stateful across env steps. A not-yet-arrived slot's latent is generated *once*
        and then frozen: a generated obs becomes part of the trajectory and is never
        re-sampled. It can only change if (a) the real obs arrives, or (b) an earlier slot
        it was rolled from is itself corrected. Because a sampled delay never exceeds
        ``max_delay`` (= D), an arrival at env step t lands at generation slot >= t - D, so
        only the mutable tail ``[t-D, t]`` can change between steps; everything older is
        committed (frozen z, and frozen time-layer K/V).

        Two equivalent realizations, selected by ``self.use_kv_cache`` (both freeze):
          * off  -> recompute agent outputs over the whole window each step (reusing frozen z for
                    committed slots); O(W) transformer work.
          * on   -> persist the committed-prefix K/V and recompute only the tail
                    ``[t-D, t]``; O(D) transformer work. Bit-identical to the off path.
        """
        end = t.stop
        window = self.agent.context_window
        W = window if (window and window > 0) else None
        lo = max(0, end - W) if W is not None else 0
        D = int(self.args.bcrbc_max_delay)
        # Frozen committed prefix is [lo, commit_boundary); mutable tail is [commit_boundary, end).
        commit_boundary = max(lo, end - 1 - D)

        device = self.device
        bs = ep_batch.batch_size
        na = self.n_agents
        nt = self.agent.num_z_tokens
        ld = self.agent.z_dim

        win = slice(lo, end)
        length = end - lo
        observations, last_actions = self._build_inputs(ep_batch, win)
        observations = observations.to(device)
        last_actions = last_actions.to(device)
        z_enc = self.agent.encode_observations(observations)
        messages = None
        if self.comm_delay is not None:
            messages = self._online_message_history[:, lo:end]
        gen_t = ep_batch["obs_gen_t"][:, win].to(device).long().squeeze(-1)  # [B, length, n]

        # Real (arrived) latents scattered to their generation slot; is_real marks them.
        z_real = torch.zeros_like(z_enc)
        is_real = torch.zeros(bs, length, na, device=device)
        arrived = gen_t >= lo
        local_gen = (gen_t - lo).clamp(min=0, max=length - 1)
        idx = local_gen.view(bs, length, na, 1, 1).expand(-1, -1, -1, nt, ld)
        z_real.scatter_(1, idx, z_enc * arrived.view(bs, length, na, 1, 1).to(z_enc.dtype))
        is_real.scatter_(1, local_gen, arrived.to(is_real.dtype))

        # Seed z_cur: real slots use their encoded real latent; committed generated slots
        # reuse the frozen value carried in state; mutable generated slots are (re)sampled
        # below. Frozen reuse is what makes a committed generated obs part of the trajectory.
        z_cur = z_real.clone()
        st = self._eval_state
        if st is not None and st["base"] <= lo:
            prev_base = st["base"]
            prev_z = st["z"]  # absolute slots [prev_base, prev_base + prev_z.size(1))
            for a in range(lo, commit_boundary):
                rel_prev = a - prev_base
                if 0 <= rel_prev < prev_z.size(1):
                    rel = a - lo
                    real_a = is_real[:, rel].view(bs, na, 1, 1)
                    z_cur[:, rel] = real_a * z_cur[:, rel] + (1.0 - real_a) * prev_z[:, rel_prev]

        if self.use_kv_cache:
            final_out = self._eval_rollout_cached(
                ep_batch, observations, last_actions, messages, z_cur, is_real,
                lo=lo, end=end, commit_boundary=commit_boundary)
        else:
            final_out = self._eval_rollout_full(
                ep_batch, observations, last_actions, messages, z_cur, is_real,
                lo=lo, end=end, commit_boundary=commit_boundary)

        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        final_out["q_values"] = final_out["q_values"].masked_fill(avail_actions == 0, -1e7)
        return final_out

    def _eval_rollout_full(self, ep_batch, observations, last_actions, messages, z_cur, is_real,
                           *, lo, end, commit_boundary):
        """Freeze reference: recompute agent outputs over the whole window each step.

        Walks [lo, end) left-to-right with the within-build incremental KV cache. Committed
        generated slots (< commit_boundary) keep their already-frozen z (seeded by caller);
        only mutable generated slots (>= commit_boundary) are sampled. Stores the finalized
        z buffer back into ``self._eval_state`` for the next step's freeze reuse.
        """
        bs, length, na = is_real.shape
        cache = None
        last_out = None
        for rel in range(length):
            a = lo + rel
            if a >= commit_boundary and (is_real[:, rel] < 1).any():
                step_messages = None if messages is None else messages[:, rel:rel + 1]
                gen_z = self.agent.sample_z(
                    last_actions[:, rel:rel + 1],
                    step_messages,
                    steps=self.args.bcrbc_flow_steps,
                    start_t=a,
                    kv_cache=cache,
                    rope_offset=a,
                )[:, 0]
                m = is_real[:, rel].view(bs, na, 1, 1)
                z_cur[:, rel] = m * z_cur[:, rel] + (1.0 - m) * gen_z
            step_messages = None if messages is None else messages[:, rel:rel + 1]
            out = self.agent(
                observations[:, rel:rel + 1], last_actions[:, rel:rel + 1],
                start_t=a, z_override=z_cur[:, rel:rel + 1],
                messages=step_messages,
                kv_cache=cache, use_kv_cache=True, rope_offset=a,
            )
            cache = out["kv_cache"]
            last_out = out
        self._eval_state = {"base": lo, "z": z_cur.detach()}
        return {k: v for k, v in last_out.items() if k != "kv_cache"}

    def _eval_rollout_cached(self, ep_batch, observations, last_actions, messages, z_cur, is_real,
                             *, lo, end, commit_boundary):
        """Committed-prefix cache: persist frozen K/V, recompute only the tail [cb, end).

        State carries the frozen committed K/V covering absolute [lo, kv_end), the frozen z
        buffer, and the agent output at the last committed slot.
        Each step: (1) evict cache entries older than ``lo``; (2) append any slots that have
        newly committed (kv_end -> commit_boundary) to the frozen cache; (3) recompute the
        mutable tail [commit_boundary, end) against the frozen cache.
        """
        bs, length, na = is_real.shape
        st = self._eval_state
        prev = st if (st is not None and "kv" in st) else None

        # Frozen cache + bookkeeping carried across steps (absolute-indexed).
        if prev is not None and prev["base"] <= lo:
            kv = prev["kv"]
            kv_base = prev["base"]
            kv_end = prev["kv_end"]
            agent_outputs_committed = prev["agent_outputs_committed"]
            # Evict committed entries that slid out of the window (< lo).
            drop = lo - kv_base
            if drop > 0 and kv is not None:
                kv = [(k[..., drop:, :], v[..., drop:, :]) for (k, v) in kv]
            kv_base = max(kv_base, lo)
        else:
            kv = None
            kv_base = lo
            kv_end = lo
            agent_outputs_committed = None

        def rel(a):
            return a - lo

        # (2) Append newly committed slots [kv_end, commit_boundary) to the frozen cache.
        for a in range(kv_end, commit_boundary):
            step_messages = None if messages is None else messages[:, rel(a):rel(a) + 1]
            out = self.agent(
                observations[:, rel(a):rel(a) + 1], last_actions[:, rel(a):rel(a) + 1],
                start_t=a, z_override=z_cur[:, rel(a):rel(a) + 1],
                messages=step_messages,
                kv_cache=kv, use_kv_cache=True, rope_offset=a,
            )
            kv = out["kv_cache"]
            agent_outputs_committed = out["agent_outputs"][:, 0]
        kv_end = max(kv_end, commit_boundary)

        # (3) Recompute the mutable tail [commit_boundary, end) against the frozen cache.
        tail_cache = kv
        last_out = None
        for a in range(commit_boundary, end):
            r = rel(a)
            if (is_real[:, r] < 1).any():
                step_messages = None if messages is None else messages[:, r:r + 1]
                gen_z = self.agent.sample_z(
                    last_actions[:, r:r + 1],
                    step_messages,
                    steps=self.args.bcrbc_flow_steps,
                    start_t=a,
                    kv_cache=tail_cache,
                    rope_offset=a,
                )[:, 0]
                m = is_real[:, r].view(bs, na, 1, 1)
                z_cur[:, r] = m * z_cur[:, r] + (1.0 - m) * gen_z
            step_messages = None if messages is None else messages[:, r:r + 1]
            out = self.agent(
                observations[:, r:r + 1], last_actions[:, r:r + 1],
                start_t=a, z_override=z_cur[:, r:r + 1],
                messages=step_messages,
                kv_cache=tail_cache, use_kv_cache=True, rope_offset=a,
            )
            tail_cache = out["kv_cache"]
            last_out = out

        self._eval_state = {
            "base": lo, "z": z_cur.detach(),
            "kv": [(k.detach(), v.detach()) for (k, v) in kv] if kv is not None else None,
            "kv_end": kv_end,
            "agent_outputs_committed": agent_outputs_committed,
        }
        return {k: v for k, v in last_out.items() if k != "kv_cache"}

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
