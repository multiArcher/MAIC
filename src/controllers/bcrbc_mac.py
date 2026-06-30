from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from components.action_selectors.action_selector import ActionSelector
from components.episode_buffer import EpisodeBatch
from modules.bcrbc.dynamics_core import BCRBCDynamicsCore
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
        self.input_shape = self._get_input_shape(scheme)
        self.action_selector: ActionSelector = ActionSelectorMaker.make(args.action_selector, args)
        self.agent = BCRBCDynamicsCore(self.input_shape, self.n_actions, self.n_agents, args, dynamic_obs_dim=self.obs_shape)
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
        self.use_kv_cache = bool(getattr(args, "bcrbc_kv_cache", False))
        self.hidden_states = None
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
        mac_out = self.forward(ep_batch, t_ep, test_mode=test_mode)
        q_values = mac_out["q_values"].squeeze(1).squeeze(-2)
        return self.action_selector.select_action(q_values[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def forward(self, ep_batch: EpisodeBatch, t: int | slice, test_mode=False, **kwargs):
        if isinstance(t, int):
            t = slice(t, t + 1)

        # Algorithm-design polymorphism: at eval (test_mode) the generative path
        # imputes/corrects not-yet-arrived observations. An explicit obs/z override
        # (used by retro replay and diagnostics) bypasses it and runs the plain pass.
        if (
            self.use_generative_eval
            and test_mode
            and "obs_override" not in kwargs
            and "z_override" not in kwargs
        ):
            return self._generative_forward(ep_batch, t)
        obs_override = kwargs.pop("obs_override", None)
        obs, last_actions = self._build_inputs(ep_batch, t, obs_override=obs_override)
        # Build delayed messages (sender observations) for the comm pathway unless a
        # caller already supplied them. Delay lives in the message content only.
        messages = kwargs.pop("messages", None)
        if self.comm_delay is not None and messages is None:
            raw_obs = cast(torch.Tensor, ep_batch["obs"][:, t]).to(self.device)
            messages = self.comm_delay(raw_obs, start_t=t.start or 0, training=not test_mode)
        out = self.agent(obs, last_actions, start_t=t.start or 0, messages=messages, **kwargs)
        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        out["q_values"] = out["q_values"].masked_fill(avail_actions == 0, -1e7)
        return out

    def _rollout_latent_buffer(self, ep_batch: EpisodeBatch, end: int, lo: int = 0):
        """Reconstruct the per-(slot, agent) latent buffer over window [lo, end).

        ``lo`` is the absolute start of the context window (0 = full history). Slots
        whose generation step falls before ``lo`` (slid out of the window) are dropped
        and left for the world model to generate; this is what bounds eval cost on long
        episodes. Returns the corrected/imputed latent buffer ``z_cur``
        [B, end-lo, n, num_token, latent_dim], the ``is_real`` mask [B, end-lo, n] (1
        where a real obs occupies the slot), plus ``obs_aug``/``last_actions`` (over the
        same window) needed to run the world model over the buffer. Shared by the eval
        forward and diagnostics.
        """
        device = self.device
        win = slice(lo, end)
        length = end - lo
        batch_size = ep_batch.batch_size
        num_agents = self.n_agents
        num_latent_tokens = self.agent.num_latent_tokens
        latent_dim = self.agent.latent_dim

        obs_aug, last_actions = self._build_inputs(ep_batch, win)
        obs_aug = obs_aug.to(device)
        last_actions = last_actions.to(device)
        z_delivered = self.agent.latent_encoder(obs_aug).reshape(
            batch_size, length, num_agents, num_latent_tokens, latent_dim
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
        idx = local_gen.view(batch_size, length, num_agents, 1, 1).expand(-1, -1, -1, num_latent_tokens, latent_dim)
        # Only scatter latents/real-flags for in-window arrived packets; dropped query
        # steps contribute nothing to any generation slot.
        z_src = z_delivered * arrived.view(batch_size, length, num_agents, 1, 1).to(z_delivered.dtype)
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
        # substituting a world-model sample drawn from the previous step's belief — then
        # (b) forward just that one step against the cached past (append-only). This is
        # provably identical to the full-prefix forward (the time layer attends each token
        # to its own causal history; appending one step reproduces the dense result), and
        # collapses the rollout to O(length).
        z_cur = z_buf.clone()
        cache = None
        belief_prev = None  # belief at the previously finalized slot
        for tau in range(length):
            if tau > 0 and (is_real[:, tau] < 1).any():
                gen_z = self.agent.flow_dynamics.sample(
                    belief_prev, steps=self.args.bcrbc_flow_steps
                )  # [B, n, num_token, latent_dim]
                real_mask = is_real[:, tau].view(batch_size, num_agents, 1, 1)
                z_cur[:, tau] = real_mask * z_cur[:, tau] + (1.0 - real_mask) * gen_z
            out = self.agent(
                obs_aug[:, tau:tau + 1], last_actions[:, tau:tau + 1],
                start_t=lo + tau, z_override=z_cur[:, tau:tau + 1],
                kv_cache=cache, use_kv_cache=True,
            )
            cache = out["kv_cache"]
            belief_prev = out["beliefs"][:, 0]  # [B, n, 1, belief_dim] -> belief at slot tau

        return z_cur, is_real, obs_aug, last_actions

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
          * off  -> recompute beliefs over the whole window each step (reusing frozen z for
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
        nt = self.agent.num_latent_tokens
        ld = self.agent.latent_dim

        win = slice(lo, end)
        length = end - lo
        obs_aug, last_actions = self._build_inputs(ep_batch, win)
        obs_aug = obs_aug.to(device)
        last_actions = last_actions.to(device)
        z_enc = self.agent.latent_encoder(obs_aug).reshape(bs, length, na, nt, ld)
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
                ep_batch, obs_aug, last_actions, z_cur, is_real,
                lo=lo, end=end, commit_boundary=commit_boundary)
        else:
            final_out = self._eval_rollout_full(
                ep_batch, obs_aug, last_actions, z_cur, is_real,
                lo=lo, end=end, commit_boundary=commit_boundary)

        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        final_out["q_values"] = final_out["q_values"].masked_fill(avail_actions == 0, -1e7)
        return final_out

    def _eval_rollout_full(self, ep_batch, obs_aug, last_actions, z_cur, is_real,
                           *, lo, end, commit_boundary):
        """Freeze reference: recompute beliefs over the whole window each step.

        Walks [lo, end) left-to-right with the within-build incremental KV cache. Committed
        generated slots (< commit_boundary) keep their already-frozen z (seeded by caller);
        only mutable generated slots (>= commit_boundary) are sampled. Stores the finalized
        z buffer back into ``self._eval_state`` for the next step's freeze reuse.
        """
        bs, length, na = is_real.shape
        cache = None
        belief_prev = None
        last_out = None
        for rel in range(length):
            a = lo + rel
            if rel > 0 and a >= commit_boundary and (is_real[:, rel] < 1).any():
                gen_z = self.agent.flow_dynamics.sample(belief_prev, steps=self.args.bcrbc_flow_steps)
                m = is_real[:, rel].view(bs, na, 1, 1)
                z_cur[:, rel] = m * z_cur[:, rel] + (1.0 - m) * gen_z
            out = self.agent(
                obs_aug[:, rel:rel + 1], last_actions[:, rel:rel + 1],
                start_t=a, z_override=z_cur[:, rel:rel + 1],
                kv_cache=cache, use_kv_cache=True, rope_offset=a,
            )
            cache = out["kv_cache"]
            belief_prev = out["beliefs"][:, 0]
            last_out = out
        self._eval_state = {"base": lo, "z": z_cur.detach()}
        return {k: v for k, v in last_out.items() if k != "kv_cache"}

    def _eval_rollout_cached(self, ep_batch, obs_aug, last_actions, z_cur, is_real,
                             *, lo, end, commit_boundary):
        """Committed-prefix cache: persist frozen K/V, recompute only the tail [cb, end).

        State carries the frozen committed K/V covering absolute [lo, kv_end), the frozen z
        buffer, and the belief at the last committed slot (needed to seed tail sampling).
        Each step: (1) evict cache entries older than ``lo``; (2) append any slots that have
        newly committed (kv_end -> commit_boundary) to the frozen cache; (3) recompute the
        mutable tail [commit_boundary, end) against the frozen cache to get the final belief.
        """
        bs, length, na = is_real.shape
        st = self._eval_state
        prev = st if (st is not None and "kv" in st) else None

        # Frozen cache + bookkeeping carried across steps (absolute-indexed).
        if prev is not None and prev["base"] <= lo:
            kv = prev["kv"]
            kv_base = prev["base"]
            kv_end = prev["kv_end"]
            belief_committed = prev["belief_committed"]
            # Evict committed entries that slid out of the window (< lo).
            drop = lo - kv_base
            if drop > 0 and kv is not None:
                kv = [(k[..., drop:, :], v[..., drop:, :]) for (k, v) in kv]
            kv_base = max(kv_base, lo)
        else:
            kv = None
            kv_base = lo
            kv_end = lo
            belief_committed = None

        def rel(a):
            return a - lo

        # (2) Append newly committed slots [kv_end, commit_boundary) to the frozen cache.
        for a in range(kv_end, commit_boundary):
            out = self.agent(
                obs_aug[:, rel(a):rel(a) + 1], last_actions[:, rel(a):rel(a) + 1],
                start_t=a, z_override=z_cur[:, rel(a):rel(a) + 1],
                kv_cache=kv, use_kv_cache=True, rope_offset=a,
            )
            kv = out["kv_cache"]
            belief_committed = out["beliefs"][:, 0]
        kv_end = max(kv_end, commit_boundary)

        # (3) Recompute the mutable tail [commit_boundary, end) against the frozen cache.
        tail_cache = kv
        belief_prev = belief_committed
        last_out = None
        for a in range(commit_boundary, end):
            r = rel(a)
            if a > lo and (is_real[:, r] < 1).any():
                # belief_prev is None only at the very first slot of the episode (a==lo==0).
                gen_z = self.agent.flow_dynamics.sample(belief_prev, steps=self.args.bcrbc_flow_steps)
                m = is_real[:, r].view(bs, na, 1, 1)
                z_cur[:, r] = m * z_cur[:, r] + (1.0 - m) * gen_z
            out = self.agent(
                obs_aug[:, r:r + 1], last_actions[:, r:r + 1],
                start_t=a, z_override=z_cur[:, r:r + 1],
                kv_cache=tail_cache, use_kv_cache=True, rope_offset=a,
            )
            tail_cache = out["kv_cache"]
            belief_prev = out["beliefs"][:, 0]
            last_out = out

        self._eval_state = {
            "base": lo, "z": z_cur.detach(),
            "kv": [(k.detach(), v.detach()) for (k, v) in kv] if kv is not None else None,
            "kv_end": kv_end,
            "belief_committed": belief_committed,
        }
        return {k: v for k, v in last_out.items() if k != "kv_cache"}

    @torch.no_grad()
    def belief_diagnostics(self, ep_batch: EpisodeBatch, full_obs: torch.Tensor) -> dict:
        """Belief-quality diagnostics (paper Tier-1 metrics).

        Measures whether generative imputation + correction-on-arrival actually
        rebuilds the timely belief, by comparing against full-information latents
        encoded from the un-delayed observations ``full_obs`` (available only for
        analysis / in simulation, never used for control).

        Returns dict of scalars:
            - diag/latent_recon_error: mean ||z_imputed - z_full||^2 over generated
              (not-yet-arrived) slots — how well the WM completes missing latents.
            - diag/correction_improvement: on slots later corrected by a delayed
              arrival, how much closer the corrected latent is to the full-info
              latent than the pure-generated estimate (positive = correction helps).
            - diag/msg_recovery_error: ||recon_msg - teacher_msg||^2 (if comm on).
        """
        device = self.device
        time_steps = full_obs.size(1)
        num_agents = self.n_agents
        batch_size = ep_batch.batch_size
        num_latent_tokens = self.agent.num_latent_tokens
        latent_dim = self.agent.latent_dim

        obs_aug_full, last_actions = self._build_inputs(ep_batch, slice(0, time_steps), obs_override=full_obs.to(device))
        obs_aug_full = obs_aug_full.to(device)
        last_actions = last_actions.to(device)
        z_full = self.agent.latent_encoder(obs_aug_full).reshape(
            batch_size, time_steps, num_agents, num_latent_tokens, latent_dim
        )

        z_corr, is_real, _, _ = self._rollout_latent_buffer(ep_batch, time_steps)

        # masks carry the token axis (broadcast over it): [B, T, n, 1, 1].
        gen_mask = (is_real < 1).view(batch_size, time_steps, num_agents, 1, 1).float()
        denom = gen_mask.sum().clamp_min(1.0)
        latent_recon_error = (((z_corr - z_full) ** 2).mean(-1, keepdim=True) * gen_mask).sum() / denom

        out = self.agent(obs_aug_full, last_actions, start_t=0, z_override=z_corr)
        beliefs = out["beliefs"]
        gen_est = z_corr.clone()
        if time_steps > 1:
            gen_est[:, 1:] = self.agent.flow_dynamics.sample(
                beliefs[:, :-1], steps=self.args.bcrbc_flow_steps
            )
        delay = ep_batch["obs_delay"][:, :time_steps].to(device).squeeze(-1)
        delayed_real = ((is_real > 0).float() * (delay > 0).float()).view(batch_size, time_steps, num_agents, 1, 1)
        d_denom = delayed_real.sum().clamp_min(1.0)
        err_gen = (((gen_est - z_full) ** 2).mean(-1, keepdim=True) * delayed_real).sum() / d_denom
        err_corr = (((z_corr - z_full) ** 2).mean(-1, keepdim=True) * delayed_real).sum() / d_denom
        correction_improvement = err_gen - err_corr

        diag = {
            "diag/latent_recon_error": latent_recon_error.item(),
            "diag/correction_improvement": correction_improvement.item(),
        }
        if self.comm_delay is not None:
            teacher = self.comm_delay(full_obs.to(device), start_t=0, training=True)
            tgt = teacher.mean(dim=3, keepdim=True)  # [B,T,n,1,d] to match recon_msg's vector axis
            diag["diag/msg_recovery_error"] = ((out["recon_msg"] - tgt) ** 2).mean().item()
        return diag

    def init_hidden(self, batch_size):
        self.hidden_states = None
        # New episode(s): drop the stateful generative-eval buffer (frozen z + committed KV).
        self._eval_state = None

    def save_models(self, path):
        torch.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(torch.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def _build_agents(self, input_shape):
        self.agent = BCRBCDynamicsCore(input_shape, self.n_actions, self.n_agents, self.args, dynamic_obs_dim=self.obs_shape)

    def _get_last_actions(self, batch, t: slice, batch_size: int, n_agents: int):
        if t.start == 0:
            zeros = torch.zeros(batch_size, 1, n_agents, 1, self.n_actions, device=self.device)
            sliced_actions = batch["actions"][:, slice(0, t.stop - 1)]
            sliced_actions = one_hot(sliced_actions.long(), num_classes=self.n_actions)
            return torch.cat([zeros, sliced_actions], dim=1).float()
        last_actions = batch["actions"][:, slice(t.start - 1, t.stop - 1)]
        return one_hot(last_actions.long(), num_classes=self.n_actions).float()

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape

    def _build_inputs(self, batch, t: slice, obs_override: torch.Tensor | None = None):
        obs_source = batch["obs"][:, t] if obs_override is None else obs_override
        obs_data = obs_source.unsqueeze(-2)
        batch_size, time_size, n_agents, _, _ = obs_data.shape
        last_actions = self._get_last_actions(batch, t, batch_size, n_agents)

        agent_id = torch.arange(self.n_agents, dtype=torch.long, device=batch.device)
        agent_id_one_hot = one_hot(agent_id, num_classes=self.n_agents).reshape(
            1, 1, self.n_agents, 1, self.n_agents
        ).repeat(batch_size, time_size, 1, 1, 1)

        if self.args.obs_agent_id:
            obs_data = torch.cat([obs_data, agent_id_one_hot], dim=-1)
        if self.args.obs_last_action:
            obs_data = torch.cat([obs_data, last_actions], dim=-1)
        return obs_data.squeeze(-2).float(), last_actions.squeeze(-2).float()   # TODO: remove squeeze
