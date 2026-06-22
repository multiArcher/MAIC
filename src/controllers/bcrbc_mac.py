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
        self.hidden_states = None

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
        delay_metadata = self._build_delay_metadata(ep_batch, t)
        delay_metadata.update(kwargs)
        if self.comm_delay is not None and "messages" not in delay_metadata:
            raw_obs = cast(torch.Tensor, ep_batch["obs"][:, t]).to(self.device)
            comm = self.comm_delay(raw_obs, start_t=t.start or 0, training=not test_mode)
            delay_metadata.update(comm)
        out = self.agent(obs, last_actions, start_t=t.start or 0, **delay_metadata)
        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        out["q_values"] = out["q_values"].masked_fill(avail_actions == 0, -1e7)
        return out

    def _rollout_latent_buffer(self, ep_batch: EpisodeBatch, end: int):
        """Reconstruct the per-(slot, agent) latent buffer over window [0, end).

        Returns the corrected/imputed latent buffer ``z_cur``
        [B, end, n, num_token, z_dim], the ``is_real`` mask [B, end, n] (1 where a
        real obs occupies the slot), plus ``obs_aug``/``last_actions``/``clean_meta``
        needed to run the world model over the buffer. Shared by the eval forward
        and the diagnostics.
        """
        device = self.device
        win = slice(0, end)
        batch_size = ep_batch.batch_size
        num_agents = self.n_agents
        num_latent_tokens = self.agent.num_latent_tokens
        latent_dim = self.agent.latent_dim

        obs_aug, last_actions = self._build_inputs(ep_batch, win)
        obs_aug = obs_aug.to(device)
        last_actions = last_actions.to(device)
        z_delivered = self.agent.latent_encoder(obs_aug).reshape(
            batch_size, end, num_agents, num_latent_tokens, latent_dim
        )

        gen_t = ep_batch["obs_gen_t"][:, win].to(device).long().squeeze(-1)   # [B, end, n]

        # Scatter each delivered latent to its *generation* slot; mark which slots
        # hold a real (received) latent. A delayed obs still provides a real
        # observation for its generation slot, so every delivered packet counts
        # regardless of freshness. Later arrivals overwrite earlier copies.
        z_buf = torch.zeros_like(z_delivered)
        is_real = torch.zeros(batch_size, end, num_agents, device=device)
        local_gen = gen_t.clamp(min=0, max=end - 1)
        idx = local_gen.view(batch_size, end, num_agents, 1, 1).expand(-1, -1, -1, num_latent_tokens, latent_dim)
        z_buf.scatter_(1, idx, z_delivered)
        is_real.scatter_(1, local_gen, torch.ones(batch_size, end, num_agents, device=device))

        clean_meta = self._clean_window_metadata(batch_size, end, num_agents, device)

        # Fill generated slots left-to-right, re-rolling from the latest real history.
        z_cur = z_buf.clone()
        for tau in range(end):
            if not (is_real[:, tau] < 1).any():
                continue
            if tau == 0:
                # No history before slot 0: keep encoded value (or zero if absent).
                continue
            prefix = slice(0, tau)
            out = self.agent(
                obs_aug[:, prefix], last_actions[:, prefix],
                start_t=0, z_override=z_cur[:, prefix], **clean_meta[prefix],
            )
            belief_prev = out["beliefs"][:, tau - 1]  # [B, n, belief_dim]
            gen_z = self.agent.flow_dynamics.sample(
                belief_prev, steps=self.args.bcrbc_flow_steps
            )  # [B, n, num_token, latent_dim]
            real_mask = is_real[:, tau].view(batch_size, num_agents, 1, 1)
            z_cur[:, tau] = real_mask * z_cur[:, tau] + (1.0 - real_mask) * gen_z

        return z_cur, is_real, obs_aug, last_actions, clean_meta

    def _generative_forward(self, ep_batch: EpisodeBatch, t: slice):
        """Eval-time autoregressive imputation + correction-on-arrival.

        Reconstructs the per-(slot, agent) bottleneck-latent buffer over the
        window [0 .. t.stop-1] from everything delivered so far, then returns the
        belief / Q at the final step. Because the episode batch accumulates real
        arrivals across per-step calls, correction-on-arrival is automatic: a late
        packet simply means this step's reconstruction has a *real* latent where an
        earlier step had a *generated* one, so the re-roll from that slot corrects
        all later latents.

        The buffer is presented to the world model with clean/timely metadata
        (delay 0, fresh 1) so the transformer stays in-distribution with the
        no-delay training regime.
        """
        end = t.stop
        win = slice(0, end)
        z_cur, is_real, obs_aug, last_actions, clean_meta = self._rollout_latent_buffer(ep_batch, end)

        out = self.agent(
            obs_aug, last_actions, start_t=0,
            z_override=z_cur, **clean_meta[win],
        )
        # Keep only the final step to match the per-step call contract.
        final = {k: (v[:, -1:] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                 for k, v in out.items()}
        avail_actions = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)
        final["q_values"] = final["q_values"].masked_fill(avail_actions == 0, -1e7)
        return final

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

        z_corr, is_real, _, _, clean_meta = self._rollout_latent_buffer(ep_batch, time_steps)

        # masks carry the token axis (broadcast over it): [B, T, n, 1, 1].
        gen_mask = (is_real < 1).view(batch_size, time_steps, num_agents, 1, 1).float()
        denom = gen_mask.sum().clamp_min(1.0)
        latent_recon_error = (((z_corr - z_full) ** 2).mean(-1, keepdim=True) * gen_mask).sum() / denom

        out = self.agent(obs_aug_full, last_actions, start_t=0,
                         z_override=z_corr, **clean_meta[slice(0, time_steps)])
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
            teacher = self.comm_delay(full_obs.to(device), start_t=0, training=True)["messages"]
            tgt = teacher.mean(dim=3, keepdim=True)  # [B,T,n,1,d] to match recon_msg's vector axis
            diag["diag/msg_recovery_error"] = ((out["recon_msg"] - tgt) ** 2).mean().item()
        return diag

    def _clean_window_metadata(self, batch_size: int, length: int, num_agents: int, device):
        """Timely (delay 0, fresh 1) metadata indexable by a time-slice.

        Returns an object whose __getitem__(slice) yields the obs delay metadata
        dict restricted to that time-slice, so the rollout can feed prefixes.
        """
        gen = torch.arange(length, device=device).view(1, length, 1, 1).expand(batch_size, -1, num_agents, -1).long()

        class _Meta:
            def __getitem__(self, sl):
                return {
                    "obs_delay": torch.zeros(batch_size, sl.stop - (sl.start or 0), num_agents, 1, dtype=torch.long, device=device),
                    "obs_gen_t": gen[:, sl],
                    "obs_fresh_mask": torch.ones(batch_size, sl.stop - (sl.start or 0), num_agents, 1, device=device),
                }

        return _Meta()

    def init_hidden(self, batch_size):
        self.hidden_states = None

    def save_models(self, path):
        torch.save(self.agent.state_dict(), f"{path}/agent.th")

    def load_models(self, path):
        self.agent.load_state_dict(torch.load(f"{path}/agent.th", map_location=lambda storage, loc: storage))

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def _build_agents(self, input_shape):
        self.agent = BCRBCDynamicsCore(input_shape, self.n_actions, self.n_agents, self.args, dynamic_obs_dim=self.obs_shape)

    def _build_delay_metadata(self, batch, t: slice):
        # The three obs-delay keys are added to the scheme unconditionally in run.py
        # and filled by both runners, so they are always present here.
        return {
            "obs_delay": batch["obs_delay"][:, t].to(self.device),
            "obs_gen_t": batch["obs_gen_t"][:, t].to(self.device),
            "obs_fresh_mask": batch["obs_fresh_mask"][:, t].to(self.device),
        }

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
        return obs_data.squeeze(-2).float(), last_actions.squeeze(-2).float()
