"""MAIC learner: QMIX/VDN TD loss plus teammate-model auxiliary losses.

The TD backup follows this repo's QLearner (standardisation, double Q,
hard/soft target updates). Extra MI / entropy / L1 terms come from
MAICAgent and are summed the same way as in the original MAIC learner.
"""

from __future__ import annotations

import copy

import torch as th
from torch.optim import Adam, AdamW, RMSprop

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from learners.learner import Learner
from utils.maker import MixerMaker


class MAICLearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        self.device = args.device

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.target_update_interval_or_tau = getattr(
            args,
            "target_update_interval_or_tau",
            getattr(args, "target_update_interval", 200),
        )

        self.mixer = None
        if args.mixer is not None:
            self.mixer = MixerMaker.make(args.mixer, args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        optimiser_name = str(
            getattr(args, "optimiser", getattr(args, "optimizer", "adam"))
        ).lower()
        if optimiser_name == "rmsprop":
            self.optimiser = RMSprop(
                params=self.params,
                lr=args.lr,
                alpha=getattr(args, "optim_alpha", 0.99),
                eps=getattr(args, "optim_eps", 1e-5),
            )
        elif optimiser_name == "adamw":
            self.optimiser = AdamW(params=self.params, lr=args.lr)
        else:
            self.optimiser = Adam(params=self.params, lr=args.lr)

        self.target_mac = copy.deepcopy(mac)

        if getattr(self.args, "standardise_returns", False):
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if getattr(self.args, "standardise_rewards", False):
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if getattr(self.args, "common_reward", True):
            assert rewards.size(2) == 1, "Expected singular agent dimension for common rewards"
            # QMIX/VDN mix to a joint value (bs, t, 1), matching original MAIC.
            # Expand only for independent Q learning where mixer is None.
            if self.mixer is None:
                rewards = rewards.expand(-1, -1, self.n_agents)

        prepare_for_logging = t_env - self.log_stats_t >= self.args.learner_log_interval
        losses = []

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, returns_ = self._mac_forward(
                self.mac,
                batch,
                t,
                train_mode=True,
                prepare_for_logging=prepare_for_logging,
            )
            mac_out.append(agent_outs)
            if isinstance(returns_, dict) and "logs" in returns_:
                del returns_["logs"]
            losses.append(returns_ if isinstance(returns_, dict) else {})
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self._mac_forward(self.target_mac, batch, t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if getattr(self.args, "standardise_returns", False):
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if getattr(self.args, "standardise_returns", False):
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        external_loss, loss_dict = self._process_loss(losses, batch)
        loss = td_loss + external_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
            self.last_target_update_episode = episode_num
        elif self.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss/td_loss", td_loss.item(), t_env)
            self.logger.log_stat("loss/total_loss", loss.item(), t_env)
            self.logger.log_stat(
                "running/grad_norm",
                grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                t_env,
            )
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "q_values/td_error_abs",
                masked_td_error.abs().sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat(
                "q_values/q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "q_values/target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self._log_for_loss(loss_dict, t_env)
            self.log_stats_t = t_env

    def _mac_forward(self, mac, batch, t, train_mode=False, **kwargs):
        outputs = mac.forward(batch, t=t, train_mode=train_mode, **kwargs)
        if isinstance(outputs, tuple):
            agent_outs = outputs[0]
            extras = outputs[1] if len(outputs) > 1 else {}
            return agent_outs, extras
        return outputs, {}

    def _process_loss(self, losses: list, batch: EpisodeBatch):
        total_loss = 0
        loss_dict = {}
        for item in losses:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if str(key).endswith("loss"):
                    loss_dict[key] = loss_dict.get(key, 0) + value
                    total_loss = total_loss + value
        seq_len = max(int(batch.max_seq_length), 1)
        for key in list(loss_dict.keys()):
            loss_dict[key] = loss_dict[key] / seq_len
        total_loss = total_loss / seq_len
        return total_loss, loss_dict

    def _log_for_loss(self, losses: dict, t):
        for key, value in losses.items():
            logged = value.item() if hasattr(value, "item") else float(value)
            self.logger.log_stat(f"loss/{key}", logged, t)

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.to(self.args.device)
        self.target_mac.to(self.args.device)
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
