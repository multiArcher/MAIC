import copy
import importlib
from typing import cast

import torch
from torch.optim import Adam, AdamW, RMSprop, SGD

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from controllers.bcrbc_mac import BCRBCMAC
from learners.learner import Learner
from modules.bcrbc.losses import (
    flow_matching_loss,
    message_reconstruction_loss,
    reconstruction_loss,
    retro_consistency_loss,
)
from modules.bcrbc.retro_replay import RetroReplay
from utils.maker import MixerMaker
from utils.rl_utils import new_build_td_lambda_targets
from utils.th_utils import get_parameters_num


class BCRBCLearner(Learner):
    """QMIX learner for no-delay BC-RBC with optional auxiliary losses."""

    def __init__(self, mac: BCRBCMAC, scheme, logger, args):
        self.logger = logger
        super().__init__()
        self.args = args
        self.device = args.device
        self.mac = mac
        self.scheme = scheme
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.params = list(mac.parameters())
        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        if args.mixer is not None:
            self.mixer = MixerMaker.make(args.mixer, args)
        else:
            self.mixer = torch.nn.Identity()
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        logger.info(f"Mixer Size: {get_parameters_num(self.mixer.parameters())}")

        match args.optimizer.lower():
            case "adam":
                self.optimizer = Adam(params=self.params, lr=args.lr, eps=getattr(args, "optim_eps", 1e-5))
            case "adamw":
                self.optimizer = AdamW(params=self.params, lr=args.lr)
            case "sgd":
                self.optimizer = SGD(params=self.params, lr=args.lr)
            case "rmsprop":
                self.optimizer = RMSprop(
                    params=self.params,
                    lr=args.lr,
                    alpha=getattr(args, "optim_alpha", 0.99),
                    eps=getattr(args, "optim_eps", 1e-5),
                )
            case "rad":
                try:
                    rad_optim = importlib.import_module("rad.optim")
                    self.optimizer = rad_optim.RAD(params=self.params, lr=args.lr, max_iter=30000)
                except ImportError:
                    self.logger.error("RAD optimizer is not installed. Falling back to Adam.")
                    self.optimizer = Adam(params=self.params, lr=args.lr)
            case _:
                raise ValueError(f"Optimizer {args.optimizer} not recognized.")

        self.target_mac = copy.deepcopy(mac)
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)
        self.retro_replay = RetroReplay(
            args.bcrbc_retro_max_replay_len,
            use_comm=args.bcrbc_use_comm,
            n_agents=self.n_agents,
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = cast(torch.Tensor, batch["reward"][:, :-1])
        actions = cast(torch.Tensor, batch["actions"][:, :-1])
        terminated = cast(torch.Tensor, batch["terminated"][:, :-1]).float()
        mask = cast(torch.Tensor, batch["filled"][:, :-1]).float()
        # A transition needs both states; the final bootstrap state has no reward.
        mask *= batch["filled"][:, 1:]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = cast(torch.Tensor, batch["avail_actions"])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards[mask.squeeze(-1).bool()])
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        self.mac.agent.train()
        self.mac.init_hidden(batch.batch_size)
        t_slice = slice(0, batch.max_seq_length)
        mac_out = self.mac.forward(batch, t_slice)
        q_values = mac_out["q_values"]

        chosen_action_values = torch.gather(q_values[:, :-1], dim=-1, index=actions.unsqueeze(-1))
        joint_action_value = self.mixer(chosen_action_values, batch["state"][:, :-1, None, None])

        with torch.no_grad():
            self.target_mac.train()
            self.target_mac.init_hidden(batch.batch_size)
            target_out = self.target_mac.forward(
                batch, t_slice, missing_mask=mac_out["missing_mask"],
                completion_noise=mac_out["completion_noise"], compute_aux=False,
            )
            target_q_values = target_out["q_values"]
            target_q_values = torch.masked_fill(target_q_values, avail_actions.unsqueeze(-2) == 0, -1e7)

            if self.args.double_q:
                mac_out_detach = q_values.detach().clone()
                mac_out_detach = torch.masked_fill(mac_out_detach, avail_actions.unsqueeze(-2) == 0, -1e7)
                cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]
                target_q_values = torch.gather(target_q_values, dim=-1, index=cur_max_actions)
            else:
                target_q_values, _ = target_q_values.max(dim=-1, keepdim=True)

            target_joint_action_value = self.target_mixer(target_q_values, batch["state"][:, :, None, None])
            if self.args.standardise_returns:
                target_joint_action_value = target_joint_action_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean

        with torch.no_grad():
            match target_type := self.args.target_type:
                case "td":
                    td_targets = rewards[..., None, None] + self.args.gamma * target_joint_action_value[:, 1:] * (1 - terminated[..., None, None])
                case "td_lambda":
                    td_targets = new_build_td_lambda_targets(
                        rewards, terminated, mask, target_joint_action_value, self.args.gamma, self.args.td_lambda
                    )
                case _:
                    raise ValueError(f"Invalid target type {target_type}")
            if self.args.standardise_returns:
                self.ret_ms.update(td_targets)
                td_targets = (td_targets - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)

        mixer_mask = mask[..., None, None]
        td_error = joint_action_value - td_targets.reshape_as(joint_action_value)
        masked_td_error = td_error * mixer_mask
        td_loss = (masked_td_error**2).sum() / mixer_mask.sum().clamp_min(1.0)

        rec_weight = self.args.rec_loss_weight
        msg_rec_weight = self.args.msg_rec_loss_weight
        flow_weight = self.args.flow_loss_weight
        generated_rec_weight = self.args.generated_rec_loss_weight

        # [B, T, N, 1, 1]; each valid agent contributes once to auxiliary losses.
        agent_mask = mask[:, :, None, None].expand(-1, -1, self.n_agents, 1, 1)

        if rec_weight > 0:
            rec_loss = reconstruction_loss(
                mac_out["reconstructed_observations"][:, :-1],
                batch["obs"][:, :-1],
                agent_mask.squeeze(-2),
            )
            masked_rec_loss = reconstruction_loss(
                mac_out["masked_reconstructed_observations"][:, :-1],
                batch["obs"][:, :-1],
                agent_mask.squeeze(-2),
            )
            rec_loss = 0.5 * (rec_loss + masked_rec_loss)
        else:
            rec_loss = td_loss.new_zeros(())
        completion_mask = mask[:, :, None, None] * mac_out["missing_mask"][:, :-1]
        flow_loss = flow_matching_loss(
            mac_out["predicted_z"][:, :-1],
            mac_out["target_z"][:, :-1], completion_mask,
        )
        generated_z_loss = flow_matching_loss(
            mac_out["z"][:, :-1],
            mac_out["target_z"][:, :-1], completion_mask,
        )
        generated_rec_loss = reconstruction_loss(
            mac_out["generated_reconstructed_observations"][:, :-1],
            batch["obs"][:, :-1], completion_mask.squeeze(-2),
        )
        if msg_rec_weight > 0 and self.mac.use_comm:
            with torch.no_grad():
                teacher_msgs = self.mac.comm_delay(
                    batch["obs"][:, t_slice].to(self.device), start_t=0, training=True
                )
            msg_rec_loss = message_reconstruction_loss(
                mac_out["reconstructed_messages"][:, :-1],
                teacher_msgs[:, :-1],
                agent_mask,
            )
        else:
            msg_rec_loss = td_loss.new_zeros(())
        retro_weight = self.args.retro_loss_weight
        if retro_weight > 0:
            retro = self.retro_replay.compute(self.mac, batch, t_slice, mac_out)
            retro_time_mask = mask[:, retro["time_slice"]]
            loss_steps = min(retro_time_mask.size(1), retro["retro_mask"].size(1))
            retro_mask = (retro["retro_mask"][:, :loss_steps] * retro_time_mask[:, :loss_steps].unsqueeze(2)).unsqueeze(-1)
            retro_loss = retro_consistency_loss(
                retro["corrected_agent_outputs"][:, :loss_steps],
                retro["target_agent_outputs"][:, :loss_steps],
                retro_mask,
            )
        else:
            retro_loss = td_loss.new_zeros(())

        total_loss = (
            self.args.td_loss_weight * td_loss
            + rec_weight * rec_loss
            + retro_weight * retro_loss
            + msg_rec_weight * msg_rec_loss
            + flow_weight * flow_loss
            + flow_weight * generated_z_loss
            + generated_rec_weight * generated_rec_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with torch.no_grad():
                mask_elems = mixer_mask.sum().item()
                self.logger.log_stat("loss/td_loss", td_loss.item(), t_env)
                self.logger.log_stat("loss/rec_loss", rec_loss.item(), t_env)
                self.logger.log_stat("loss/retro_loss", retro_loss.item(), t_env)
                self.logger.log_stat("loss/msg_rec_loss", msg_rec_loss.item(), t_env)
                self.logger.log_stat("loss/flow_loss", flow_loss.item(), t_env)
                self.logger.log_stat("loss/generated_z_loss", generated_z_loss.item(), t_env)
                self.logger.log_stat(
                    "loss/generated_rec_loss", generated_rec_loss.item(), t_env,
                )
                self.logger.log_stat("loss/total_loss", total_loss.item(), t_env)
                self.logger.log_stat("running/grad_norm", grad_norm.item(), t_env)
                self.logger.log_stat("q_values/td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
                self.logger.log_stat("q_values/q_taken_mean", (chosen_action_values * mixer_mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("q_values/target_mean", (td_targets * mixer_mask).sum().item() / mask_elems, t_env)
                self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.to(self.device)
        self.target_mac.to(self.device)
        if self.mixer is not None:
            self.mixer.to(self.device)
            self.target_mixer.to(self.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        torch.save(self.optimizer.state_dict(), f"{path}/opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(torch.load(f"{path}/opt.th", map_location=lambda storage, loc: storage))
