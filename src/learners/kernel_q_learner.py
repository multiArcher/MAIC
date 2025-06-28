import copy
import torch
from torch.optim import Adam, AdamW, RMSprop, SGD

from utils.maker import MixerMaker
from learners.learner import Learner
from components.episode_buffer import EpisodeBatch
from controllers.kernel_controller import KernelMAC
from utils.custom_logging import PyMARLLogger
from utils.th_utils import get_parameters_num
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_q_lambda_targets, new_build_td_lambda_targets


class KernelQLearner(Learner):
    """
    Learner for the Kernel (QMIX-like) algorithm.
    
    This learner computes the standard TD-loss for value-based MARL.
    """
    
    def __init__(self, mac: KernelMAC, scheme, logger, args):
        self.logger = PyMARLLogger("run")
        super(KernelQLearner, self).__init__()
        
        self.args = args
        self.device = args.device
        self.mac = mac
        self.scheme = scheme
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
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
                self.optimizer = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
            case "adamw":
                self.optimizer = AdamW(params=self.params, lr=args.lr)
            case "sgd":        
                self.optimizer = SGD(params=self.params, lr=args.lr)
            case "rmsprop":
                self.optimizer = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            case "rad":
                try:
                    from rad.optim import RAD
                    self.optimizer = RAD(params=self.params, lr=args.lr, max_iter=30000)
                except ImportError:
                    self.logger.error("RAD optimizer is not installed. Please install refering to https://github.com/TobiasLv/RAD. Falling back to Adam.")
                    self.optimizer = Adam(params=self.params, lr=args.lr)
            case _:
                raise ValueError(f"Optimizer {args.optimizer} not recognized.")

        self.target_mac = copy.deepcopy(mac)

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        self.mac.agent.train()
        self.mac.init_hidden(batch.batch_size)

        t_slice = slice(0, batch.max_seq_length)
        q_values = self.mac.forward(batch, t_slice)  # 0 ~ T

        choosen_action_values = torch.gather(q_values, dim=-1, index=actions.unsqueeze(-1))  # 0 ~ T-1
        joint_action_value = self.mixer(
            choosen_action_values, batch["state"][:, :-1, None, None]
            )  # 0 ~ T-1

        with torch.no_grad():
            self.target_mac.train()
            self.target_mac.init_hidden(batch.batch_size)

            target_q_values = self.target_mac.forward(batch, t_slice)  # 0 ~ T            
            target_q_values = torch.masked_fill(target_q_values, avail_actions.unsqueeze(-2)==0, -1e7)

            if self.args.double_q:
                mac_out_detach = q_values.detach().clone()  # 0 ~ T
                mac_out_detach = torch.masked_fill(mac_out_detach, avail_actions.unsqueeze(-2)==0, -1e7)
                cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]  # 0 ~ T
                target_q_values = torch.gather(target_q_values, dim=-1, index=cur_max_actions)  # 0 ~ T
            else:
                target_q_values, _ = target_q_values.max(dim=-1, keepdim=True)  # 0 ~ T
            
            target_joint_action_value = self.target_mixer(
                target_q_values, batch["state"][:, :, None, None]
                )  # 0 ~ T
            
            if self.args.standardise_returns:
                target_joint_action_value = (
                    target_joint_action_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
                )

        with torch.no_grad():
            match target_type := getattr(self.args, "target_type", "td"):
                # 0 ~ T-1
                case "td":
                    td_targets = rewards[..., None, None] + self.args.gamma * target_joint_action_value[:, 1:] * (1 - terminated[..., None, None])
                case "td_lambda":
                    # Note rewards, terminated, mask is 0 ~ T-1, target_joint_action_value is 0 ~ T.
                    td_targets = new_build_td_lambda_targets(rewards, terminated, mask, target_joint_action_value,
                                                      self.args.gamma, self.args.td_lambda)
                case "q_lambda":
                    raise NotImplementedError("Q-Lambda targets are not implemented in KernelQLearner.")
                    # qvals = torch.gather(target_q_values[:, 1:], -1, actions.unsqueeze(-1)).squeeze(-1)
                    # qvals = self.target_mixer(qvals, batch["state"][:, 1:, None, None])
                    # td_targets = build_q_lambda_targets(rewards, terminated, mask, target_joint_action_value, qvals,
                    #                                  self.args.gamma, self.args.td_lambda)
                case _:
                    raise ValueError(f"Invalid target type {target_type}")
            
            if self.args.standardise_returns:
                self.ret_ms.update(td_targets)
                td_targets = (td_targets - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)
            
        mask = mask[..., None, None]
        td_error = joint_action_value - td_targets.reshape_as(joint_action_value)
        masked_td_error = td_error * mask
        td_loss = (masked_td_error**2).sum() / mask.sum()
        
        total_loss = td_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with torch.no_grad():
                mask_elems = mask.sum().item()
                self.logger.log_stat("loss/td_loss", td_loss.item(), t_env)
                self.logger.log_stat("running/grad_norm", grad_norm.item(), t_env)
                self.logger.log_stat("q_values/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("q_values/q_taken_mean", (choosen_action_values * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("q_values/target_mean", (td_targets * mask).sum().item() / mask_elems, t_env)
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
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        self.optimizer.load_state_dict(
            torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
