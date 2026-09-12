import copy

import torch
from torch.optim import Adam, RMSprop

from components.episode_buffer import EpisodeBatch
from learners.learner import Learner
from modules.mixers.pymarl2_qmix_mixer import Mixer
from utils.rl_utils import build_td_lambda_targets
from utils.th_utils import get_parameters_num


class RDCQLearner(Learner):
    """Echo predictor plus QMIX, without cheating or teacher-student losses."""

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.device = args.device
        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1

        if args.mixer != "pymarl2_qmix_mixer":
            raise ValueError("RDC QMIX requires mixer='pymarl2_qmix_mixer'")
        self.mixer = Mixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mac = copy.deepcopy(mac)

        self.rl_params = list(self.mac.agent.parameters()) + list(self.mixer.parameters())
        self.predictor_params = list(self.mac.predictor.parameters())
        if args.optimizer == "adam":
            self.optimiser = Adam(
                self.rl_params,
                lr=args.lr,
                weight_decay=getattr(args, "weight_decay", 0),
            )
            self.predictor_optimiser = Adam(
                self.predictor_params,
                lr=args.pd_lr,
                weight_decay=getattr(args, "weight_decay", 0),
            )
        else:
            self.optimiser = RMSprop(
                self.rl_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
            )
            self.predictor_optimiser = RMSprop(
                self.predictor_params,
                lr=args.pd_lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        logger.info(f"RDC mixer size: {get_parameters_num(self.mixer.parameters())}")
        logger.info(f"RDC predictor size: {get_parameters_num(self.mac.predictor.parameters())}")

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        predicted_obs, predictor_stats = self._train_predictor(batch)

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.agent.train()
        self.mac.init_hidden(batch.batch_size)
        mac_out = []
        for timestep in range(batch.max_seq_length):
            mac_out.append(
                self.mac.forward(batch, timestep, predicted_obs=predicted_obs)
            )
        mac_out = torch.stack(mac_out, dim=1)
        mac_out = mac_out.masked_fill(avail_actions == 0, -1e7)
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).squeeze(3)

        with torch.no_grad():
            self.target_mac.agent.eval()
            self.target_mac.init_hidden(batch.batch_size)
            target_mac_out = []
            for timestep in range(batch.max_seq_length):
                target_mac_out.append(
                    self.target_mac.forward(
                        batch, timestep, predicted_obs=predicted_obs
                    )
                )
            target_mac_out = torch.stack(target_mac_out, dim=1)
            target_mac_out = target_mac_out.masked_fill(avail_actions == 0, -1e7)
            cur_max_actions = mac_out.detach().max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(
                target_mac_out, 3, cur_max_actions
            ).squeeze(3)
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])
            targets = build_td_lambda_targets(
                rewards,
                terminated,
                mask,
                target_max_qvals,
                self.args.n_agents,
                self.args.gamma,
                self.args.td_lambda,
            )

        chosen_action_qvals = self.mixer(
            chosen_action_qvals, batch["state"][:, :-1]
        )
        td_error = chosen_action_qvals - targets.detach()
        expanded_mask = mask.expand_as(td_error)
        masked_td_error = td_error * expanded_mask
        td_loss = 0.5 * masked_td_error.pow(2).sum() / expanded_mask.sum().clamp_min(1.0)

        self.optimiser.zero_grad()
        td_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.rl_params, self.args.grad_norm_clip
        )
        self.optimiser.step()

        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = expanded_mask.sum().item()
            self.logger.log_stat("loss/rdc_td", td_loss.item(), t_env)
            self.logger.log_stat("loss/rdc_predictor", predictor_stats["loss"], t_env)
            self.logger.log_stat("loss/rdc_regression", predictor_stats["regression"], t_env)
            self.logger.log_stat("loss/rdc_classification", predictor_stats["classification"], t_env)
            self.logger.log_stat("loss/rdc_observation", predictor_stats["observation"], t_env)
            self.logger.log_stat("delay/rdc_train_nonzero_ratio", predictor_stats["nonzero_ratio"], t_env)
            self.logger.log_stat("delay/rdc_train_mean", predictor_stats["delay_mean"], t_env)
            self.logger.log_stat("delay/rdc_train_max", predictor_stats["delay_max"], t_env)
            self.logger.log_stat("running/rdc_predictor_grad_norm", predictor_stats["grad_norm"], t_env)
            self.logger.log_stat("running/rdc_grad_norm", float(grad_norm), t_env)
            self.logger.log_stat(
                "q_values/rdc_td_error_abs",
                masked_td_error.abs().sum().item() / max(mask_elems, 1.0),
                t_env,
            )
            self.log_stats_t = t_env

        return {}

    def _train_predictor(self, batch: EpisodeBatch):
        self.mac.predictor.train()
        chunks = min(int(self.args.num_minibatch_predictor), batch.batch_size)
        boundaries = torch.linspace(0, batch.batch_size, chunks + 1).long().tolist()
        predictions = []
        totals = {
            "loss": 0.0,
            "regression": 0.0,
            "classification": 0.0,
            "observation": 0.0,
            "nonzero_ratio": 0.0,
            "delay_mean": 0.0,
            "delay_max": 0.0,
            "grad_norm": 0.0,
        }

        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            minibatch = batch[start:stop]
            output = self.mac.predict_batch(
                minibatch, training=True, compute_loss=True
            )
            predictor_loss = (
                self.args.rdc_regression_loss_weight * output["regression_loss"]
                + self.args.rdc_classification_loss_weight
                * output["classification_loss"]
            )
            self.predictor_optimiser.zero_grad()
            predictor_loss.backward()
            predictor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.predictor_params, self.args.grad_norm_clip
            )
            self.predictor_optimiser.step()

            weight = (stop - start) / batch.batch_size
            predictions.append(output["predicted_obs"].detach())
            totals["loss"] += predictor_loss.item() * weight
            totals["regression"] += output["regression_loss"].item() * weight
            totals["classification"] += output["classification_loss"].item() * weight
            totals["observation"] += output["observation_loss"].item() * weight
            totals["nonzero_ratio"] += (output["delays"] > 0).float().mean().item() * weight
            totals["delay_mean"] += output["delays"].float().mean().item() * weight
            totals["delay_max"] = max(
                totals["delay_max"], float(output["delays"].max().item())
            )
            totals["grad_norm"] += float(predictor_grad_norm) * weight

        self.mac.predictor.eval()
        return torch.cat(predictions, dim=0), totals

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated RDC target network")

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.agent.parameters(), self.mac.agent.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        for target_param, param in zip(
            self.target_mixer.parameters(), self.mixer.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def cuda(self):
        self.to(torch.device("cuda"))

    def to(self, device):
        self.device = device
        self.mac.to(device)
        self.target_mac.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.mac.predictor.state_dict(), f"{path}/predictor.th")
        torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        torch.save(self.optimiser.state_dict(), f"{path}/opt.th")
        torch.save(
            self.predictor_optimiser.state_dict(), f"{path}/predictor_opt.th"
        )

    def load_models(self, path):
        self.mac.load_models(path)
        predictor_state = torch.load(
            f"{path}/predictor.th", map_location=lambda storage, loc: storage
        )
        self.mac.predictor.load_state_dict(predictor_state)
        self.target_mac.load_models(path)
        self.target_mac.predictor.load_state_dict(predictor_state)
        self.mixer.load_state_dict(
            torch.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage)
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(
            torch.load(f"{path}/opt.th", map_location=lambda storage, loc: storage)
        )
        self.predictor_optimiser.load_state_dict(
            torch.load(
                f"{path}/predictor_opt.th", map_location=lambda storage, loc: storage
            )
        )
        self._move_optimizer_state(self.optimiser, self.device)
        self._move_optimizer_state(self.predictor_optimiser, self.device)

    @staticmethod
    def _move_optimizer_state(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
