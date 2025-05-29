import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, RMSprop, SGD # type: ignore

from utils.maker import MixerMaker
from learners.learner import Learner
from components.episode_buffer import EpisodeBatch

from controllers.code_controller import CodeMAC
from utils.custom_logging import PyMARLLogger
from utils.th_utils import get_parameters_num
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_q_lambda_targets, new_build_td_lambda_targets
from modules.layers import IntentDecoder


class CodeLearner(Learner):
    def __init__(self, mac: CodeMAC, scheme, logger, args):
        self.logger = PyMARLLogger("main").get_child_logger(f"{self.__class__.__name__}")
        super(CodeLearner, self).__init__()
        
        self.args = args
        self.device = args.device
        self.mac = mac # This is an instance of CodeMAC, which contains the CodeAgent
        self.scheme = scheme
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.predict_k_future_actions = args.predict_k_future_actions

        self.params = list(mac.parameters()) # Parameters of CodeAgent
        self.last_target_update_episode = 0
        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1 # For logging

        if args.mixer is not None:
            self.mixer = MixerMaker.make(args.mixer, args)
        else: # VDN like, sum of individual Qs
            self.mixer = torch.nn.Identity()
        self.params += list(self.mixer.parameters())

        # Intent decoder.
        self.intent_decoder = IntentDecoder(args, self.mac.input_shape)
        self.params += list(self.intent_decoder.parameters())

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
            case _:
                raise ValueError(f"Optimizer {args.optimizer} not recognized.")

        # Target MAC (contains target CodeAgent)
        self.target_mac = copy.deepcopy(mac) # This deepcopies the CodeAgent inside MAC

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        #-1. Prepare inputs.
        rewards = batch["reward"][:, :-1]  # Rewards for all agents
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]  # type: ignore
        terminated = batch["terminated"][:, :-1].float()  # type: ignore
        mask = batch["filled"][:, :-1].float()  # type: ignore
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # Mask the final step.
        avail_actions: torch.Tensor = batch["avail_actions"][:, :-1]  # type: ignore
        observations = batch["obs"][:, :-1]  # type: ignore

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1  # type: ignore
            ), "Expected singular agent dimension for common rewards"

        #- 2. Online MAC forward.
        self.mac.agent.train()
        self.mac.init_hidden(batch.batch_size)
        t = slice(0, batch.max_seq_length-1)  # Exclude last timestep
        encoded_trajectory, intents, q_values, intent_mu, intent_std, attn_weights = self.mac.forward(batch, t)

        # Pick the Q values for the actions taken by each agent.
        choosen_action_values = torch.gather(q_values, dim=-1, index=actions.unsqueeze(-1))  # type: ignore b * t * n * 1 * 1
        
        # Mix
        joint_action_value = self.mixer(choosen_action_values, batch["state"][:, :-1, None, None])   # b * t * 1 * 1 * 1

        # Decode intents into actions.
        intent_actions = self.intent_decoder(
            observations=observations.unsqueeze(-2),    # type: ignore
            actions=actions_onehot,  # type: ignore
            encoded_trajectory=encoded_trajectory,
            intents=intents
            )   # b * t * n * k+1 * n_actions

        #- 3. Target MAC forward.
        with torch.no_grad():
            self.target_mac.train()

            # MAC forward.
            self.target_mac.init_hidden(batch.batch_size)
            t_target = slice(0, batch.max_seq_length)  # Include last timestep for target calculation
            _, _, target_q_values, _, _, _ = self.target_mac.forward(batch, t_target)

            # Mask out unavailable actions
            target_q_values = torch.masked_fill(target_q_values, batch["avail_actions"].unsqueeze(-2)==0, -1e7)  # type: ignore

            # Max over target Q-Values
            if self.args.double_q is True:
                # current_q -> aa mask -> current actions -> current actions q values_from target net
                mac_out_detach = q_values.detach().clone()
                mac_out_detach = torch.masked_fill(mac_out_detach, avail_actions.unsqueeze(-2)==0, -1e7)
                cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]
                target_q_values = torch.gather(target_q_values[:, 1:], dim=-1, index=cur_max_actions)
            else:
                target_q_values = target_q_values[:, 1:].max(dim=-1)[0]
            
            target_joint_action_value = self.target_mixer(target_q_values, batch["state"][:, 1:, None, None])
            
            if self.args.standardise_returns is True:
                target_joint_action_value = (
                    target_joint_action_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
                )

        #- 4. Calculate loss.
        # TD loss.
        with torch.no_grad():
            match target_type := getattr(self.args, "target_type", "td"):
                case "td":
                    td_targets = rewards + self.args.gamma * (1 - terminated) * target_joint_action_value.detach()
                case "td_lambda":
                    td_targets = new_build_td_lambda_targets(rewards, terminated, mask, target_joint_action_value,
                                                      self.args.gamma, self.args.td_lambda)
                case "q_lambda":
                    qvals = torch.gather(target_q_values[:, 1:], -1, actions.unsqueeze(-1)).squeeze(-1)  # type: ignore
                    qvals = self.target_mixer(qvals, batch["state"][:, 1:, None, None])
                    td_targets = build_q_lambda_targets(rewards, terminated, mask, target_joint_action_value, qvals,
                                                     self.args.gamma, self.args.td_lambda)
                case _:
                    raise ValueError(f"Invalid target type {target_type}")
            
            if self.args.standardise_returns:
                self.ret_ms.update(td_targets)
                td_targets = (td_targets - self.ret_ms.mean) / torch.sqrt(self.ret_ms.var)
            
        mask = mask[..., None, None]  # Mask for terminated steps.

        td_error = joint_action_value - td_targets[..., None, None].detach()
        masked_td_error = td_error * mask
        td_loss = (masked_td_error**2).sum() / mask.sum()

        # Inference loss.
        action_targets = [
            F.pad(
                actions_onehot[:, idx:],    # type: ignore
                (*(0, 0), *(0, 0), *(0, idx)),     # pad last actions at step -1 along dim 1.
                value=0
                )  # b * t * n * 1 * n_actions
            for idx in range(self.predict_k_future_actions+1)   # 0, 1, ..., k
        ]
        action_targets = torch.stack(action_targets, dim=-2)  # b * t * n * k+1 * n_actions
        action_error = intent_actions - action_targets  # b * t * n * k+1 * n_actions
        
        # Create triangular mask for future action prediction validity
        batch_size, time_size, n_agents = mask.shape[:3]
        k_steps = self.predict_k_future_actions + 1
        
        # Create triangular mask: valid when t + k < time_size
        # Shape: (time_size, k_steps)
        time_indices = torch.arange(time_size, device=mask.device).unsqueeze(1)  # t, 1
        future_steps = torch.arange(k_steps, device=mask.device).unsqueeze(0)   # 1, k+1
        action_validity_mask = (future_steps + time_indices < time_size).float()  # t, k+1

        # Expand to match action_error dimensions: b * t * n * k+1 * n_actions
        action_validity_mask = action_validity_mask.unsqueeze(0).unsqueeze(2).unsqueeze(4).expand(
            batch_size, time_size, n_agents, k_steps, action_error.shape[-1]
        )
        
        # Combine with original episode mask
        episode_mask_expanded = mask.expand_as(action_error)
        combined_action_mask = episode_mask_expanded * action_validity_mask
        
        masked_action_error = action_error * combined_action_mask
        action_loss = (masked_action_error**2).sum() / combined_action_mask.sum()

        # Continuity loss.
        cos_similarity = F.cosine_similarity(intents[:, 1:], intents[:, :-1], dim=-1, eps=1e-6)  # b * t-1 * n * 1
        mask_continuity = mask[:, 1:].squeeze(-1).expand_as(cos_similarity)  # b * t-1 * n * 1
        masked_continuity = cos_similarity * mask_continuity  # b * t-1 * n * 1
        continue_loss =  - masked_continuity.sum() / mask_continuity.sum()  # Mean cosine similarity

        # Auxiliary loss.
        aux_error = -0.5 * (intent_mu**2 + intent_std**2 - torch.log(intent_std**2) - 1).sum(dim=-1)  # b * t * n
        mask_aux = mask.squeeze(-1).expand_as(aux_error)  # b * t * n
        masked_aux_error = aux_error * mask_aux  # b * t * n
        aux_loss = masked_aux_error.sum() / mask_aux.sum()  # Mean auxiliary loss

        # Intent alignment entropy regularization.
        intent_alignment_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-6), dim=-1)  # b * t * n
        mask_entropy = mask.squeeze(-1).expand_as(intent_alignment_entropy)  # b * t * n
        masked_intent_alignment_entropy = intent_alignment_entropy * mask_entropy  # b * t * n
        intent_alignment_entropy_loss = masked_intent_alignment_entropy.sum() / mask_entropy.sum()  # Mean intent alignment entropy        

        total_loss = td_loss + action_loss + continue_loss + aux_loss + intent_alignment_entropy_loss

        # Optimize
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
            self.logger.log_stat("total_loss", total_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (choosen_action_values * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (td_targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
        
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
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimizer.load_state_dict(
            torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
