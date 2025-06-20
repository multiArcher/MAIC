import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, RMSprop, SGD  # type: ignore

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
    """
    CoDe Learner with comprehensive loss components and monitoring
    
    Loss Components:
    1. TD Loss: Standard Q-learning objective
    2. Action Loss: Future action prediction from intents  
    3. Continuity Loss: Intent temporal consistency
    4. Auxiliary Loss: KL divergence regularization
    5. Entropy Loss: Communication diversity encouragement
    """
    
    def __init__(self, mac: CodeMAC, scheme, logger, args):
        self.logger = PyMARLLogger("run")
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

        # Loss component weights with defaults
        self.td_loss_weight = getattr(args, 'td_loss_weight', 1.0)
        self.action_loss_weight = getattr(args, 'action_loss_weight', 1.0) 
        self.continue_loss_weight = getattr(args, 'continue_loss_weight', 0.1)
        self.aux_loss_weight = getattr(args, 'aux_loss_weight', 0.01)
        self.entropy_loss_weight = getattr(args, 'entropy_loss_weight', 0.01)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Enhanced training with comprehensive monitoring"""
        
        #-1. Prepare inputs.
        rewards = batch["reward"][:, :-1]  # Rewards for all agents
        actions = batch["actions"][:, :-1]
        actions_onehot = batch["actions_onehot"][:, :-1]  # type: ignore
        terminated = batch["terminated"][:, :-1].float()  # type: ignore
        mask = batch["filled"][:, :-1].float()  # type: ignore
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # Mask the final step.
        avail_actions: torch.Tensor = batch["avail_actions"]  # type: ignore
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
        t = slice(0, batch.max_seq_length)  # Exclude last timestep
        encoded_trajectory, intents, q_values, intent_mu, intent_std, attn_weights = self.mac.forward(batch, t)

        # Pick the Q values for the actions taken by each agent.
        choosen_action_values = torch.gather(q_values, dim=-1, index=actions.unsqueeze(-1))  # type: ignore b * t * n * 1 * 1
        
        # Mix
        joint_action_value = self.mixer(choosen_action_values, batch["state"][:, :-1, None, None])   # b * t * 1 * 1 * 1

        # Decode intents into actions.
        intent_actions = self.intent_decoder(
            observations=observations.unsqueeze(-2),    # type: ignore
            actions=actions_onehot,  # type: ignore
            encoded_trajectory=encoded_trajectory[:, :-1],  # Exclude last timestep
            intents=intents[:, :-1]
            )   # b * t * n * k+1 * n_actions

        #- 3. Target MAC forward.
        with torch.no_grad():
            self.target_mac.train()

            # MAC forward.
            self.target_mac.init_hidden(batch.batch_size)
            t_target = slice(0, batch.max_seq_length)  # Include last timestep for target calculation
            _, _, target_q_values, _, _, _ = self.target_mac.forward(batch, t_target)

            # Mask out unavailable actions
            target_q_values = torch.masked_fill(target_q_values, avail_actions.unsqueeze(-2)==0, -1e7)  # type: ignore

            # Max over target Q-Values
            if self.args.double_q is True:
                # current_q -> aa mask -> current actions -> current actions q values_from target net
                mac_out_detach = q_values.detach().clone()
                mac_out_detach = torch.masked_fill(mac_out_detach, avail_actions.unsqueeze(-2)==0, -1e7)
                cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]
                target_q_values = torch.gather(target_q_values, dim=-1, index=cur_max_actions)
            else:
                target_q_values, _ = target_q_values[:, 1:].max(dim=-1, keepdim=True)
            
            target_joint_action_value = self.target_mixer(target_q_values[:, 1:], batch["state"][:, 1:, None, None])
            
            if self.args.standardise_returns is True:
                target_joint_action_value = (
                    target_joint_action_value * torch.sqrt(self.ret_ms.var) + self.ret_ms.mean
                )

        #- 4. Calculate loss.
        # TD loss.
        with torch.no_grad():
            match target_type := getattr(self.args, "target_type", "td"):
                case "td":
                    td_targets = rewards[..., None, None] + self.args.gamma * target_joint_action_value * (1 - terminated[..., None, None])
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

        td_error = joint_action_value - td_targets.reshape_as(joint_action_value)  # b * t * 1 * 1 * 1
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
        action_error = F.cross_entropy(
            intent_actions.reshape(-1, self.n_actions),  # Flatten to b * t * n * k+1 * n_actions
            action_targets.reshape(-1, self.n_actions),  # Flatten to b * t * n * k+1 * n_actions
            reduction='none'
        ).reshape(
            *intent_actions.shape[:-1],  # b * t * n * k+1
            1
        )  # b * t * n * k+1 * 1

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
        action_loss = masked_action_error.sum() / combined_action_mask.sum()

        # Continuity loss.
        last_intents = F.pad(
            intents[:, :-1],  # Exclude last timestep
            (*(0, 0), *(0, 0), *(0, 0), *(1, 0)),
            mode='constant',
            value=0
        )

        cos_similarity = F.cosine_similarity(intents[:, :-1], last_intents[:, :-1].detach(), dim=-1, eps=1e-6)  # b * t-1 * n * 1
        continuity_mask = mask.squeeze(-1).expand_as(cos_similarity)  # b * t-1 * n * 1
        masked_continuity = cos_similarity * continuity_mask  # b * t-1 * n * 1
        continue_loss =  - masked_continuity.sum() / continuity_mask.sum()  # Mean cosine similarity

        # Auxiliary loss.
        aux_error = 0.5 * (intent_mu**2 + intent_std**2 - torch.log(intent_std**2 + 1e-6) - 1).sum(dim=-1, keepdim=True)  # b * t * n
        aux_mask = mask.expand_as(aux_error[:, :-1])  # b * t * n
        masked_aux_error = aux_error[:, :-1] * aux_mask  # b * t * n
        aux_loss = masked_aux_error.sum() / aux_mask.sum()  # Mean auxiliary loss

        # Intent alignment entropy regularization.
        intent_alignment_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-6), dim=-1)  # b * t * n
        mask_entropy = mask.squeeze(-1).expand_as(intent_alignment_entropy[:, :-1])  # b * t * n
        masked_intent_alignment_entropy = intent_alignment_entropy[:, :-1] * mask_entropy  # b * t * n
        intent_alignment_entropy_loss = masked_intent_alignment_entropy.sum() / mask_entropy.sum()  # Mean intent alignment entropy        

        # Enhanced loss calculation with weighted components
        total_loss = (self.td_loss_weight * td_loss + 
                     self.action_loss_weight * action_loss + 
                     self.continue_loss_weight * continue_loss + 
                     self.aux_loss_weight * aux_loss + 
                     self.entropy_loss_weight * intent_alignment_entropy_loss)

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

        # region Enhanced tensorboard logging
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            with torch.no_grad():            
                encoded_trajectory = encoded_trajectory[:, :-1]  # Exclude last timestep
                intents = intents[:, :-1]  # Exclude last timestep
                q_values = q_values[:, :-1]  # Exclude last timestep
                intent_mu = intent_mu[:, :-1]  # Exclude last timestep
                intent_std = intent_std[:, :-1]  # Exclude last timestep
                attn_weights = attn_weights[:, :-1]  # Exclude last timestep


                mask_elems = mask.sum().item()
                bool_mask_bt = mask.bool() # Create a boolean mask of shape (B, T)
                
                # === Core Training Metrics ===
                self.logger.log_stat("running/grad_norm", grad_norm.item(), t_env)
                
                # === Individual Loss Components ===
                self.logger.log_stat("loss/total_loss", total_loss.item(), t_env)
                self.logger.log_stat("loss/td_loss", td_loss.item(), t_env)
                self.logger.log_stat("loss/action_loss", action_loss.item(), t_env) 
                self.logger.log_stat("loss/continue_loss", continue_loss.item(), t_env)
                self.logger.log_stat("loss/aux_loss", aux_loss.item(), t_env)
                self.logger.log_stat("loss/entropy_loss", intent_alignment_entropy_loss.item(), t_env)
                
                # === Q-Value Statistics ===
                self.logger.log_stat("q_values/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
                self.logger.log_stat("q_values/q_taken_mean", (choosen_action_values * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
                self.logger.log_stat("q_values/target_mean", (td_targets * mask).sum().item() / mask_elems, t_env)
                
                # Q-value distribution statistics
                self.logger.log_stat("q_values/q_max_mean", q_values.max(dim=-1)[0].mean().item(), t_env)
                self.logger.log_stat("q_values/q_min_mean", q_values.min(dim=-1)[0].mean().item(), t_env) 
                self.logger.log_stat("q_values/q_std_mean", q_values.std(dim=-1).mean().item(), t_env)
                
                # === Intent Analysis ===
                # Intent magnitude and diversity
                masked_intents = intents * mask
                self.logger.log_stat("intent/intent_mean", masked_intents.sum().item() / (mask_elems * self.args.n_agents * self.args.intent_dim), t_env)
                valid_intents = intents[bool_mask_bt.expand_as(intents)] # Select only valid timesteps
                self.logger.log_stat("intent/intent_std", valid_intents.std().item(), t_env)

                # Intent diversity across agents (inter-agent variance)
                intent_var_across_agents = torch.var(intents, dim=2, correction=0)  # Variance across agents [B, T, 1, 1, I]
                masked_intent_var = (intent_var_across_agents.mean(dim=-1) * mask.squeeze(dim=-1).squeeze(dim=-1)).sum() / mask_elems
                self.logger.log_stat("intent/diversity_across_agents", masked_intent_var.item(), t_env)
                
                # Intent temporal consistency
                continuity_mean = masked_continuity.sum().item() / continuity_mask.sum().item()
                self.logger.log_stat("intent/temporal_consistency", continuity_mean, t_env)
                
                # Intent distribution parameters
                masked_mu = intent_mu * mask
                self.logger.log_stat("intent/mu_mean", masked_mu.sum().item() / (mask_elems * self.args.n_agents * self.args.intent_dim), t_env)
                valid_mu = intent_mu[bool_mask_bt.expand_as(intents)]
                self.logger.log_stat("intent/mu_std", valid_mu.std().item(), t_env)

                masked_std = intent_std * mask
                self.logger.log_stat("intent/std_mean", masked_std.sum().item() / (mask_elems * self.args.n_agents * self.args.intent_dim), t_env)
                valid_std = intent_std[bool_mask_bt.expand_as(intents)]
                self.logger.log_stat("intent/std_std", valid_std.std().item(), t_env)
                
                # === Communication Analysis ===
                # Attention entropy (communication diversity)
                attn_entropy_mean = masked_intent_alignment_entropy.sum().item() / (mask_elems * self.args.n_agents)
                self.logger.log_stat("communication/attention_entropy", attn_entropy_mean, t_env)
                
                # Attention concentration (max attention weight)
                attn_max = attn_weights.max(dim=-1)[0]  # [B, T, N, 1]
                masked_attn_max = attn_max * mask.squeeze(-1)
                self.logger.log_stat("communication/attention_max_mean", masked_attn_max.sum().item() / (mask_elems * self.args.n_agents), t_env)
                
                # Communication efficiency (how much attention is used)
                attn_top_k = torch.topk(attn_weights, k=int(self.n_agents / 2), dim=-1)[0]  # [B, T, N, K]
                masked_attn_top_k_sum = attn_top_k.sum(dim=-1) * mask.squeeze(-1)
                self.logger.log_stat("communication/attention_utilization", masked_attn_top_k_sum.sum().item() / (mask_elems * self.args.n_agents), t_env)
                
                # === Action Prediction Analysis ===
                # Action prediction accuracy
                decoded_ations = torch.argmax(intent_actions, dim=-1)  # [B, T, N, K+1]
                target_actions_idx = torch.argmax(action_targets, dim=-1)  # [B, T, N, K+1]
                action_accuracy = (decoded_ations == target_actions_idx).float()
                masked_accuracy = action_accuracy * combined_action_mask.any(dim=-1).float()
                
                if combined_action_mask.any(dim=-1).sum().item() > 0:
                    accuracy_mean = masked_accuracy.sum().item() / combined_action_mask.any(dim=-1).sum().item()
                    self.logger.log_stat("action_prediction/accuracy", accuracy_mean, t_env)
                    
                    # Per-step accuracy breakdown
                    for k in range(self.predict_k_future_actions + 1):
                        step_mask = combined_action_mask[:, :, :, k].any(dim=-1).float()
                        if step_mask.sum().item() > 0:
                            step_accuracy = (masked_accuracy[:, :, :, k] * step_mask).sum().item() / step_mask.sum().item()
                            self.logger.log_stat(f"action_prediction/accuracy_step_{k}", step_accuracy, t_env)
                
                # Action prediction confidence (entropy of predicted action distributions)
                action_probs = torch.softmax(intent_actions, dim=-1)
                action_entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
                masked_action_entropy = action_entropy * combined_action_mask.any(dim=-1).float()
                if combined_action_mask.any(dim=-1).sum().item() > 0:
                    entropy_mean = masked_action_entropy.sum().item() / combined_action_mask.any(dim=-1).sum().item()
                    self.logger.log_stat("action_prediction/confidence_entropy", entropy_mean, t_env)
                
                # === Training Dynamics ===
                # Learning rate (if using scheduler)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_stat("running/learning_rate", current_lr, t_env)
                
                # Target network update frequency
                self.logger.log_stat("running/target_update_step", self.last_target_update_step, t_env)
                
                # Memory usage
                if hasattr(torch.cuda, 'memory_allocated'):
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    self.logger.log_stat("system/gpu_memory_allocated_gb", memory_allocated, t_env)
                    self.logger.log_stat("system/gpu_memory_reserved_gb", memory_reserved, t_env)
                
                # === Loss Component Weights (for hyperparameter tracking) ===
                self.logger.log_stat("weights/td_loss_weight", self.td_loss_weight, t_env)
                self.logger.log_stat("weights/action_loss_weight", self.action_loss_weight, t_env)
                self.logger.log_stat("weights/continue_loss_weight", self.continue_loss_weight, t_env)
                self.logger.log_stat("weights/aux_loss_weight", self.aux_loss_weight, t_env)
                self.logger.log_stat("weights/entropy_loss_weight", self.entropy_loss_weight, t_env)
                
                self.log_stats_t = t_env
        # endregion

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
        """Enhanced model saving with intent decoder"""
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        # Save intent decoder
        torch.save(self.intent_decoder.state_dict(), "{}/intent_decoder.th".format(path))
        torch.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        """Enhanced model loading with intent decoder"""
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        # Load intent decoder with error handling
        try:
            self.intent_decoder.load_state_dict(
                torch.load("{}/intent_decoder.th".format(path), map_location=lambda storage, loc: storage)
            )
        except FileNotFoundError:
            self.logger.warning("Intent decoder weights not found, using random initialization")
        
        self.optimizer.load_state_dict(
            torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
