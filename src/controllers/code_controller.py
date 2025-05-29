"""
@File    : code_controller.py
@Created : 2025/05/24
@Author  : Loren
@Version : 1.0
@Desc    : Controller for paper "CoDe: Communication and Decision Making in Multi-Agent Reinforcement Learning"
@Link    : https://ui.adsabs.harvard.edu/abs/2025arXiv250105207S/abstract
"""

from types import SimpleNamespace
from typing import Any

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from utils.maker import AgentMaker, ActionSelectorMaker
from components.action_selectors.action_selector import ActionSelector
from modules.agents.code_agent import CodeAgent
from components.communication_model import CommunicationModel
from utils.custom_logging import PyMARLLogger
from components.episode_buffer import EpisodeBatch


class CodeMAC(MAC):
    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super(CodeMAC, self).__init__(scheme, groups, args)
        self.args: SimpleNamespace = args
        self.device: torch.device = args.device
        self.n_agents: int = args.n_agents

        self.input_shape = self._get_input_shape(scheme)
        self.agent_output_type: str = args.agent_output_type
        self.action_selector: ActionSelector = ActionSelectorMaker.make(args.action_selector, args)

        self._build_agents(self.input_shape)  # Initialize the agents.

        self.communication_model = CommunicationModel(args)

        self.hidden_states: torch.Tensor    # Agent hidden states.

    def select_actions(
            self, 
            ep_batch: EpisodeBatch, 
            t_ep: int, 
            t_env: int, 
            bs=slice(None), 
            test_mode: bool=False
        ) -> Any:
        # Slice available actions.  
        avail_actions = ep_batch["avail_actions"][:, t_ep]  # b * n * action_dim

        # MAC Forward.
        _, _, agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        agent_outputs = agent_outputs.squeeze(1).squeeze(-2)  # b * n * action_dim

        chosen_actions = self.action_selector.select_action(agent_outputs[bs], 
                                                            avail_actions[bs], 
                                                            t_env, 
                                                            test_mode=test_mode)   # b * n
        
        return chosen_actions
    
    def forward(
            self, 
            ep_batch: EpisodeBatch, 
            t: int | slice, 
            test_mode=False, 
            **kwargs
        ):
        #- 1. Prepare inputs.
        if isinstance(t, int):
            t = slice(t, t + 1)
        
        obs, last_actions, time_step_tensor, agent_id_tensor = self._build_inputs(ep_batch, t)  # b * t * n * 1 * d
        avail_actions: torch.Tensor = ep_batch["avail_actions"][:, t].unsqueeze(-2)  # b * t * n * 1 * d  # type: ignore 

        #- 2. Intent extraction.
        encoded_trajectory, intents, sent_messages, intents_mu, intents_std = self.agent.extract_intent(obs, 
                                                           self.hidden_states, 
                                                           last_actions, 
                                                           time_step_tensor, 
                                                           agent_id_tensor)  # b * t * n * 1 * d
        
        #- 3. Communication.
        received_messages = self.communication_model.process_communication(sent_messages, t)    # b * t * n * n-1 * d

        #- 4. Q value computation.
        attn_weights, action_values = self.agent.forward(sent_messages, received_messages)  # b * t * n * 1 * d

        # For politic action selection. Not tested yet.
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                action_values[avail_actions == 0] = -1e10

            action_values = torch.nn.functional.softmax(action_values, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = action_values.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()

                action_values = ((1 - self.action_selector.epsilon) * action_values  # type: ignore
                              + torch.ones_like(action_values) * self.action_selector.epsilon / epsilon_action_num)    # type: ignore

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    action_values[avail_actions == 0] = 0.0
        
        return encoded_trajectory, intents, action_values, intents_mu, intents_std, attn_weights

    def init_hidden(self, batch_size):
        # hidden_states for the shared CodeAgent, for each agent in the team
        # Shape: RNN_layers * (batch_size * n_agents), hidden_dim
        self.hidden_states = self.agent.init_hidden(batch_size).repeat(1, self.n_agents, 1)
        self.communication_model.reset(batch_size)  # Reset communication cache.

    def save_models(self, path): # For saving models
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path): # For loading models
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def load_state(self, agent_state_dict): # Changed from load_state_dict to avoid nn.Module conflict if not inheriting
        self.agent.load_state_dict(agent_state_dict)
    
    def _get_last_actions(self, batch, t: slice, batch_size, n_agents):
        """
        Return last actions of time slice t。

        Args:
            batch: PyMARL batch。
            t (slice): time slice。
            batch_size (int): batch size。
            n_agents (int): agent number。

        Returns:
            torch.Tensor: last actions of time slice t。
        """
        if t.start == 0:
            zeros = torch.zeros(batch_size, 1, n_agents, 1, device=self.device)
            sliced_actions = batch["actions"][:, slice(0, t.stop - 1)]
            last_actions = torch.cat([zeros, sliced_actions], dim=1).long()
        else:
            last_actions = batch["actions"][:, slice(t.start - 1, t.stop - 1)].long()

        return one_hot(last_actions, num_classes=self.args.n_actions)


    def _get_input_shape(self, scheme):
        # From scheme, determine the input shape for the agent's FC1 layer
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] 
        return input_shape

    def _build_agents(self, input_shape):
        self.agent: CodeAgent = AgentMaker.make(self.args.agent, input_shape, self.args)
        
    def _build_inputs(self, batch, t):
        batch_size, max_length, n_agents, obs_size = batch["obs"].shape
        obs_data = batch["obs"][:, t].unsqueeze(-2) # b * t * n_agents * 1 * obs_dim
        time_size = obs_data.shape[1]   # sequence length

        last_actions = self._get_last_actions(batch, t, batch_size, n_agents)   # b * t * n * 1 * n_actions
        time_indices = torch.arange(t.start, t.stop, device=self.device, dtype=torch.int)

        # time tensor shape b * t * n * 1 * 1
        time_step_tensor = time_indices.reshape(1, -1, 1, 1, 1).expand(batch_size, -1, self.n_agents, -1, -1)

        agent_id = torch.arange(self.n_agents, dtype=torch.long, device=batch.device)  # [1, 2, 3, 4, 5, ...]
        agent_id_one_hot = one_hot(agent_id, num_classes=self.n_agents  # n * 1 * n
                            ).reshape(1, 1, self.n_agents, 1, self.n_agents  # 1 * 1 * n * 1 * n
                            ).repeat(batch_size, time_size, 1, 1, 1)    # b * t * n_agents * 1 * n_agents

        if self.args.obs_agent_id:
            obs_data = torch.cat([obs_data, agent_id_one_hot], dim=-1)
        if self.args.obs_last_action:
            obs_data = torch.cat([obs_data, last_actions], dim=-1)

        return obs_data, last_actions, time_step_tensor, agent_id_one_hot
    