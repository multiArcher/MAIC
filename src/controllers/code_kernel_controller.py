from types import SimpleNamespace
from typing import Any

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from utils.maker import AgentMaker, ActionSelectorMaker
from components.action_selectors.action_selector import ActionSelector
from modules.agents.code_kernel_agent import CodeKernelAgent
from components.episode_buffer import EpisodeBatch


class CodeKernelMAC(MAC):
    """
    Multi-Agent Controller for the Kernel (QMIX-like) Algorithm.
    
    This controller orchestrates a basic multi-agent system:
    1. Manages agent action selection.
    2. Processes observations and builds agent inputs.
    """
    
    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super(CodeKernelMAC, self).__init__(scheme, groups, args)
        self.args: SimpleNamespace = args
        self.device: torch.device = args.device
        self.n_agents: int = args.n_agents

        self.input_shape = self._get_input_shape(scheme)
        self.agent_output_type: str = args.agent_output_type
        
        self.action_selector: ActionSelector = ActionSelectorMaker.make(args.action_selector, args)
        self._build_agents(self.input_shape)

        self.hidden_states: torch.Tensor

    def select_actions(
            self, 
            ep_batch: EpisodeBatch, 
            t_ep: int, 
            t_env: int, 
            bs=slice(None), 
            test_mode: bool = False
        ) -> Any:
        """Select actions for all agents at a given timestep."""
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        agent_outputs = agent_outputs.squeeze(1).squeeze(-2)

        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], 
            avail_actions[bs], 
            t_env, 
            test_mode=test_mode
        )
        return chosen_actions
    
    def forward(
            self, 
            ep_batch: EpisodeBatch, 
            t: int | slice, 
            test_mode=False, 
            **kwargs
        ):
        """Forward pass for the kernel MAC."""
        if isinstance(t, int):
            t = slice(t, t + 1)
        
        obs, _, _, _ = self._build_inputs(ep_batch, t)
        avail_actions: torch.Tensor = ep_batch["avail_actions"][:, t].unsqueeze(-2)

        action_values, self.hidden_states = self.agent(obs, self.hidden_states)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                action_values[avail_actions == 0] = -1e10
            action_values = torch.nn.functional.softmax(action_values, dim=-1)
            if not test_mode:
                epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()
                action_values = (
                    (1 - self.action_selector.epsilon) * action_values +
                    torch.ones_like(action_values) * self.action_selector.epsilon / epsilon_action_num
                )
                action_values[avail_actions == 0] = 0.0
        
        return action_values

    def init_hidden(self, batch_size):
        """Initialize hidden states for all agents."""
        self.hidden_states = self.agent.init_hidden(batch_size).repeat(1, self.n_agents, 1)

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
    
    def _get_last_actions(self, batch, t: slice, batch_size, n_agents):
        if t.start == 0:
            zeros = torch.zeros(batch_size, 1, n_agents, 1, self.args.n_actions, device=self.device)
            sliced_actions = batch["actions"][:, slice(0, t.stop - 1)]
            sliced_actions = one_hot(sliced_actions, num_classes=self.args.n_actions)
            last_actions = torch.cat([zeros, sliced_actions], dim=1).long()
        else:
            last_actions = batch["actions"][:, slice(t.start - 1, t.stop - 1)].long()
            last_actions = one_hot(last_actions, num_classes=self.args.n_actions)
        return last_actions

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] 
        return input_shape

    def _build_agents(self, input_shape):
        self.agent: CodeKernelAgent = AgentMaker.make(self.args.agent, input_shape, self.args)
        
    def _build_inputs(self, batch, t):
        batch_size, _, n_agents, _ = batch["obs"].shape
        obs_data = batch["obs"][:, t].unsqueeze(-2)
        time_size = obs_data.shape[1]

        last_actions = self._get_last_actions(batch, t, batch_size, n_agents)
        
        time_indices = torch.arange(t.start, t.stop, device=self.device, dtype=torch.int)
        time_step_tensor = time_indices.reshape(1, -1, 1, 1, 1).expand(
            batch_size, -1, self.n_agents, -1, -1
        )

        agent_id = torch.arange(self.n_agents, dtype=torch.long, device=batch.device)
        agent_id_one_hot = one_hot(agent_id, num_classes=self.n_agents).reshape(
            1, 1, self.n_agents, 1, self.n_agents
        ).repeat(batch_size, time_size, 1, 1, 1)

        if self.args.obs_agent_id:
            obs_data = torch.cat([obs_data, agent_id_one_hot], dim=-1)
        if self.args.obs_last_action:
            obs_data = torch.cat([obs_data, last_actions], dim=-1)

        return obs_data, last_actions, time_step_tensor, agent_id_one_hot
