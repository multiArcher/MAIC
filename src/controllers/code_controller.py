"""
@File    : code_controller.py
@Created : 2025/05/24
@Author  : Loren
@Version : 1.0
@Desc    : Controller for paper "CoDe: Communication and Decision Making in Multi-Agent Reinforcement Learning"
@Link    : https://ui.adsabs.harvard.edu/abs/2025arXiv250105207S/abstract
"""

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch.nn.functional import one_hot

from .mac import MAC
from utils.maker import AgentMaker, ActionSelectorMaker
from components.action_selectors.action_selector import ActionSelector
from modules.agents.code_agent import CodeAgent
from components.communication_model import CommunicationModel
from components.episode_buffer import EpisodeBatch


class CodeMAC(MAC):
    """
    Multi-Agent Controller for CoDe Algorithm
    
    This controller orchestrates the CoDe multi-agent system:
    1. Manages agent interactions and communication
    2. Coordinates intent extraction and message passing
    3. Handles action selection for all agents
    4. Processes observations and builds agent inputs
    """
    
    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super(CodeMAC, self).__init__(scheme, groups, args)
        self.args: SimpleNamespace = args
        self.device: torch.device = args.device
        self.n_agents: int = args.n_agents

        # Determine input shape for agents based on observation scheme
        self.input_shape = self._get_input_shape(scheme)
        self.agent_output_type: str = args.agent_output_type
        
        # Initialize action selector (e.g., epsilon-greedy)
        self.action_selector: ActionSelector = ActionSelectorMaker.make(args.action_selector, args)

        # Build the shared CodeAgent that all agents use
        self._build_agents(self.input_shape)

        # Initialize communication model for message passing between agents
        self.communication_model = CommunicationModel(
            args, 
            delay_mean=args.comm_gaussian_delay_mean, 
            delay_std=args.comm_gaussian_delay_std
            )

        # Storage for agent hidden states across timesteps
        self.hidden_states: torch.Tensor


    def select_actions(
            self, 
            ep_batch: EpisodeBatch, 
            t_ep: int, 
            t_env: int, 
            bs=slice(None), 
            test_mode: bool = False
        ) -> Any:
        """
        Select actions for all agents at given timestep
        
        Args:
            ep_batch: Episode batch containing observations and history
            t_ep: Current episode timestep
            t_env: Current environment timestep (for exploration scheduling)
            bs: Batch slice selection
            test_mode: Whether in evaluation mode
            
        Returns:
            chosen_actions: Selected actions for each agent [B, N]
        """
        # Extract available actions for current timestep
        avail_actions = ep_batch["avail_actions"][:, t_ep]  # [B, N, A]

        # Forward pass through MAC to get Q-values or policy logits
        _, _, agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        agent_outputs = agent_outputs.squeeze(1).squeeze(-2)  # [B, N, A]

        # Select actions using the action selector (e.g., epsilon-greedy for Q-values)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], 
            avail_actions[bs], 
            t_env, 
            test_mode=test_mode
        )   # [B, N]
        
        return chosen_actions
    
    def forward(
            self, 
            ep_batch: EpisodeBatch, 
            t: int | slice, 
            test_mode=False, 
            **kwargs
        ):
        """
        Forward pass through the CoDe multi-agent system
        
        This implements the complete CoDe forward pass:
        1. Build inputs from observations, actions, and agent IDs
        2. Extract intents using variational encoding
        3. Process communication with delay modeling
        4. Compute Q-values using fused communication context
        
        Args:
            ep_batch: Episode batch containing observations and history
            t: Timestep(s) to process (int for single step, slice for sequence)
            test_mode: Whether in evaluation mode
            
        Returns:
            Tuple of (encoded_trajectory, intents, action_values, intents_mu, intents_std, attn_weights)
        """
        
        # 1. Prepare inputs for the given timestep(s)
        if isinstance(t, int):
            t = slice(t, t + 1)
        
        # Build structured inputs: observations, actions, timestamps, agent IDs
        obs, last_actions, time_step_tensor, agent_id_tensor = self._build_inputs(ep_batch, t)
        avail_actions: torch.Tensor = cast(torch.Tensor, ep_batch["avail_actions"][:, t]).unsqueeze(-2)  # [B, T, N, 1, A]

        # 2. Intent Extraction Phase
        # Each agent extracts its intent from observations and previous actions
        encoded_trajectory, self.hidden_states, intents, sent_messages, intents_mu, intents_std = self.agent.extract_intent(
            obs,                    # Current observations
            self.hidden_states,     # Previous hidden states
            last_actions,           # Previous actions (one-hot)
            time_step_tensor,       # Current timestamps
            agent_id_tensor         # Agent identities
        )

        # 3. Communication Phase
        # Process message passing with delay modeling and topology constraints
        received_messages = self.communication_model.process_communication(sent_messages, t, self.training)

        # 4. Q-value Computation Phase
        # Use dual alignment attention to fuse messages and compute action values
        attn_weights, action_values = self.agent.forward(sent_messages, received_messages)

        # 5. Handle different output types (Q-values vs policy logits)
        if self.agent_output_type == "pi_logits":
            # For policy-based methods, convert logits to probabilities
            if getattr(self.args, "mask_before_softmax", True):
                # Mask unavailable actions before softmax to prevent selection
                action_values[avail_actions == 0] = -1e10

            action_values = torch.nn.functional.softmax(action_values, dim=-1)
            
            if not test_mode:
                # Apply epsilon-greedy exploration during training
                epsilon_action_num = action_values.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # Only consider available actions for uniform exploration
                    epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()

                # Mix policy probabilities with uniform exploration
                action_values = (
                    (1 - self.action_selector.epsilon) * action_values +
                    torch.ones_like(action_values) * self.action_selector.epsilon / epsilon_action_num
                )

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out unavailable actions after exploration mixing
                    action_values[avail_actions == 0] = 0.0
        
        return encoded_trajectory, intents, action_values, intents_mu, intents_std, attn_weights

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for all agents
        
        Args:
            batch_size: Batch size for initialization
        """
        # Create hidden states for all agents using the shared CodeAgent
        # Shape: (num_layers, batch_size * n_agents, hidden_dim)
        self.hidden_states = self.agent.init_hidden(batch_size).repeat(1, self.n_agents, 1)
        
        # Reset communication model cache for new episodes
        self.communication_model.reset(batch_size)

    def save_models(self, path):
        """Save agent models to specified path"""
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """Load agent models from specified path"""
        self.agent.load_state_dict(
            torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def load_state(self, other_mac):
        """Load state from another MAC instance (for target networks)"""
        self.agent.load_state_dict(other_mac.agent.state_dict())
    
    def _get_last_actions(self, batch, t: slice, batch_size, n_agents):
        """
        Extract last actions for the given time slice with proper padding
        
        Args:
            batch: Episode batch
            t: Time slice to process
            batch_size: Batch size
            n_agents: Number of agents
            
        Returns:
            last_actions: One-hot encoded last actions [B, T, N, 1, A]
        """
        if t.start == 0:
            # For the first timestep, there are no previous actions
            # Pad with zeros at the beginning
            zeros = torch.zeros(batch_size, 1, n_agents, 1, self.args.n_actions, device=self.device)
            sliced_actions = batch["actions"][:, slice(0, t.stop - 1)]
            sliced_actions = one_hot(sliced_actions, num_classes=self.args.n_actions)
            last_actions = torch.cat([zeros, sliced_actions], dim=1).long()
        else:
            # Extract actions from previous timesteps
            last_actions = batch["actions"][:, slice(t.start - 1, t.stop - 1)].long()
            last_actions = one_hot(last_actions, num_classes=self.args.n_actions)

        # Convert to one-hot encoding for neural network input
        return last_actions

    def _get_input_shape(self, scheme):
        """
        Calculate total input shape for the agent based on observation scheme
        
        Args:
            scheme: Data scheme from environment
            
        Returns:
            input_shape: Total input dimension for agent networks
        """
        # Start with base observation shape
        input_shape = scheme["obs"]["vshape"]
        
        # Add agent ID dimension if enabled (one-hot encoding of agent identity)
        if self.args.obs_agent_id:
            input_shape += self.n_agents
            
        # Add last action dimension if enabled (one-hot encoding of previous action)
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] 
            
        return input_shape

    def _build_agents(self, input_shape):
        """
        Build the shared CodeAgent used by all agents in the team
        
        Args:
            input_shape: Input dimension for the agent networks
        """
        self.agent: CodeAgent = AgentMaker.make(self.args.agent, input_shape, self.args)
        
    def _build_inputs(self, batch, t):
        """
        Build structured input tensors for the agents from batch data
        
        This function processes raw episode data into the structured format
        required by the CoDe agents, including:
        - Observations with optional agent ID and last action
        - Last actions (one-hot encoded)
        - Timestamps for temporal alignment
        - Agent identity tensors
        
        Args:
            batch: Episode batch from replay buffer
            t: Time slice to process
            
        Returns:
            Tuple of (observations, last_actions, time_step_tensor, agent_id_tensor)
        """        
        # Extract observations and add sequence dimension for consistency
        obs_data = batch["obs"][:, t].unsqueeze(-2)  # [B, T, N, 1, D]
        batch_size, time_size, n_agents, _, _ = obs_data.shape

        # Get last actions with proper padding and one-hot encoding
        last_actions = self._get_last_actions(batch, t, batch_size, n_agents)  # [B, T, N, 1, A]
        
        # Create timestamp tensors for temporal alignment in communication
        time_indices = torch.arange(t.start, t.stop, device=self.device, dtype=torch.int)
        time_step_tensor = time_indices.reshape(1, -1, 1, 1, 1).expand(
            batch_size, -1, self.n_agents, -1, -1
        )  # [B, T, N, 1, 1]

        # Create agent identity tensors (one-hot encoded agent IDs)
        agent_id = torch.arange(self.n_agents, dtype=torch.long, device=batch.device)
        agent_id_one_hot = one_hot(agent_id, num_classes=self.n_agents).reshape(
            1, 1, self.n_agents, 1, self.n_agents
        ).repeat(batch_size, time_size, 1, 1, 1)  # [B, T, N, 1, N]

        # Optionally concatenate additional features to observations
        if self.args.obs_agent_id:
            # Include agent identity in observations for agent-specific processing
            obs_data = torch.cat([obs_data, agent_id_one_hot], dim=-1)
        if self.args.obs_last_action:
            # Include last action in observations for action-conditional processing
            obs_data = torch.cat([obs_data, last_actions], dim=-1)

        return obs_data, last_actions, time_step_tensor, agent_id_one_hot
