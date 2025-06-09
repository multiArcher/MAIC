import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot


class IntentDecoder(nn.Module):
    """
    Intent Decoder for Future Action Prediction
    
    This module predicts a sequence of future actions based on:
    1. Current and future observations
    2. Agent intents
    3. Historical trajectory encoding
    
    Architecture:
    - Separate GRU for each prediction step (current + k future)
    - MLP for action prediction from intent + hidden state
    """
    
    def __init__(self, args, input_shape):
        super().__init__()
        # Configuration parameters
        self.predict_k_future_actions = args.predict_k_future_actions
        self.gru_layers = args.agent_gru_layers
        self.intent_dim = args.intent_dim
        self.n_actions = args.n_actions
        self.agent_hidden_dim = args.agent_hidden_dim
        self.batch_size = args.batch_size
        self.device = args.device
        self.input_shape = input_shape

        # Create GRU modules for each prediction step
        # Each GRU processes (observation + action) sequences
        self.gru =nn.GRU(
                self.input_shape + self.n_actions,  # obs_dim + n_actions
                self.agent_hidden_dim, 
                self.gru_layers, 
                batch_first=True, 
                device=self.device
            )         
        
        # MLP modules for action prediction from (intent + hidden_state)
        self.mlp = nn.Linear(
                self.intent_dim + self.agent_hidden_dim, 
                self.n_actions, 
                device=self.device
            )

    def forward(
            self, 
            observations: torch.Tensor,  # b * t * n * 1 * d - agent observations
            actions: torch.Tensor,       # b * t-1 * n * 1 * n_actions - past actions (one-hot)
            encoded_trajectory: torch.Tensor,  # b * t * n * 1 * d - encoded agent trajectories
            intents: torch.Tensor        # b * t * n * 1 * d - agent intents
    ):
        """
        Predict action sequences using intents and observations
        
        For each prediction step k:
        1. Use appropriate observation and action history
        2. Process through step-specific GRU  
        3. Combine with intent to predict action probabilities
        4. Use predicted action for next step's input
        
        Returns:
            predicted_actions: [B, T, N, K+1, A] - action probabilities for current + k future steps
        """
        target_shape = observations.shape[:-1]  # [B, T, N, 1]
        batch_size, time_size, n_agents, _ = target_shape

        initial_last_actions = F.pad(
            actions.reshape(*target_shape, self.n_actions)[:, :-1], 
            (*(0, 0), *(0, 0), *(0, 0), *(1, 0)),   # pad at beginning of time dimension
            value=0.0
        )

        # Initialize trajectory estimation storage
        estimated_actions_probs = []
        last_actions_for_gru = initial_last_actions  # Start with padded last actions
        hidden_state = encoded_trajectory.detach()  # [b, t, n, 1, d] - encoded trajectory
        hidden_state = hidden_state.transpose(1, 2).reshape(1, batch_size * n_agents * time_size, -1)

        # Process each prediction step
        for idx in range(self.predict_k_future_actions + 1):
            current_obs = F.pad(
                observations[:, idx:],
                (*(0, 0), *(0, 0), *(0, 0), *(0, idx)),  # pad at ending
                value=0.0
            )

            gru_inputs = torch.cat([current_obs, last_actions_for_gru], dim=-1)

            # Reshape for GRU processing: (B*N*T, 1, input_dim)
            gru_inputs = gru_inputs.transpose(1, 2).reshape(batch_size * n_agents * time_size, 1, -1)
            
            # GRU forward pass
            estimated_trajectory, hidden_state = self.gru(gru_inputs, hidden_state)

            # Reshape back to original format
            estimated_trajectory = estimated_trajectory.reshape(batch_size, n_agents, time_size, 1, -1).transpose(1, 2)

            # Predict action probabilities from intent + hidden state
            estimated_actions_prob = self.mlp(torch.cat([intents, estimated_trajectory], dim=-1))
            estimated_actions_probs.append(estimated_actions_prob)

            # Convert probabilities to one-hot for next iteration
            last_actions_for_gru = one_hot(
                torch.argmax(estimated_actions_prob, dim=-1), 
                num_classes=self.n_actions
            )

        # Combine all predicted action probabilities
        estimated_actions_probs = torch.cat(estimated_actions_probs, dim=-2)  # [B, T, N, K+1, A]

        return estimated_actions_probs
