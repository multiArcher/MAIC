import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot


class IntentDecoder(nn.Module):
    def __init__(self, args, input_shape):
        super().__init__()
        self.predict_k_future_actions = args.predict_k_future_actions
        self.gru_layers = args.agent_gru_layers
        self.intent_dim = args.intent_dim
        self.n_actions = args.n_actions
        self.agent_hidden_dim = args.agent_hidden_dim
        self.batch_size = args.batch_size
        self.device = args.device
        self.input_shape = input_shape

        self.gru_modules = nn.ModuleList([
            nn.GRU(self.input_shape + self.n_actions, self.agent_hidden_dim, self.gru_layers, batch_first=True, device=self.device) 
            for _ in range(self.predict_k_future_actions+1)     # current step + k future steps
            ])
        self.mlp_modules = nn.ModuleList([
            nn.Linear(self.intent_dim + self.agent_hidden_dim, self.n_actions, device=self.device)
            for _ in range(self.predict_k_future_actions+1)
            ])



    def forward(
            self, 
            observations: torch.Tensor,  # b * t * n * 1 * d
            actions: torch.Tensor, # b * t-1 * n * 1 * n_actions
            encoded_trajectory: torch.Tensor,  # b * t * n * 1 * d
            intents: torch.Tensor  # b * t * n * 1 * d
            ):
        target_shape = observations.shape[:-1]
        batch_size, time_size, n_agents, _ = target_shape

        estimated_trajectory = torch.zeros_like(encoded_trajectory, device=self.device)  # b * t * n * 1 * d
        estimated_actions_probs = []

        for idx, (gru, mlp) in enumerate(zip(self.gru_modules, self.mlp_modules)):
            if idx == 0:
                hidden_state = encoded_trajectory.detach()
                last_actions = F.pad(
                    actions.reshape(*target_shape, self.n_actions)[:, :-1], 
                    (*(0, 0), *(0, 0), *(0, 0), *(1, 0)),   # pad last actions at step 0
                    value=0.0
                    )
                gru_inputs = torch.cat([observations, last_actions], dim=-1)
            else:
                hidden_state = estimated_trajectory
                obs_vector = F.pad(observations[:, idx:], (*(0, 0), *(0, 0), *(0, 0), *(0, idx)), value=0.0)
                gru_inputs = torch.cat([obs_vector, estimated_action], dim=-1)

            gru_inputs = gru_inputs.transpose(1, 2).reshape(batch_size * n_agents * time_size, 1, -1)
            hidden_state = hidden_state.transpose(1, 2).reshape(1, batch_size * n_agents * time_size, -1)
            
            estimated_trajectory, hidden_state = gru(gru_inputs, hidden_state)

            estimated_trajectory = estimated_trajectory.reshape(batch_size, n_agents, time_size, 1, -1).transpose(1, 2)  # b * t * n * 1 * d
            hidden_state = hidden_state.reshape(batch_size, n_agents, time_size, 1, -1).transpose(1, 2)

            estimated_actions_prob = mlp(torch.cat([intents, estimated_trajectory], dim=-1))  # b * t * n * 1 * n_actions
            estimated_actions_probs.append(estimated_actions_prob)

            estimated_action = one_hot(torch.argmax(estimated_actions_prob, dim=-1), num_classes=self.n_actions)  # b * t * n * 1 * n_actions

        estimated_actions_probs = torch.cat(estimated_actions_probs, dim=-2)  # b * t * n * k * n_actions

        return estimated_actions_probs
