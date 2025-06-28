import torch
import torch.nn as nn

from modules.agents.agent import Agent
from utils.th_utils import get_parameters_num
from utils.custom_logging import PyMARLLogger


class KernelAgent(Agent):
    """Kernel Agent with QMIX for Multi-Agent Reinforcement Learning."""
    
    def __init__(self, input_shape, args):
        super(KernelAgent, self).__init__()
        self.args = args
        self.device = args.device
        
        # Network architecture hyperparameters
        self.gru_layers = args.agent_gru_layers
        self.agent_hidden_dim = args.agent_hidden_dim
        self.n_actions = args.n_actions

        # --- Core Agent Architecture ---
        self.activation = nn.LeakyReLU
        
        # 1. RNN for history encoding h_i^t
        self.rnn_projection = nn.Sequential(
            nn.Linear(input_shape, self.agent_hidden_dim),
            self.activation()
        )
        self.rnn = nn.GRU(
            self.agent_hidden_dim, 
            self.agent_hidden_dim, 
            num_layers=self.gru_layers, 
            batch_first=True
        )

        # 2. Q-value Network
        self.q_net = nn.Linear(self.agent_hidden_dim, self.n_actions)

        PyMARLLogger.fast_logger().info(f"Kernel Agent Size: {get_parameters_num(self.parameters())}")

    def init_hidden(self, batch_size=1):
        """Initialize GRU hidden state"""
        return torch.zeros(self.gru_layers, batch_size, self.agent_hidden_dim, device=self.device)

    def forward(self, obs, hidden_state):
        """
        Compute Q-values from observations and hidden state.
        
        Process:
        1. Project observations to hidden dimension.
        2. Encode sequence through GRU to get current hidden state.
        3. Compute Q-values from the GRU's output.
        """
        batch_size, time_size, n_agents, _, _ = obs.shape

        # 1. Observation Encoding
        x = self.rnn_projection(obs)
        
        # Reshape for GRU: (B*N, T, H)
        x_reshaped = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.agent_hidden_dim)
        rnn_output, new_hidden_state = self.rnn(x_reshaped, hidden_state)
        
        # Reshape back: (B, T, N, 1, H)
        rnn_output_reshaped = rnn_output.reshape(batch_size, n_agents, time_size, 1, -1).transpose(1, 2)

        # 2. Q-value Computation
        q_values = self.q_net(rnn_output_reshaped)
        
        return q_values, new_hidden_state
