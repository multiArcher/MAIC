import torch
import torch.nn as nn
from torch.distributions import Normal

from modules.agents.agent import Agent
from modules.layers import RMSNorm
from components.communication_model import CoDeBatchedMessageData
from utils.th_utils import get_parameters_num
from utils.custom_logging import PyMARLLogger


class CodeAgent(Agent):
    """
    CoDe Agent implementing Communication and Decision Making
    
    Core components:
    1. GRU-based history encoding for trajectory representation
    2. Variational intent extraction with reparameterization trick
    3. Dual alignment attention for message fusion
    4. Q-value estimation with fused communication context
    """
    
    def __init__(self, input_shape, args):
        super(CodeAgent, self).__init__()
        # Storage configs.
        self.args = args
        self.device = args.device
        
        # Network architecture hyperparameters
        self.gru_layers = args.agent_gru_layers
        self.agent_hidden_dim = args.agent_hidden_dim
        self.intent_dim = args.intent_dim
        self.intent_encoder_hidden_dim = args.intent_encoder_hidden_dim
        self.intent_decoder_gru_hidden_dim = args.intent_decoder_gru_hidden_dim
        self.dual_alignment_attn_dim = args.dual_alignment_attn_dim
        self.predict_k_future_actions = args.predict_k_future_actions
        self.intent_action_predictor_hidden_sizes = args.intent_action_predictor_hidden_sizes
        self.temporal_discount_gamma_T = args.temporal_discount_gamma_T

        # Environment parameters
        self.n_actions = args.n_actions
        
        # Message structure: [agent_id + intent + hidden_state + timestamp]
        self.message_dim = self.args.n_agents + self.intent_dim + self.agent_hidden_dim + 1

        # --- Core Agent Architecture ---
        self.activation = nn.LeakyReLU()
        
        # 1. RNN for history encoding h_i^t
        # Projects raw observations to hidden dimension before GRU processing
        self.rnn_projection = nn.Sequential(
            nn.Linear(input_shape, self.agent_hidden_dim),
            self.activation,
        )
        # Remove duplicate line that was causing issues
        self.rnn = nn.GRU(
            self.agent_hidden_dim, 
            self.agent_hidden_dim, 
            num_layers=self.gru_layers, 
            batch_first=True
        )

        # 2. Intent Extraction: (h_i^t, a_i^{t-1}) -> μ, σ for variational intent e_i^t
        self.intent_encoder_fc = nn.Sequential(
            nn.Linear(self.agent_hidden_dim + self.n_actions, self.intent_encoder_hidden_dim),
            RMSNorm(self.intent_encoder_hidden_dim),
            self.activation,
            nn.Linear(self.intent_encoder_hidden_dim, self.intent_dim * 2) # mu and log_var
        )

        # 3. Message Fusion with Dual Alignment Attention
        # Projects intents and contexts to attention space for message fusion
        self.query_projection = nn.Linear(self.intent_dim, self.dual_alignment_attn_dim)
        self.key_projection = nn.Linear(self.intent_dim, self.dual_alignment_attn_dim)
        self.value_projection = nn.Linear(self.intent_dim + self.agent_hidden_dim, self.dual_alignment_attn_dim)

        # Attention scaling factor for stable gradients
        self.attn_scale_factor = self.dual_alignment_attn_dim ** -0.5

        # 4. Q-value Network: Combines own message with fused communication context
        self.q_net = nn.Sequential(
            nn.Linear(
                self.message_dim + self.dual_alignment_attn_dim, 
                self.agent_hidden_dim
            ),
            self.activation,
            nn.Linear(self.agent_hidden_dim, self.n_actions)
        )

        PyMARLLogger.fast_logger().info(f"Agent Size: {get_parameters_num(self.parameters())}")

    def init_hidden(self, batch_size=1):
        """Initialize GRU hidden state"""
        return torch.zeros(self.gru_layers, batch_size, self.agent_hidden_dim, device=self.device)

    def extract_intent(
            self, 
            obs,  # (bs, ts, n_agents, 1, obs_dim)
            hidden_state,  # (gru_layers, bs * n_agents, agent_hidden_dim)
            last_action,   # (bs, ts, n_agents, 1, n_actions)
            current_time_step_tensor,  # (bs, ts, n_agents, 1, 1)
            agent_id_tensor,  # (bs, ts, n_agents, 1, n_agents)
        ):
        """
        Extract variational intents and package messages for communication
        
        Process:
        1. Encode observation sequence through GRU
        2. Generate intent distribution parameters (μ, σ)
        3. Sample intent using reparameterization trick
        4. Package structured message for communication
        """
        batch_size, time_size, n_agents, _, _ = obs.shape

        # 1. Observation Encoding h_i^t
        x = self.rnn_projection(obs)  # Project to hidden dimension
        
        # Reshape for GRU: (B*N, T, H)
        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.agent_hidden_dim)
        x, hidden_state = self.rnn(x, hidden_state)   # Sequential encoding
        
        # Reshape back: (B, T, N, 1, H)
        x = x.reshape(batch_size, n_agents, time_size, 1, -1).transpose(1, 2)

        # 2. Intent Generation with Variational Approach
        intent_encoder_input = torch.cat((x, last_action), dim=-1)
        intent_params_flat = self.intent_encoder_fc(intent_encoder_input)
        mu, log_var = torch.split(intent_params_flat, self.intent_dim, dim=-1)

        # Convert log-variance to standard deviation for numerical stability
        std = torch.exp(0.5 * log_var)

        # Sample intent using reparameterization trick for gradient flow
        intent_distribution = Normal(mu, std)
        intent = intent_distribution.rsample()  # Differentiable sampling

        # 3. Message Packaging for Communication
        messages = CoDeBatchedMessageData(agent_id_tensor, intent, x, current_time_step_tensor)

        return x, intent, messages, mu, std

    def forward(
            self, 
            sent_messages: CoDeBatchedMessageData, 
            received_messages: CoDeBatchedMessageData,
        ):
        """
        Compute Q-values using dual alignment attention for message fusion
        
        Process:
        1. Project own intent as query, received intents as keys/values
        2. Apply temporal discounting based on message timestamps
        3. Compute attention weights and fuse received messages
        4. Generate Q-values from combined context
        """
        
        # 1. Dual Alignment Attention Setup
        query = self.query_projection(sent_messages.intents)    # Own intent as query
        key = self.key_projection(received_messages.intents)    # Others' intents as keys  
        value = self.value_projection(
            torch.cat((received_messages.intents, received_messages.hiddens), dim=-1)
        )   # Combined intent+hidden as values
        
        # 2. Temporal Alignment with Message Age Discounting
        delta_t = sent_messages.sent_times - received_messages.sent_times   # Message age
        if delta_t.min() < 0:
            PyMARLLogger.fast_logger().fatal(f"Negative message age detected: {delta_t.min().item()}")
            delta_t = torch.clamp(delta_t, min=0)  # Ensure non-negative age
        
        gamma_t = (self.temporal_discount_gamma_T ** delta_t).transpose(-1, -2)  # Temporal discount

        # 3. Scaled Dot-Product Attention with Temporal Weighting
        attn_weights = query @ key.transpose(-2, -1) * self.attn_scale_factor
        attn_weights = torch.softmax(attn_weights, dim=-1) * gamma_t  # Apply temporal discount
        
        # 4. Message Fusion
        combined_messages = attn_weights @ value

        # 5. Q-value Computation from Fused Context
        state_embedding = torch.cat([
            sent_messages.sender_id,    # Agent identity
            sent_messages.intents,      # Own intent
            sent_messages.hiddens,      # Own hidden state  
            sent_messages.sent_times,   # Current timestamp
            combined_messages           # Fused received messages
        ], dim=-1)

        action_values = self.q_net(state_embedding)
        return attn_weights, action_values
