import torch
import torch.nn as nn
from torch.distributions import Normal

from modules.agents.agent import Agent
from components.communication_model import CoDeBatchedMessageData
from utils.th_utils import get_parameters_num
from utils.custom_logging import PyMARLLogger

class CodeAgent(Agent):
    def __init__(self, input_shape, args):
        super(CodeAgent, self).__init__()
        # Storage configs.
        self.args = args
        self.device = args.device
        
        self.gru_layers = args.agent_gru_layers
        self.agent_hidden_dim = args.agent_hidden_dim

        self.intent_dim = args.intent_dim
        self.intent_encoder_hidden_dim = args.intent_encoder_hidden_dim
        self.intent_decoder_gru_hidden_dim = args.intent_decoder_gru_hidden_dim
        self.dual_alignment_attn_dim = args.dual_alignment_attn_dim

        self.predict_k_future_actions = args.predict_k_future_actions
        self.intent_action_predictor_hidden_sizes = args.intent_action_predictor_hidden_sizes

        self.temporal_discount_gamma_T = args.temporal_discount_gamma_T

        # Initialize variables.
        self.n_actions = args.n_actions
        self.message_dim = self.args.n_agents + self.intent_dim + self.agent_hidden_dim + 1 # agent_id + intent_e + hidden_state + time

        # --- Core Agent Architecture ---
        # 1. RNN for history encoding h_i^t
        #    Input to rnn_projection is the raw observation part of the input_shape
        self.rnn_projection = nn.Sequential(
            nn.Linear(input_shape, self.agent_hidden_dim),
            nn.ReLU(),
        )
        nn.Linear(input_shape, self.agent_hidden_dim)
        self.rnn = nn.GRU(
            self.agent_hidden_dim, 
            
            self.agent_hidden_dim, 
            num_layers=self.gru_layers, 
            batch_first=True
            )

        # 2. Intent Extraction (h_i^t, a_i^{t-1}) -> e_i^t
        self.intent_encoder_fc = nn.Sequential(
            nn.Linear(self.agent_hidden_dim + self.n_actions, self.intent_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.intent_encoder_hidden_dim, self.intent_dim * 2) # mu and log_var
        )

        # 3. Message Fusion (Dual Alignments) (e_i^t, e_{-i}^{?<t}, (e_{-i}^{?<t}, h_{-i}^{?<t})) -> (c_i^t)
        self.query_projection = nn.Linear(self.intent_dim, self.dual_alignment_attn_dim)
        self.kye_projection = nn.Linear(self.intent_dim, self.dual_alignment_attn_dim)
        self.value_projection = nn.Linear(self.intent_dim + self.agent_hidden_dim, self.dual_alignment_attn_dim)

        self.attn_scale_factor = self.dual_alignment_attn_dim ** -0.5

        # 4. Q-value MLP: Q(m_i^t, c_i^t) -> Q_i^t
        self.q_net = nn.Sequential(
            nn.Linear(
                self.message_dim + self.dual_alignment_attn_dim, 
                self.agent_hidden_dim
                ),
            nn.ReLU(),
            nn.Linear(self.agent_hidden_dim, self.n_actions)
        )

        PyMARLLogger.fast_logger().info(f"Agent Size: {get_parameters_num(self.parameters())}")


    def init_hidden(self, batch_size=1):
        # For nn.GRU, hidden state is (num_layers * num_directions, batch, hidden_size)
        # TODO: move batch out.
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
        Extracts the intent from the observation and last action, and packs messages.
        Args:
            obs: Observation tensor.
            hidden_state: Hidden state tensor.
            last_action: Last action tensor.
            current_time_step_tensor: Current time step tensor.
            agent_id_tensor: Agent ID tensor.
        """
        batch_size, time_size, n_agents, _, _ = obs.shape

        # - 1. Observation Encoding h_i^t
        # TODO: Shoule be done seperately.
        x = self.rnn_projection(obs)  # batch * time * n_agents * hidden_dim
        
        # b * t * n * 1 * d -> b * n * t * 1 * d -> (b * n) * t * (1 * d)
        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size, self.agent_hidden_dim)

        x, hidden_state = self.rnn(x, hidden_state)   # GRU forward.
        # (b * n) * t * d -> b * n * t * d -> b * t * n * d
        x = x.reshape(batch_size, n_agents, time_size, 1, -1).transpose(1, 2)

        # - 2. Intent Generation
        intent_encoder_input = torch.cat((x, last_action), dim=-1)
        intent_params_flat = self.intent_encoder_fc(intent_encoder_input)
        mu, log_var = torch.split(intent_params_flat, self.intent_dim, dim=-1)  # mu: b * t * n * 1 * d, log_var: b * t * n * 1 * d

        std = torch.exp(0.5 * log_var) # Corrected: 0.5 for std deviation

        intent_distribution = Normal(mu, std)
        # Sample from the distribution
        intent = intent_distribution.rsample()  # b * t * n * 1 * d

        # - 3. Message packing
        messages = CoDeBatchedMessageData(agent_id_tensor, intent, x, current_time_step_tensor)

        return x, intent, messages, mu, std

    def forward(
            self, 
            sent_messages: CoDeBatchedMessageData, 
            received_messages: CoDeBatchedMessageData,
            ):
        """
        Forward self message and received messages, return q values.
        """
        # - 1. Message Fusion
        query = self.query_projection(sent_messages.intents)    # b * t * n * 1 * d
        key = self.kye_projection(received_messages.intents)    # b * t * n * n-1 * d
        value = self.value_projection(
                torch.cat((received_messages.intents, received_messages.hiddens), dim=-1)
            )   # b * t * n * n-1 * d
        
        # Timeliness Alignment
        delta_t = sent_messages.sent_times - received_messages.sent_times   # b * t * n * n-1 * 1
        gamma_t = (self.temporal_discount_gamma_T ** delta_t).transpose(-1, -2)  # b * t * n * n-1 * 1 

        attn_weights = query @ key.transpose(-2, -1) * self.attn_scale_factor    #? No scale metioned in the paper. 
        attn_weights = torch.softmax(attn_weights, dim=-1) * gamma_t  # a_{-i}: b * t * n * 1 * n-1 
        
        combined_messages = attn_weights @ value

        # - 2. Q-value MLP
        # (bs, ts, n_agents, 1, message_dim) + (bs, ts, n_agents, 1, dual_alignment_attn_dim)
        state_embedding = torch.cat([
            sent_messages.sender_id,
            sent_messages.intents,
            sent_messages.hiddens,
            sent_messages.sent_times,
            combined_messages
        ], dim=-1)

        action_values = self.q_net(state_embedding)  # (bs, ts, n_agents, 1, n_actions)
        return attn_weights, action_values
