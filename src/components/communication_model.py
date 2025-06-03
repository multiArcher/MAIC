from collections import namedtuple
from types import SimpleNamespace as SN
from typing import Optional

import torch

from utils.custom_logging import PyMARLLogger


# Structured message data for CoDe communication
CoDeBatchedMessageData = namedtuple("CoDeAgentBroadcastData", [
    "sender_id",  # (bs, ts_slice, n_a, 1, n_agents): Agent ID (one-hot)
    "intents",    # (bs, ts_slice, n_a, 1, intent_dim): Intent vector
    "hiddens",    # (bs, ts_slice, n_a, 1, hidden_dim): Agent hidden state
    "sent_times"  # (bs, ts_slice, n_a, 1, 1): Message timestamp
])


class CommunicationModel:
    """
    Communication Model for CoDe Algorithm
    
    Handles message passing between agents with:
    1. Gaussian delay modeling for realistic communication
    2. Message caching and retrieval based on arrival times
    3. Support for different communication topologies
    """

    def __init__(self, args, delay_mean: float = 1, delay_std: float = 1):
        self.args: SN = args
        self.device: torch.device | str = args.device
        self.n_agents: int = args.n_agents
        self.intent_dim: int = args.intent_dim
        self.agent_hidden_dim: int = args.agent_hidden_dim

        # Communication configuration
        self.comm_type: str = getattr(args, "comm_type", "broadcast")

        # Gaussian delay model parameters
        self.comm_gaussian_delay_mean: float = delay_mean
        self.comm_gaussian_delay_std: float = max(0.0, delay_std)

        # Cache configuration
        self.max_cache_size: int = args.env_info["episode_limit"] + 1
        
        # Pre-computed tensors for efficient message querying
        self.query_times = torch.arange(
            self.max_cache_size, 
            device=self.device, 
            dtype=torch.int
        ).reshape(1, self.max_cache_size, 1, 1, 1, 1)
        
        # Initialize caches
        self.reset(0)

    def reset(self, batch_size: int):
        """Reset message caches for new training batch"""
        cache_shape = (batch_size, self.max_cache_size, self.n_agents, 1)

        # Initialize all caches with appropriate default values
        self.cache_arrival_times = torch.full(
            (*cache_shape, 1), self.max_cache_size, dtype=torch.int, device=self.device
        )
        self.cache_sent_times = torch.full(
            (*cache_shape, 1), -1, dtype=torch.int, device=self.device
        )        
        self.cache_intents = torch.zeros(
            (*cache_shape, self.intent_dim), dtype=torch.float, device=self.device
        )
        self.cache_hidden_states = torch.zeros(
            (*cache_shape, self.agent_hidden_dim), dtype=torch.float, device=self.device
        )
        self.cache_sender_ids = torch.zeros(
            (*cache_shape, self.n_agents), dtype=torch.long, device=self.device
        )

    def process_communication(
            self, 
            broadcasts: CoDeBatchedMessageData, 
            t: slice,
            training: bool = False
            ) -> CoDeBatchedMessageData:
        """
        Process communication with delay modeling and message retrieval
        
        Process:
        1. Store new messages in cache with calculated arrival times
        2. Retrieve messages that have arrived by current timestep
        3. Apply communication topology (broadcast, etc.)
        """
        batch_size, max_len, n_agents, _, dim_intent = self.cache_intents.shape
 
        # 1. Store new messages in cache
        self.cache_sent_times[:, t, ...] = broadcasts.sent_times.long()
        self.cache_intents[:, t, ...] = broadcasts.intents.detach()
        self.cache_hidden_states[:, t, ...] = broadcasts.hiddens.detach()
        self.cache_sender_ids[:, t, ...] = broadcasts.sender_id

        # Calculate and store arrival times with Gaussian delay
        arrival_times = self._calculate_arrival_times(broadcasts.sent_times, training=training)
        self.cache_arrival_times[:, t, ...] = arrival_times.long()

        # 2. Retrieve messages that have arrived by current timestep
        # Create mask for messages that have arrived at each query time
        arrival_times_broadcast = self.cache_arrival_times.unsqueeze(1).expand(
            -1, self.max_cache_size, -1, -1, -1, -1
        )
        has_arrived_mask = arrival_times_broadcast <= self.query_times

        # Find the latest arrived message for each agent at each timestep
        received_message_sent_time = torch.min(has_arrived_mask, dim=2, keepdim=False)[1] - 1

        # Extract the latest received messages
        batch_idx = torch.arange(batch_size, device=self.device).reshape(batch_size, 1, 1, 1)
        time_idx = received_message_sent_time[:, t, :, 0]
        sender_id = torch.arange(n_agents, device=self.device).reshape(1, 1, n_agents, 1)

        current_messages = CoDeBatchedMessageData(
            sender_id=self.cache_sender_ids[batch_idx, time_idx, sender_id, 0],
            intents=self.cache_intents[batch_idx, time_idx, sender_id, 0],
            hiddens=self.cache_hidden_states[batch_idx, time_idx, sender_id, 0],
            sent_times=self.cache_sent_times[batch_idx, time_idx, sender_id, 0]
        )

        # 3. Apply communication topology
        match self.comm_type:
            case "broadcast":
                current_messages = self._broadcast_communication(current_messages)
            case "no_comm":
                # Return empty messages for no communication
                current_messages = self._no_communication(current_messages)
            case _:
                PyMARLLogger.fast_logger("CommunicationModel").error(
                    f"Unknown communication type: {self.comm_type}"
                )
                raise ValueError(f"Unknown communication type: {self.comm_type}")

        return current_messages 
        
    def _broadcast_communication(
            self, 
            messages: CoDeBatchedMessageData, 
            comm_matrix: Optional[torch.Tensor] = None,
            batch_indices: Optional[torch.Tensor] = None,
            time_indices: Optional[torch.Tensor] = None,
            agent_indices: Optional[torch.Tensor] = None
        ) -> CoDeBatchedMessageData:
        """
        Implement broadcast communication (all-to-all except self)
        """
        batch_size, time_len, n_agents, _, _ = messages.intents.shape

        if comm_matrix is None or comm_matrix.shape != (batch_size, time_len, n_agents, n_agents):
            # Update related tensors.
            comm_matrix = torch.eye(n_agents, device=self.device, dtype=torch.bool).logical_not()  # n * n
            comm_matrix = comm_matrix.reshape(1, 1, n_agents, n_agents  # 1 * 1 * n * n
                                    ).expand(batch_size, time_len, -1, -1)  # b * t * n * n
            
            batch_indices = torch.arange(batch_size, device=self.device).reshape(batch_size, 1, 1, 1)
        
        if time_indices is None:
            time_indices = torch.arange(time_len, device=self.device).reshape(1, time_len, 1, 1)
        if agent_indices is None:
            agent_indices = torch.arange(n_agents).reshape(1, 1, 1, n_agents).repeat(1, 1, n_agents, 1)

        def broadcast_func(m):
            """Apply broadcasting to message tensor"""
            m = m[batch_indices, time_indices, agent_indices, 0][comm_matrix]
            m = m.reshape(batch_size, time_len, n_agents, n_agents - 1, -1)
            return m
        
        return CoDeBatchedMessageData(
            sender_id=broadcast_func(messages.sender_id),
            intents=broadcast_func(messages.intents),
            hiddens=broadcast_func(messages.hiddens),
            sent_times=broadcast_func(messages.sent_times)
        )

    def _no_communication(self, messages: CoDeBatchedMessageData) -> CoDeBatchedMessageData:
        """No communication - return empty messages"""
        batch_size, time_len, n_agents, _, _ = messages.intents.shape
        
        return CoDeBatchedMessageData(
            sender_id=torch.zeros(batch_size, time_len, n_agents, 0, self.n_agents, device=self.device),
            intents=torch.zeros(batch_size, time_len, n_agents, 0, self.intent_dim, device=self.device),
            hiddens=torch.zeros(batch_size, time_len, n_agents, 0, self.agent_hidden_dim, device=self.device),
            sent_times=torch.zeros(batch_size, time_len, n_agents, 0, 1, device=self.device)
        )

    def _calculate_arrival_times(
            self, sent_times: torch.Tensor, 
            training: bool=False
            ) -> torch.Tensor:
        """
        Calculate message arrival times using Gaussian delay model
        
        Clamps delays to reasonable bounds to prevent numerical issues
        """
        delay_sample = torch.normal(
            self.comm_gaussian_delay_mean if training is False else 0.0,
            self.comm_gaussian_delay_std if training is False else 0.0,
            size=sent_times.shape,
            device=self.device
        ).clamp(min=0.0)

        arrive_times = sent_times + delay_sample
        arrive_times = torch.clamp(arrive_times.ceil(), max=self.max_cache_size)

        return arrive_times.long()
