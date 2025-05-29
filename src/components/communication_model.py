from collections import namedtuple
from types import SimpleNamespace as SN
from typing import Optional

import torch

from utils.custom_logging import PyMARLLogger


# Batched communication data structure.
CoDeBatchedMessageData = namedtuple("CoDeAgentBroadcastData", [
    "sender_id",  # (bs, ts_slice, n_a, 1, 1): Agent ID
    "intents",    # (bs, ts_slice, n_a, 1, intent_dim): Intent vector
    "hiddens",    # (bs, ts_slice, n_a, 1, hidden_dim): Agent hidden state
    "sent_times"  # (bs, ts_slice, n_a, 1, 1) - actual episode time for each
])


class CommunicationModel:
    def __init__(self, args):
        self.args: SN = args
        self.device: torch.device | str = args.device
        self.n_agents: int = args.n_agents
        self.intent_dim: int = args.intent_dim
        self.agent_hidden_dim: int = args.agent_hidden_dim

        # Communication model parameters
        self.comm_type: str = getattr(args, "comm_type", "broadcast")   # Communication type. e.g., "broadcast", "no_comm"

        # Delay model parameters
        # TODO: Add support for other delay models.
        self.comm_gaussian_delay_mean: float = float(getattr(args, "comm_gaussian_delay_mean", 3))
        self.comm_gaussian_delay_std: float = float(getattr(args, "comm_gaussian_delay_std", 1))
        if self.comm_gaussian_delay_std < 0:
            self.comm_gaussian_delay_std = 0.0

        # Cache stores messages based on their actual sending time step
        self.max_cache_size: int = args.env_info["episode_limit"] + 1
        self.query_times = torch.arange(self.max_cache_size, device=self.device, dtype=torch.int
                                ).reshape(1, self.max_cache_size, 1, 1, 1, 1)   # Tensor for query arrived messages.
        
        # Initialize caches.
        self.reset(0)

    def reset(self, batch_size: int):
        """Resets the cache, typically at the start of a new training run or episode batch set."""
        cache_shape = (batch_size, self.max_cache_size, self.n_agents, 1)   # b * t * n * 1

        # Cache for message arrive time, default to max_cache_size which means message lost.
        self.cache_arrival_times = torch.full(
            (*cache_shape, 1), self.max_cache_size, dtype=torch.int, device=self.device
            )   
        # Cache for message sent time, default to max_cache_size which means message is never sent.
        self.cache_sent_times = torch.full(
            (*cache_shape, 1), self.max_cache_size, dtype=torch.int, device=self.device
            )        
        # Cache for intent, default to zero as initial intent.
        self.cache_intents = torch.zeros(
            (*cache_shape, self.intent_dim), dtype=torch.float, device=self.device
            )
        # Cache for hidden state, default to zero as initial hidden state.
        self.cache_hidden_states = torch.zeros(
            (*cache_shape, self.agent_hidden_dim), dtype=torch.float, device=self.device
            )
        # Cache for message sender ID, default to zero which means no sender ID.
        self.cache_sender_ids = torch.zeros(
            (*cache_shape, self.n_agents), dtype=torch.long, device=self.device
            )

    def process_communication(self, broadcasts: CoDeBatchedMessageData, t: slice) -> CoDeBatchedMessageData:
        # TODO: Seperate different communication types.
        """
        Processes a batch of messages broadcast by agents over a time slice.
        Calculates arrival times and stores them.

        Note: Storage hashes at arrival time step is feasible only in batch = 1.
        This is because the arrival time step is not unique.
        This method need to iterate over all time steps.

        To calculate in batch, we need to store the messages in the cache at the 
        time step of the sender.
        """       
        batch_size, max_len, n_agents, _, dim_intent = self.cache_intents.shape
 
        # - 1. Update the cache with the newly sent messages
        self.cache_sent_times[:, t, ...] = broadcasts.sent_times.long()
        self.cache_intents[:, t, ...] = broadcasts.intents.detach()
        self.cache_hidden_states[:, t, ...] = broadcasts.hiddens.detach()
        self.cache_sender_ids[:, t, ...] = broadcasts.sender_id

        arrival_times = self._calculate_arrival_times(broadcasts.sent_times)
        self.cache_arrival_times[:, t, ...] = arrival_times.long()

        # - 2. Choose current arrived messages.
        # Get a matrix including which messages have arrived at every time step for all agents.
        arrival_times_broadcast = self.cache_arrival_times.unsqueeze(1
                                                            ).expand(-1, self.max_cache_size, -1, -1, -1, -1)  # b * t * n * t * 1 * 1
        has_arrived_mask = arrival_times_broadcast <= self.query_times  # b * t * t * n * 1 * 1   
        
        # The sent time step of the latest message that has arrived. 
        received_message_sent_time = torch.max(has_arrived_mask.logical_not(), dim=2, keepdim=False)[1] - 1  # b * t * n * 1 * 1

        # Get the latest received messages for each agent
        batch_idx = torch.arange(batch_size, device=self.device).reshape(batch_size, 1, 1, 1)
        time_idx = received_message_sent_time[:, t, :, 0]
        sender_id = torch.arange(n_agents, device=self.device).reshape(1, 1, n_agents, 1)

        current_messages = CoDeBatchedMessageData(
            sender_id=self.cache_sender_ids[batch_idx, time_idx, sender_id, 0],
            intents=self.cache_intents[batch_idx, time_idx, sender_id, 0],
            hiddens=self.cache_hidden_states[batch_idx, time_idx, sender_id, 0],
            sent_times=self.cache_sent_times[batch_idx, time_idx, sender_id, 0]
        )

        # - 3. Process the messages based on the communication type
        match self.comm_type:
            case "broadcast":
                # For broadcast, we need to ensure all agents receive the same message.
                current_messages = self._broadcast_communication(current_messages)  # b * t * n * n-1 * d

            case "no_comm":
                pass

            case _:
                PyMARLLogger.fast_logger("CommunicationModel").error(
                    f"Unknown communication type: {self.comm_type}. Only suppert broadcast now."
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
        Broadcasts messages to all agents.
        """
        batch_size, time_len, n_agents, _, _ = messages.intents.shape   # b * t * n * 1 * dim

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
            m = m[batch_indices, time_indices, agent_indices, 0][comm_matrix]
            m = m.reshape(batch_size, time_len, n_agents, n_agents - 1, -1)
            return m
        
        return CoDeBatchedMessageData(
            sender_id=broadcast_func(messages.sender_id),
            intents=broadcast_func(messages.intents),
            hiddens=broadcast_func(messages.hiddens),
            sent_times=broadcast_func(messages.sent_times)
        )   # b * t * n * n-1 * d

    def _calculate_arrival_times(self, sent_times: torch.Tensor) -> torch.Tensor:
        delay_sample = torch.normal(
            self.comm_gaussian_delay_mean,
            self.comm_gaussian_delay_std,
            size=sent_times.shape,
            device=self.device
            )   # Sample delay times.
        arrive_times = torch.clamp(
            sent_times + delay_sample, 
            min=max(0.0, self.comm_gaussian_delay_mean - 3 * self.comm_gaussian_delay_std), 
            max=min(self.max_cache_size, self.comm_gaussian_delay_mean + 3 * self.comm_gaussian_delay_std)
            )   # Arrival time = sent time + delay time.

        return arrive_times.long()
