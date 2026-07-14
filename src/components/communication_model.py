from collections import namedtuple
from types import SimpleNamespace as SN
from typing import Optional

import torch

from utils.custom_logging import PyMARLLogger


# Structured message data for CoDe communication，批次（并行的数量），时间步（时刻），有几个智能体，队友数量（来自哪个队友）。后面两列可以理解为一个矩阵
# 在实现算法时不要忽略“1”，这样才符合矩阵乘法
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

        # 1.Pre-compute communication matrix for broadcast
        comm_matrix = torch.eye(self.n_agents, device=self.device, dtype=torch.bool).logical_not()  # n * n
        self.comm_matrix_base = comm_matrix.reshape(1, 1, self.n_agents, self.n_agents)  # 1 * 1 * n * n

        # 2. Pre-compute the base agent indices tensor for gathering
        agent_indices = torch.arange(self.n_agents, device=self.device)
        self.agent_indices = agent_indices.reshape(1, 1, 1, self.n_agents).repeat(1, 1, self.n_agents, 1)

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
        # batch_size, max_len, n_agents, _, _ = self.cache_intents.shape
        time_start, time_stop = t.start, t.stop
        # time_len = time_stop - time_start
 
        # 1. Store new messages in cache
        self.cache_sent_times[:, t, ...] = broadcasts.sent_times.long()
        self.cache_intents[:, t, ...] = broadcasts.intents.detach()
        self.cache_hidden_states[:, t, ...] = broadcasts.hiddens.detach()
        self.cache_sender_ids[:, t, ...] = broadcasts.sender_id
        # Calculate and store arrival times with Gaussian delay
        arrival_times = self._calculate_arrival_times(broadcasts.sent_times, training=training)#应该到达的时间
        self.cache_arrival_times[:, t, ...] = arrival_times.long()#缓存的信息到达时间
        # 计算当前收到的消息本该什么时候发出
        relevant_history_slice = slice(0, time_stop)
        sliced_cache_arrival_times = self.cache_arrival_times[:, relevant_history_slice]
        sliced_cache_sent_times = self.cache_sent_times[:, relevant_history_slice]
        query_times_slice = self.query_times[:, time_start:time_stop]

        has_arrived_mask = sliced_cache_arrival_times.unsqueeze(1) <= query_times_slice#标记之前发出的信息当前有没有到达
        # 查询信息什么时候来的
        valid_sent_times = torch.where(
            has_arrived_mask,
            sliced_cache_sent_times.unsqueeze(1),
            torch.tensor(-1, device=self.device, dtype=torch.int)
        )
        time_idx, _ = valid_sent_times.max(dim=2)
        no_messages_mask = time_idx < 0  # Mask for no messages have arrived.
        safe_time_idx = time_idx.masked_fill(no_messages_mask, 0)  # Set 0 for no messages.
        #通信矩阵+观测延迟=信息延迟
        def gather_from_cache(#组成智能体当前应该拿到的通信矩阵
                cache_tensor: torch.Tensor, 
                safe_time_idx:torch.Tensor, 
                no_messages_mask:torch.Tensor
                ) -> torch.Tensor:
            idx_shape = torch.Size([*safe_time_idx.shape[:-1], cache_tensor.shape[-1]])

            # This is to avoid indexing errors when no messages have arrived.
            # Set -1 to 0 -> Index fake tensor with 0 time data temporarily -> set where -1 to 0.
            expanded_idx = safe_time_idx.expand(idx_shape)
            # Gather messages from cache tensor with the safe time index.
            # cache_tensor (b, t, n, 1, d) -> gather on dim=1
            fake_data = torch.gather(cache_tensor[:, relevant_history_slice], 1, expanded_idx.long())
            # If no messages have arrived, set the data to 0.
            true_data = fake_data.masked_fill(no_messages_mask, 0)

            return true_data
        # gather_from_cache是并行的高阶索引
        current_messages = CoDeBatchedMessageData(
            sender_id=gather_from_cache(self.cache_sender_ids, safe_time_idx, no_messages_mask),
            intents=gather_from_cache(self.cache_intents, safe_time_idx, no_messages_mask),
            hiddens=gather_from_cache(self.cache_hidden_states, safe_time_idx, no_messages_mask),
            sent_times=gather_from_cache(self.cache_sent_times, safe_time_idx, no_messages_mask)
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
        
    def _broadcast_communication(#使用滞后矩阵，先滞后再广播（这里顺序存疑）
            self, 
            messages: CoDeBatchedMessageData, 
        ) -> CoDeBatchedMessageData:
        """
        Implement broadcast communication (all-to-all except self)
        """
        batch_size, time_len, n_agents, _, _ = messages.intents.shape

        # Update related tensors.        
        comm_matrix = self.comm_matrix_base.expand(batch_size, time_len, -1, -1)  # b * t * n * n
        batch_indices = torch.arange(batch_size, device=self.device).reshape(batch_size, 1, 1, 1)        
        time_indices = torch.arange(time_len, device=self.device).reshape(1, time_len, 1, 1)

        def broadcast_func(m):  # 广播方法，给所有队友发送自己收到的信息
            """Apply broadcasting to message tensor"""
            m = m[batch_indices, time_indices, self.agent_indices, 0][comm_matrix]
            m = m.reshape(batch_size, time_len, n_agents, n_agents - 1, -1)
            return m
        
        return CoDeBatchedMessageData(
            sender_id=broadcast_func(messages.sender_id).detach(),
            intents=broadcast_func(messages.intents).detach(),
            hiddens=broadcast_func(messages.hiddens).detach(),
            sent_times=broadcast_func(messages.sent_times).detach()
        )
    # 不交流的情况，用不上
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
