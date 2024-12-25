from collections import OrderedDict
from abc import ABC, abstractmethod

import torch


class Transform(ABC):
    @abstractmethod
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform input to output"""
        pass

    @abstractmethod
    def infer_output_info(self, vshape_in: tuple[int], dtype_in: torch.dtype):
        """Infer output shape and type from input shape and type"""
        pass


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), torch.float32

# Original implementation of EntityState. Now this function is implemented in entity controller.
# class EntityState(Transform):
#     """
#     Convert state vector into entity state matrix.
#
#     Usage:
#         - Use infer_output_info to get desired output shape and type.
#         - Use transform to convert input tensor into output tensor.
#
#     Arguments:
#         n_agents (int): Number of agents.
#         n_enemies (int): Number of enemies.
#         agent_features (list[str]): List of agent feature names.
#         enemy_features (list[str]): List of enemy feature names.
#         state_shape (int): Shape of state vector.
#         state_feature_names (list[str]): List of state feature names. Not used anymore. Just for examine mapping correction
#         n_actions (int): Number of actions.
#         state_last_action (bool): Whether to include last action in state. Defaule is True.
#
#     """
#     def __init__(
#             self,
#             n_agents: int,
#             n_enemies: int,
#             agent_features: list[str],
#             enemy_features: list[str],
#             state_shape: int,
#             state_feature_names: list[str],
#             n_actions: int,
#             state_last_action: bool = True,
#             *args,
#             **kwargs,
#     ):
#         self.n_agents = n_agents
#         self.n_enemies = n_enemies
#         self.state_feature_names = state_feature_names
#         self.state_shape = state_shape
#         self.agent_features = agent_features
#         self.enemy_features = enemy_features
#         self.n_actions = n_actions
#         self.state_last_actions = state_last_action
#
#         self.n_entities = self.n_agents + self.n_enemies
#         self.agent_feature_dim = len(self.agent_features)
#         self.enemy_feature_dim = len(self.enemy_features)
#
#         # Do set() operation keeping order.
#         self.entity_features = list(
#             OrderedDict.fromkeys(self.agent_features + self.enemy_features).keys()
#         )
#
#         self.entity_dim = len(self.entity_features)
#
#         # Calculate indices for mapping state vector to entity state matrix.
#         self.col_indices = None
#         self.row_indices = None
#         self._init_mapping_indices()
#
#
#     def infer_output_info(self, vshape_in, dtype_in):
#         """Infer output shape and dtype from input shape and dtype."""
#         return (self.n_entities, self.entity_dim + self.n_actions), torch.float32
#
#     def transform(self, in_tensor: torch.Tensor):
#         """
#         Convert state vector into entity state matrix.
#
#         Arguments:
#             in_tensor (torch.Tensor): Input tensor of shape (batch_size, time_steps, state_dim)
#         Returns:
#             out_tensor (torch.Tensor): Output tensor of shape (batch_size, time_steps, num_entities, entity_dim)
#         """
#         # return entity state with shape (batch_size, time_slice, num_entities, entity_dim)
#         out_tensor = torch.zeros(
#             (self.n_entities, (self.entity_dim + self.n_actions)),
#             device=in_tensor.device
#         ).repeat(in_tensor.shape[0], in_tensor.shape[1], 1, 1)
#
#         # use pre-calculated mapping index to map state vector to entity state matrix
#         out_tensor[..., self.row_indices, self.col_indices] = in_tensor
#
#         return out_tensor
#
#     def _init_mapping_indices(self):
#         """Calculate indices for mapping state vector to entity state matrix."""
#         # TODO: A new key "state_components" is added to env_info, This implement is outdated.
#         # Mapping index from state of each ally/enemy to entity state.
#         index_mapping_ally = [
#             self.entity_features.index(feature) for feature in self.agent_features
#         ]   # From state index of each ally to entity state index.
#         index_mapping_enemy = [
#             self.entity_features.index(feature) for feature in self.enemy_features
#         ]   # From state index of each enemy to entity state index.
#
#         # Ally state bits mapping. Each pair in indices is the new index in entity state matrix.
#         ally_row_indices = torch.arange(0, self.n_agents).repeat_interleave(self.agent_feature_dim)
#         ally_col_indices = torch.tensor(index_mapping_ally).repeat(self.n_agents)
#
#         # Enemy state bits mapping.
#         enemy_row_indices = torch.arange(0, self.n_agents).repeat_interleave(self.enemy_feature_dim) + self.n_agents
#         enemy_col_indices = torch.tensor(index_mapping_enemy).repeat(self.n_enemies)
#
#         col_indices = torch.cat([ally_col_indices, enemy_col_indices], dim=0)
#         row_indices = torch.cat([ally_row_indices, enemy_row_indices], dim=0)
#
#         if self.state_last_actions is True:
#             # last action bits mapping.
#             last_action_row_indices = torch.arange(0, self.n_agents).repeat_interleave(self.n_actions)
#             last_action_col_indices = torch.arange(0, self.n_actions).repeat(self.n_agents) + self.entity_dim
#
#             col_indices = torch.cat([col_indices, last_action_col_indices], dim=0)
#             row_indices = torch.cat([row_indices, last_action_row_indices], dim=0)
#
#         self.col_indices = col_indices
#         self.row_indices = row_indices
#
#         #TODO: There are more elements in statevector. This function is built on agent and enemy features.
#         # If the assert below fails, new mapping need to be added like the state_last_actions mapping above.
#
#         assert self.col_indices.shape[0] == self.row_indices.shape[0] == self.state_shape, "Mapping index is not correct."
#         # Check if the mapping is correct.
#         # out_list = [[0] * (self.entity_dim + self.n_actions) for _ in range(self.n_entities)]
#         #
#         # for i in range(len(self.state_feature_names)):
#         #     out_list[self.row_indices[i]][self.col_indices[i]] = self.state_feature_names[i]
#
#
# class EntityObs(Transform):
#     """
#     Convert observation matrix into entity observation matrix.
#     """
#     def __init__(self, obs_components):
#         self.obs_components = obs_components
#         self.n_enemies, self.n_enemy_feats_dim = self.obs_components["n_enemy_feats"]
#         self.n_ally, self.n_ally_feats_dim = self.obs_components["n_ally_feats"]
#         self.move_feats_size = self.obs_components["move_feats_size"]
#         self.own_feats_size = self.obs_components["own_feats_size"]
#
#         self.slices: dict[str: slice] = {}
#         self._init_mapping_indices()
#
#     def infer_output_info(self, vshape_in: tuple[int], dtype_in: torch.dtype):
#         """Infer output shape and dtype from input shape and dtype."""
#         return (
#             self.move_feats_size,
#             (self.n_enemies, self.n_enemy_feats_dim),
#             (self.n_ally, self.n_ally_feats_dim),
#              self.own_feats_size
#         ), dtype_in
#
#     def transform(self, tensor: torch.Tensor) -> torch.Tensor:
#         move_action_vector = tensor[..., :4]
#
#
#     def _init_mapping_indices(self):
#         # Mapping indices.
#         bit_count = 0
#         self.slices["move_feats"] = slice(bit_count, bit_count:= bit_count + self.move_feats_size)
#         self.slices["n_enemy_feats"] = slice(bit_count, bit_count:= bit_count + self.n_enemies * self.n_enemy_feats_dim)
#         self.slices["n_ally_feats"] = slice(bit_count, bit_count:= bit_count + self.n_ally * self.n_ally_feats_dim)
#         self.slices["own_feats"] = slice(bit_count, bit_count:= bit_count + self.own_feats_size)
#
#         # For debugging
#         obs_shape = 92  # 94 in p 5v5 map
#         assert bit_count == obs_shape, "The mapping is not correct."
