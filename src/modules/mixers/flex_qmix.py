from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as functional
from modules.layers import EntityAttentionLayer


class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args:SimpleNamespace, extra_dims=0, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims

        self.n_agents = args.env_info["n_agents"]
        self.n_enemies = args.env_info["n_enemies"]
        self.agent_features = args.env_info["agent_features"]
        self.enemy_features = args.env_info["enemy_features"]
        self.n_actions = args.env_info["n_actions"]
        self.state_shape = args.env_info["state_shape"]
        self.state_last_actions = args.env_info["state_last_action"]

        self.n_entities = self.n_agents + self.n_enemies
        self.agent_feature_dim = len(self.agent_features)
        self.enemy_feature_dim = len(self.enemy_features)
        # Do set() operation keeping order.
        self.entity_features = list(
            OrderedDict.fromkeys(self.agent_features + self.enemy_features).keys()
        )

        self.entity_dim = len(self.entity_features)



        # if self.args.entity_last_action:
        #     self.entity_dim += args.n_actions

        self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        # TODO: temp 19, must change one day!!!!!
        self.fc1 = nn.Linear(19, hypernet_embed)
        # if args.pooling_type is None:
        self.attn = EntityAttentionLayer(hypernet_embed,
                                         hypernet_embed,
                                         hypernet_embed, args)

        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask=None, attn_mask=None):
        x1 = functional.relu(self.fc1(entities))
        # without entity_mask
        bs, ne, _ = entities.shape
        entity_mask = torch.ones(bs, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        if attn_mask is None:
            # create attn_mask from entity mask
            attn_mask = 1 - torch.bmm((1 - agent_mask.to(torch.float)).unsqueeze(2),
                                   (1 - entity_mask.to(torch.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask.to(torch.uint8),
                       post_mask=agent_mask) 
        x3 = self.fc2(x2)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2).bool(), 0) #[bs, na, edim]
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            return x3.mean(dim=2)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3


class FlexQMixer(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super(FlexQMixer, self).__init__()
        self.args = args

        # Initialize entity mapping.
        # TODO: This implementation is based on reconstruction from state_names,
        #  which is not compatible with new state format. A more robust solution
        #  is implemented in entity controller. This method should be rewritten.
        self.n_agents = args.env_info["n_agents"]
        self.n_enemies = args.env_info["n_enemies"]
        self.state_shape = args.env_info["state_shape"]
        self.agent_features = args.env_info["agent_features"]
        self.enemy_features = args.env_info["enemy_features"]
        self.n_actions = args.n_actions
        self.state_last_actions = args.state_last_actions

        self.n_entities = self.n_agents + self.n_enemies
        self.agent_feature_dim = len(self.agent_features)
        self.enemy_feature_dim = len(self.enemy_features)

        self.entity_features = list(
            OrderedDict.fromkeys(self.agent_features + self.enemy_features).keys()
        )
        self.entity_dim = len(self.entity_features)
        # Calculate indices for mapping state vector to entity state matrix.
        self.col_indices = None
        self.row_indices = None
        self._init_mapping_indices()

        # Mixing networks.
        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionHyperNet(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = AttentionHyperNet(args, mode='scalar')

        self.non_lin = functional.elu

    def forward(self, agent_qs, inputs, ret_ingroup_prop=False):
        entities = self._build_entity_state(inputs)
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)

        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)   #[4800,1,8]

        # First layer
        w1 = self.hyper_w_1(entities) # [4800,8,32]
        b1 = self.hyper_b_1(entities) # [4800,32]
        w1 = w1.view(bs * max_t, -1, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        w1 = functional.softmax(w1, dim=-1)

        hidden = self.non_lin(torch.bmm(agent_qs, w1) + b1) #[4800,1,32]

        # Second layer
        w_final = functional.softmax(self.hyper_w_final(entities), dim=-1) #[4800,32]

        w_final = w_final.view(-1, self.embed_dim, 1) 

        v = self.V(entities)  # State-dependent bias


        y = torch.bmm(hidden, w_final) + v  # Compute final output


        q_tot = y.view(bs, -1, 1)  # Reshape and return

        if ret_ingroup_prop:
            ingroup_w = w1.clone()
            ingroup_w[:, self.n_agents:] = 0  # zero-out out of group weights
            ingroup_prop = (ingroup_w.sum(dim=1)).mean()
            return q_tot, ingroup_prop

        return q_tot


    def _init_mapping_indices(self):
        """Calculate indices for mapping state vector to entity state matrix."""
        # TODO: A new key "state_components" is added to env_info, This implementation is outdated.
        # Mapping index from state of each ally/enemy to entity state.
        index_mapping_ally = [
            self.entity_features.index(feature) for feature in self.agent_features
        ]  # From state index of each ally to entity state index.
        index_mapping_enemy = [
            self.entity_features.index(feature) for feature in self.enemy_features
        ]  # From state index of each enemy to entity state index.

        # Ally state bits mapping. Each pair in indices is the new index in entity state matrix.
        ally_row_indices = torch.arange(0, self.n_agents).repeat_interleave(self.agent_feature_dim)
        ally_col_indices = torch.tensor(index_mapping_ally).repeat(self.n_agents)

        # Enemy state bits mapping.
        enemy_row_indices = torch.arange(0, self.n_agents).repeat_interleave(self.enemy_feature_dim) + self.n_agents
        enemy_col_indices = torch.tensor(index_mapping_enemy).repeat(self.n_enemies)

        col_indices = torch.cat([ally_col_indices, enemy_col_indices], dim=0)
        row_indices = torch.cat([ally_row_indices, enemy_row_indices], dim=0)

        if self.state_last_actions is True:
            # last action bits mapping.
            last_action_row_indices = torch.arange(0, self.n_agents).repeat_interleave(self.n_actions)
            last_action_col_indices = torch.arange(0, self.n_actions).repeat(self.n_agents) + self.entity_dim

            col_indices = torch.cat([col_indices, last_action_col_indices], dim=0)
            row_indices = torch.cat([row_indices, last_action_row_indices], dim=0)

        self.col_indices = col_indices
        self.row_indices = row_indices

        # TODO: There are more elements in statevector. This function is built on agent and enemy features.
        # If the assert below fails, new mapping need to be added like the state_last_actions mapping above.

        assert self.col_indices.shape[0] == self.row_indices.shape[
            0] == self.state_shape, "Mapping index is not correct."
        # Check if the mapping is correct.
        # out_list = [[0] * (self.entity_dim + self.n_actions) for _ in range(self.n_entities)]
        #
        # for i in range(len(self.state_feature_names)):
        #     out_list[self.row_indices[i]][self.col_indices[i]] = self.state_feature_names[i]

    def _build_entity_state(self, in_tensor: torch.Tensor):
        """
            Convert state vector into entity state matrix.

            Arguments:
                in_tensor (torch.Tensor): Input tensor of shape (batch_size, time_steps, state_dim)
            Returns:
                out_tensor (torch.Tensor): Output tensor of shape (batch_size, time_steps, num_entities, entity_dim)
        """
        # return entity state with shape (batch_size, time_slice, num_entities, entity_dim)
        out_tensor = torch.zeros(
            (self.n_entities, (self.entity_dim + self.n_actions)),
            device=in_tensor.device
        ).repeat(in_tensor.shape[0], in_tensor.shape[1], 1, 1)

        # use pre-calculated mapping index to map state vector to entity state matrix
        out_tensor[..., self.row_indices, self.col_indices] = in_tensor

        return out_tensor