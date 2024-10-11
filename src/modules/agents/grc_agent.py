import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Any, Optional


class GRCAgent(nn.Module):
    def __init__(self, obs_shape: int, args: SimpleNamespace) -> None:
        super().__init__()
        self.args = args
        self.args.obs_shape = obs_shape

        self.obs_encoding = nn.Linear(obs_shape, args.hidden_dim)
        """obs_encoding encodes the obs to rnn input."""
        # TODO Maybe a net with GRU is better.

        self.communication_net = self.CommunicationNet(args)
        """net used to generate what message to send"""

        # self.obs_embedding = nn.Linear(obs_shape, args.message_dim)
        """obs_embedding embeds the obs to feature of self graph node"""
        # TODO Maybe a net with GRU is better.

        # self.message_embedding = nn.Linear(args.message_dim, args.node_feature_dim)
        """message_embedding embeds teammates' message into node features."""

        self.gnn = self.GRCNet(self.args)
        """gnn ia a GNN with feature propagation and (graph forward)"""

        self.aggregator = nn.Linear(
            args.node_feature_dim * args.n_agents, args.node_feature_dim
        )
        """aggregator ia a GNN with feature propagation and (graph backward)"""

        # The rest are normal actor net.
        self.actor_net_fc1 = nn.Linear(args.hidden_dim + args.node_feature_dim, args.hidden_dim)
        self.actor_net_rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.actor_net_fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.hidden_state = None

    def init_hidden(self):
        # make hidden states on same device as model
        return self.actor_net_fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, obs, messages, hidden_state) -> (torch.Tensor, torch.Tensor):
        obs_encoding = self.obs_encoding(obs)   # n_agent * hidden_dim

        # message handle
        # obs_embedding = self.obs_embedding(obs)  # n_agent * message_dim
        # message_embedding = self.message_embedding(messages)    # n_agent * n_agent * message_dim
        self_message = self.communication_net(obs)

        # Replace the message_embedding of self graph node with obs_embedding.
        # message_embedding = self._replace_self_message_embedding(message_embedding, obs_embedding)
        messages = self._replace_self_message_embedding(messages, self_message)

        # Storage message embedding in graph structure
        self.gnn.node_features = messages

        # Propagate through graph network. Including feature propagation.
        self.gnn.graph_reconstruction()
        # self.gnn.propagate()
        message_embedding = self.aggregator(
            self.gnn.reconstructed_graph.reshape(
                -1, self.args.n_agents, self.args.n_agents * self.args.node_feature_dim
            )
        )  # batch * n_a * message_dim

        actor_input = torch.cat((obs_encoding, message_embedding), dim=-1)

        # Actor forward
        batch_size, _, actor_dim = actor_input.shape
        x = F.leaky_relu(self.actor_net_fc1(actor_input.reshape(-1, actor_dim)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.actor_net_rnn(x, h_in)
        q = self.actor_net_fc2(h)

        return q.reshape(batch_size, self.args.n_agents, -1), h.reshape_as(hidden_state)

    def generate_message(self, obs) -> torch.Tensor:
        # Encode the message agents want to send
        messages = self.communication_net(obs)
        return messages

    class GRCNet:
        def __init__(self, args: SimpleNamespace):
            """
            This net is used to storage the communication in multi-agent systems.
            Including communication signals as node features, selected communications as edges.
            """

            super().__init__()

            self.args: SimpleNamespace = args
            self.graph: Optional[Any] = None

            self._node_features: Optional[torch.Tensor] = None
            self.node_features_missing_mask: Optional[torch.Tensor] = None  # Mask for missing node features.
            self.reconstructed_graph: Optional[torch.Tensor] = None

            self._symmetrically_normalized_adjacency = (
                    self._get_full_symmetrically_normalized_adjacency(self.args.n_agents)
            )

        @property
        def node_features(self) -> torch.Tensor:
            return self._node_features

        @node_features.setter
        def node_features(self, node_features):
            self._node_features = node_features
            self.node_features_missing_mask = ~node_features.isnan()

        def graph_reconstruction(self) -> None:
            self.reconstructed_graph = torch.zeros_like(self.node_features)
            self.reconstructed_graph[self.node_features_missing_mask] = (
                self.node_features[self.node_features_missing_mask]
            )

            for _ in range(40):
                self.reconstructed_graph = (
                        self._symmetrically_normalized_adjacency @ self.reconstructed_graph
                )
                self.reconstructed_graph[self.node_features_missing_mask] = (
                    self.node_features[self.node_features_missing_mask]
                )

        def propagate(self) -> None:
            # An privato operation on self._graph_data.
            self.node_features = self.node_features

        def _get_full_symmetrically_normalized_adjacency(self, n_nodes):
            # all edge in graph exists
            adjacency_matrix = torch.ones((n_nodes, n_nodes), dtype=torch.float32, device=self.args.device)
            degree_matrix = adjacency_matrix.sum(dim=1)
            deg_inv_sqrt = degree_matrix.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt.isinf(), 0)
            D_inv_sqrt_matrix = torch.diag(deg_inv_sqrt)
            symmetrically_normalized_adjacency = D_inv_sqrt_matrix @ adjacency_matrix @ D_inv_sqrt_matrix
            return symmetrically_normalized_adjacency

    class CommunicationNet(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.message_encoding = nn.Linear(args.obs_shape, args.message_dim)

        def forward(self, obs) -> torch.Tensor:
            obs_in = obs.reshape([-1, self.args.obs_shape])
            message_encoded = self.message_encoding(obs_in)

            # batch * n_agents * message_dim
            return message_encoded.reshape([obs.shape[0], obs.shape[1], self.args.message_dim])

    @staticmethod
    def _replace_self_message_embedding(message_embedding, obs_embedding):
        """
        Replace the message_embedding of self graph node with obs_embedding.
        Args:
            message_embedding:  (batch, n_agents, n_agents, dim)
            obs_embedding:  (batch, n_agents, dim)

        Returns:  (batch, n_agents, n_agents, dim)

        """
        batch_size, n_agents, _, message_dim = message_embedding.shape
        # (n_agents, n_agents) mask for selection.
        mask = torch.eye(n_agents, device=message_embedding.device).bool()
        message_embedding[mask.unsqueeze(0).expand(batch_size, -1, -1)] = (
            obs_embedding.view((-1, message_dim))
        )

        return message_embedding
