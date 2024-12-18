# code adapted from https://github.com/wendelinboehmer/dcg
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
        # TODO The obs_shape in obs_encoding should be message_passing_shape.
        # TODO Maybe a net with GRU is better.

        self.communication_net = self.CommunicationNet(args)

        self.obs_embedding = nn.Linear(obs_shape, args.node_feature_dim)
        """obs_embedding embeds the obs to feature of self graph node"""
        # TODO Maybe a net with GRU is better.

        self.message_embedding = nn.Linear(args.message_dim, args.node_feature_dim)
        """message_embedding embeds teammates' message into node features."""

        self.gnn = self.GRCNet(self.args)
        """gnn ia a GNN with feature propagation and (graph forward)"""
        # TODO nodes should be represented in graph to propagate. Now is just a walk around.

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
        obs_embedding = self.obs_embedding(obs)  # n_agent * message_dim
        message_embedding = self.message_embedding(messages)    # n_agent * n_agent * message_dim

        # Replace the message_embedding of self graph node with obs_embedding.
        message_embedding = self._replace_self_message_embedding(message_embedding, obs_embedding)

        # Storage message embedding in graph structure
        self.gnn.node_features = message_embedding

        # Propagate through graph network. Including feature propagation.
        self.gnn.graph_reconstruction()
        self.gnn.propagate()
        message_embedding = self.aggregator(
            self.gnn.node_features.reshape(-1, self.args.n_agents, self.args.n_agents * self.args.node_feature_dim)
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
        # TODO There should be a net to encode the mesages, use obs instead temporarily.
        messages = self.communication_net(obs)
        return messages

    class GRCNet:
        def __init__(self, args: SimpleNamespace):
            """
            This net is used to storage the communication in multi-agent systems.
            Including communication signals as node features, selected communications as edges.
            """
            self.warning_dict = {
                "class": True,
                "node_features": True,
                "propagate": True,
                "graph_reconstruction": True
            }

            if self.warning_dict["class"]:
                print(f"WARNING! USING NOT IMPLEMENTED CLASS GRCNet.")
                self.warning_dict["class"] = False

            super().__init__()

            self.args: SimpleNamespace = args
            self._node_features: Optional[torch.Tensor] = None
            self.graph: Optional[Any] = None

        @property
        def node_features(self) -> torch.Tensor:
            return self._node_features

        @node_features.setter
        def node_features(self, x):
            self._node_features = x
            if self.warning_dict["node_features"]:
                print(f"WARNING! USING NOT IMPLEMENTED Functions GRCNET.nodefeatures.")
                self.warning_dict["node_features"] = False

            # TODO Implement a graph node feature map.

        def propagate(self) -> None:
            # An privato operation on self._graph_data.
            self.node_features = self.node_features
            if self.warning_dict["propagate"]:
                print(f"WARNING! USING NOT IMPLEMENTED Functions GRCNET.propagate.")
                self.warning_dict["propagate"] = False
            # TODO A method of propagating through graph structure.

        def graph_reconstruction(self) -> None:
            # An privato operation on self._graph_data.
            self.node_features = self.node_features
            if self.warning_dict["graph_reconstruction"]:
                print(f"WARNING! USING NOT IMPLEMENTED Functions GRCNET.graph_reconstruction.")
                self.warning_dict["graph_reconstruction"] = False
                # TODO Use the feature propagation to reconstruct the missing message.

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
        mask = torch.eye(n_agents).bool().to(message_embedding.device)
        message_embedding[mask.unsqueeze(0).expand(batch_size, -1, -1)] = (
            obs_embedding.view((-1, message_dim))
        )

        return message_embedding
