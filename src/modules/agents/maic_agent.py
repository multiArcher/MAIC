"""MAIC agent with decentralized teammate modeling and incentive messages.

Ported from the official PyMARL implementation:
https://github.com/LAMDA-RL/MAIC

Yuan et al., "Multi-Agent Incentive Communication via Decentralized
Teammate Modeling", AAAI 2022.
"""

from __future__ import annotations

import torch as th
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

from modules.agents.agent import Agent


def hidden_dim_from_args(args) -> int:
    """Resolve the RNN width across this repo and original MAIC configs."""
    for key in ("hidden_dim", "rnn_hidden_dim", "agent_hidden_dim"):
        value = getattr(args, key, None)
        if value is not None:
            return int(value)
    return 64


class MAICAgent(Agent):
    """Shared-parameter MAIC agent.

    Each agent (i) maintains a per-teammate Gaussian embedding z_{ij}, generates
    an action-value incentive message for every teammate, and gates those
    messages with attention. Auxiliary MI and entropy losses are returned
    during training so the learner can add them to the TD objective.
    """

    def __init__(self, input_shape, args):
        super(MAICAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions
        self.hidden_dim = hidden_dim_from_args(args)
        self.attention_dim = args.attention_dim
        self.var_floor = args.var_floor

        nn_hidden_size = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        # Teammate model: h_i -> (mu_{ij}, var_{ij}) for every teammate j.
        self.embed_net = nn.Sequential(
            nn.Linear(self.hidden_dim, nn_hidden_size),
            nn.BatchNorm1d(nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, args.n_agents * args.latent_dim * 2),
        )

        # Inference model q(z | h_i, a_j) used by the MI regularizer.
        self.inference_net = nn.Sequential(
            nn.Linear(self.hidden_dim + args.n_actions, nn_hidden_size),
            nn.BatchNorm1d(nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, args.latent_dim * 2),
        )

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

        # Incentive message: (h_i, z_{ij}) -> Delta Q_j
        self.msg_net = nn.Sequential(
            nn.Linear(self.hidden_dim + args.latent_dim, nn_hidden_size),
            activation_func,
            nn.Linear(nn_hidden_size, args.n_actions),
        )

        self.w_query = nn.Linear(self.hidden_dim, args.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, args.attention_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        latent_parameters = self.embed_net(h)
        latent_parameters[:, -self.n_agents * self.latent_dim :] = th.clamp(
            th.exp(latent_parameters[:, -self.n_agents * self.latent_dim :]),
            min=self.var_floor,
        )

        latent_embed = latent_parameters.reshape(
            bs * self.n_agents, self.n_agents * self.latent_dim * 2
        )

        if test_mode:
            latent = latent_embed[:, : self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(
                latent_embed[:, : self.n_agents * self.latent_dim],
                (latent_embed[:, self.n_agents * self.latent_dim :]) ** (1 / 2),
            )
            latent = gaussian_embed.rsample()
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)

        h_repeat = (
            h.view(bs, self.n_agents, -1)
            .repeat(1, self.n_agents, 1)
            .view(bs * self.n_agents * self.n_agents, -1)
        )
        msg = self.msg_net(th.cat([h_repeat, latent], dim=-1)).view(
            bs, self.n_agents, self.n_agents, self.n_actions
        )

        query = self.w_query(h).unsqueeze(1)
        key = self.w_key(latent).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = th.bmm(query / (self.attention_dim ** (1 / 2)), key).view(
            bs, self.n_agents, self.n_agents
        )
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        if test_mode:
            alpha = alpha.clone()
            alpha[alpha < (0.25 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg
        return_q = q + th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)

        returns = {}
        if kwargs.get("train_mode", False):
            if getattr(self.args, "mi_loss_weight", 0) > 0:
                returns["mi_loss"] = self.calculate_action_mi_loss(h, bs, latent_embed, return_q)
            if getattr(self.args, "entropy_loss_weight", 0) > 0:
                query_d = self.w_query(h.detach()).unsqueeze(1)
                key_d = (
                    self.w_key(latent.detach())
                    .reshape(bs * self.n_agents, self.n_agents, -1)
                    .transpose(1, 2)
                )
                alpha_d = F.softmax(th.bmm(query_d, key_d), dim=-1).reshape(
                    bs, self.n_agents, self.n_agents
                )
                returns["entropy_loss"] = self.calculate_entropy_loss(alpha_d)
            msg_l1_weight = getattr(self.args, "msg_l1_reg_loss_weight", 0)
            if msg_l1_weight > 0:
                returns["msg_l1_reg_loss"] = gated_msg.abs().mean() * msg_l1_weight

        return return_q, h, returns

    def calculate_action_mi_loss(self, h, bs, latent_embed, q):
        latent_embed = latent_embed.view(bs * self.n_agents, 2, self.n_agents, self.latent_dim)
        g1 = D.Normal(
            latent_embed[:, 0, :, :].reshape(-1, self.latent_dim),
            latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1 / 2),
        )
        hi = (
            h.view(bs, self.n_agents, 1, -1)
            .repeat(1, 1, self.n_agents, 1)
            .view(bs * self.n_agents * self.n_agents, -1)
        )

        selected_action = th.max(q, dim=1)[1].unsqueeze(-1)
        one_hot_a = th.zeros(
            selected_action.shape[0], self.n_actions, device=h.device
        ).scatter(1, selected_action, 1)
        one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(1, self.n_agents, 1, 1)
        one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)

        latent_infer = self.inference_net(th.cat([hi, one_hot_a], dim=-1)).view(
            bs * self.n_agents * self.n_agents, -1
        )
        latent_infer[:, self.latent_dim :] = th.clamp(
            th.exp(latent_infer[:, self.latent_dim :]), min=self.var_floor
        )
        g2 = D.Normal(
            latent_infer[:, : self.latent_dim],
            latent_infer[:, self.latent_dim :] ** (1 / 2),
        )
        mi_loss = kl_divergence(g1, g2).sum(-1).mean()
        return mi_loss * self.args.mi_loss_weight

    def calculate_entropy_loss(self, alpha):
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = -(alpha * th.log2(alpha)).sum(-1).mean()
        return entropy_loss * self.args.entropy_loss_weight
