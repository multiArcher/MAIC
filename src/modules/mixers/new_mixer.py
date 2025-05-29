import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class NewMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(NewMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(
                            nn.Linear(self.input_dim, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim)
                        )
        self.hyper_b1 = nn.Linear(self.input_dim, self.embed_dim)

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(
                            nn.Linear(self.input_dim, args.hypernet_embed),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.hypernet_embed, self.embed_dim)
                        )
        self.hyper_b2 = nn.Sequential(
                            nn.Linear(self.input_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1)
                        )

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals: torch.Tensor, states):        
        # First layer
        w1 = self.hyper_w1(states).reshape(*qvals.shape[:-1], -1)  # b * t * n_agents * 1 * d
        b1 = self.hyper_b1(states)  # b * t * 1 * 1 * d

        # Second layer
        w2 = self.hyper_w2(states)  # b * t * 1 * 1 * d
        b2 = self.hyper_b2(states)  # b * t * 1 * 1 * 1

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(qvals.transpose(-3, -1) @ w1.transpose(-3, -2) + b1)  # b * t * 1 * 1 * d
        y = hidden @ w2.transpose(-1, -2) + b2  # b * t * 1 * 1 * 1

        return y

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return torch.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return torch.abs(x)
