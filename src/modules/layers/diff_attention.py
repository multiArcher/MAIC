import math
from types import SimpleNamespace

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm


class EntityDiffAttnLayer(nn.Module):
    """
    Differential Attention Layer for entity scheme multi-agent systems..
    Args:
        args: args initialized in pymarl.
        embed_dim: size of embedding feature.
        depth: depth of attention layer, used for calculating lambda_init.
        num_heads: number of attention heads.
    """
    def __init__(
            self,
            args: SimpleNamespace,
            embed_dim: int,
            depth,
            num_heads: int
    ):
        super(EntityDiffAttnLayer, self).__init__()
        self.args = args
        # MAS arguments.
        self.n_agents = args.n_agents
        # self.n_entities = args.n_entities

        # Attn arguments.
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads // 2
        self.scaling = self.head_dim ** -0.5

        # networks
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = self._lambda_inti_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        for name, module in self.named_modules():
            module.full_name = name

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}"

    def forward(
            self,
            entities: torch.Tensor,
            obs_mask: torch.Tensor = None,
            agent_mask: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            entities: Entity representations.
                [batch size, # of entities, embedding dimension]
            obs_mask: Which agent-entity pairs are not available (observability and/or padding).
                Mask out before attention. 0 is visible and 1 is invisible.
                [batch_size, # of agents, # of entities]
            agent_mask: Which agents/entities are not available. Zero out their outputs tow
                prevent gradients from flowing back. Shape of 2nd dim determines
                whether to compute queries for all entities or just agents.
                batch size, # of agents or entities]

        Return:
            tuple(attn, attn_weights):
            - attn: attentioned entities state., [batch size, n_agents, embedding dimension]
            - attn_weights: attention weights from agent to entities. [batch size, n_agents, embedding dimension]
        """
        batch_size, n_entities, embed_dim = entities.shape

        assert embed_dim == self.embed_dim, "Wrong embedding dimension, check input feature dimension."
        assert obs_mask.shape == torch.Size((batch_size, self.n_agents, n_entities)), "Wrong observation mask shape."
        assert agent_mask.shape == torch.Size((batch_size, self.n_agents)), "Wrong agent mask shape."

        # Entities we can control.
        agents = entities[:, :self.n_agents]

        # calculate q, k, v.
        q = self.q_proj(agents)  # batch * agents * embed_dim
        k = self.k_proj(entities)  # batch * entities * embed_dim
        v = self.v_proj(entities)  # batch * entities * embed_dim

        q = q.reshape(batch_size, self.n_agents, 2 * self.num_heads, self.head_dim)  # batch * agents * (2 * heads) * head_dim
        k = k.reshape(batch_size, n_entities, 2 * self.num_heads, self.head_dim)  # batch * entities * (2 * heads) * head_dim
        v = v.reshape(batch_size, n_entities, self.num_heads, 2 * self.head_dim)  # batch * entities * heads * (2 * head_dim)

        q = q.transpose(1, 2)  # batch * (2 * heads) * agents * head_dim
        q = q * self.scaling
        k = k.transpose(1, 2)  # batch * (2 * heads) * entities * head_dim
        v = v.transpose(1, 2)  # batch * heads * entities * (2 * head_dim)

        # calculate attention weights.
        attn_weights = torch.matmul(q, k.transpose(-1, -2))  # batch * (2 * n_heads) * agents * entities

        # if obs_mask is not None:
        #     attn_mask = torch.where(obs_mask == 1, float("-inf"), 0.)
        #     if attn_weights.shape != attn_mask.shape:
        #     attn_weights = attn_weights + attn_mask

        if obs_mask is not None:
            # Apply obs mask to attention matrix.
            if attn_weights.shape != obs_mask.shape:
                # obs_mask hasn't been expanded to multi heads.
                obs_mask = obs_mask.unsqueeze(1).expand_as(attn_weights)

            attn_weights = attn_weights.masked_fill(obs_mask.bool(), -float('Inf'))


        attn_weights = F.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32
        ).type_as(attn_weights)

        attn_weights = torch.nan_to_num(attn_weights, nan=0)

        # calculate lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # split to two parts
        attn_weights = attn_weights.reshape(
            batch_size,      # batch
            self.num_heads,  # heads
            2,               # two parts of diff attention.
            self.n_agents,   # agents
            n_entities       # entities(targets)
        )

        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]  # differential

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)     # Normalization.
        attn = attn * (1 - self.lambda_init)    # scale to match ori transformer scale. Might not useful in MARL.
        attn = attn.transpose(1, 2).reshape(batch_size, self.n_agents, 2 * self.num_heads * self.head_dim)

        attn = self.out_proj(attn)  # batch * agents * embedding_dim

        if agent_mask is not None:
            # mask unavailable agents.
            attn_mask = 1 - agent_mask.unsqueeze(-1).expand_as(attn)
            attn = attn * attn_mask


        return attn, attn_weights

    @staticmethod
    def _lambda_inti_fn(depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)


if __name__ == '__main__':
    from utils.torch_optimizer import optimize_tensor_display
    optimize_tensor_display(torch)

    # Args for test.
    from types import SimpleNamespace
    args = SimpleNamespace()

    args.batch_size = 4

    args.n_agents = 6
    args.n_entities = 8

    args.attn_n_heads = 10
    args.attn_embedding_dim = 500

    # Inputs for test.
    test_entities = th.rand((args.batch_size, args.n_entities, args.attn_embedding_dim))
    test_pre_mask = th.randint(0, 2, (args.batch_size, args.n_agents, args.n_entities))
    test_post_mask = th.randint(0, 2, (args.batch_size, args.n_agents))

    # Single layer network.
    diff_attention_layer = EntityDiffAttnLayer(
        args=args,
        embed_dim=args.attn_embedding_dim,
        depth=1,
        num_heads=args.attn_n_heads
    )

    # Forward inputs.
    attention_out, attn_weight = diff_attention_layer(entities=test_entities, obs_mask=test_pre_mask, agent_mask=test_post_mask)

    loss = attention_out.sum()

    loss.backward()


    pass
