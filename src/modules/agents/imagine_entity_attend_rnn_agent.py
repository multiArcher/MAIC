from types import SimpleNamespace

import torch as th
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import ImagineEntityAttnLayer
from utils.rms_norm import RMSNorm
from utils.custom_logging import PyMARLLogger
from .agent import Agent
from .entity_attend_rnn_agent import EntityAttnRNNAgent


class ImagineEntityAttnRNNAgent(EntityAttnRNNAgent):
    def __init__(self, input_shape, args: SimpleNamespace):
        super(ImagineEntityAttnRNNAgent, self).__init__(input_shape, args)
        self.attn = ImagineEntityAttnLayer(args, self.attn_dim, self.n_heads)

    # copy from refil, used to calculate matrix logical
    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def entitymask2attnmask(self, entity_mask):
        bs, ts, na, ne = entity_mask.shape
        # agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - entity_mask.to(th.float)).reshape(bs * ts * na, ne, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts * na, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, na, ne, ne).to(th.uint8)

    def forward(self, inputs, hidden_state, ret_attn_logits=None, msg=None, ret_attn_weights=False):
        # Head
        own_feats, ally_feats, enemy_feats, last_actions, agent_id = inputs
        batch_size, time_size, _, _ = own_feats.shape

        # Feature embedding. (own_feats_dim, enemy_feats_dim, ally_feats_dim, ...) -> hidden_dim(entity_dim)
        own_embedding = self.own_embed(own_feats)
        ally_embedding = self.ally_embed(ally_feats)
        enemy_embedding = self.enemy_embed(enemy_feats)

        # TODO: pymarl3 use sum to concreate three own embeddings. Maybe a learnable weight is better.
        if self.agent_id_embed is not None:
            agent_id_embedding = self.agent_id_embed(agent_id)
            own_embedding = own_embedding + agent_id_embedding

        if self.last_action_embed is not None:
            last_action_embedding = self.last_action_embed(last_actions)
            own_embedding = own_embedding + last_action_embedding

        entities = th.cat([own_embedding.unsqueeze(-2), ally_embedding, enemy_embedding], dim=-2)

        # with open(".tmp.txt", "w") as f:
        #     f.write(str(entities.shape))
        # print(entities.shape)
        batch_size, time_size, n_agents, n_entities, _ = entities.shape

        # create random split of entities (once per episode)
        groupin_probs = th.rand(batch_size, 1, n_agents, 1, device=entities.device).repeat(1, 1, 1, n_entities)

        groupin = th.bernoulli(groupin_probs).to(th.uint8)
        groupout = self.logical_not(groupin)

        # convert entity mask to attention mask
        groupinattnmask = self.entitymask2attnmask(groupin)
        groupoutattnmask = self.entitymask2attnmask(groupout)
        # create attention mask for interactions between groups
        interactattnmask = self.logical_or(self.logical_not(groupinattnmask),
                                           self.logical_not(groupoutattnmask))
        # get within group attention mask
        withinattnmask = self.logical_not(interactattnmask)

        entities = entities.repeat(2, 1, 1, 1, 1)
        # no obs_mask, so dim * 2 not like source code * 3
        attn_mask = th.cat([withinattnmask, interactattnmask], dim=0)
        print("hidden:")
        print(hidden_state.shape)
        hidden_state = hidden_state.repeat(2, 1, 1)
        print(hidden_state.shape)
        # Encoding  hidden_dim -> attn_dim

        # A single transformer encoder.
        # TODO: Test multiple structure of attention.
        x = self.encoding(entities)  # TODO: Maybe not useful because all information has already been embedded.
        x = self.norm(x[..., 0, :] + self.attn(x, attn_mask))
        x = self.norm(x + self.feedforward(x))

        # TODO: After the first entity attention layer, the rest should be self attention layer. Not implemented.

        # Output.   attn_dim -> n_actions
        # TODO: RNN might not be advanced. Transformer decoder seems to work here.
        x = self.rnn_proj(x)     # attn_dim -> hidden_dim
        h = hidden_state.reshape(-1, self.hidden_dim)
        hs = []
        for t in range(time_size):
            curr_x = x[:, t].reshape(-1, self.hidden_dim)
            h = self.rnn(curr_x, h)
            hs.append(h.reshape(batch_size * 2, self.n_agents, self.hidden_dim))
        hs = torch.stack(hs, dim=1)
        q = self.decoding(hs)
        # q = q.reshape(batch_size, time_size, self.args.n_agents, -1)
        return q, hs
