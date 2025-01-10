from types import SimpleNamespace

import torch as th
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import ImagineEntityAttnLayer
from modules.layers.rms_norm import RMSNorm
from sympy import false
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

    def forward(self, inputs, hidden_state, ret_attn_logits=None, msg=None, ret_attn_weights=False, imagine=False):
        entities = torch.cat(
            [layer(feature_input) for feature_input, layer in zip(inputs, self.embedding_layers)],
            dim=-2
        )  # batch * time * n_agents * n_entities * hidden_dim

        # with open(".tmp.txt", "w") as f:
        #     f.write(str(entities.shape))
        # print(entities.shape)
        batch_size, time_size, n_agents, n_entities, _ = entities.shape
        obs_mask = th.zeros(batch_size, time_size, n_entities, n_entities).to(entities.device)
        entity_mask = th.zeros(batch_size, time_size, n_entities).to(entities.device)

        if imagine:
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

            entities = entities.repeat(3, 1, 1, 1, 1)
            entity_mask = entity_mask.repeat(3, 1, 1)
            agent_mask = entity_mask[:, :self.args.n_agents]
            # # na * ne * ne -> ne * ne, every agent use 1 * ne
            t_withinattnmask = withinattnmask[:, :, 0, :, :]
            t_interactattnmask = interactattnmask[:, :, 0, :, :]
            for i in range(n_agents):
                t_withinattnmask[:, :, i, :] = withinattnmask[:, :, i, i, :]
                t_interactattnmask[:, :, i, :] = interactattnmask[:, :, i, i, :]
            obs_mask = th.cat(
                [obs_mask, t_withinattnmask.repeat(1, time_size, 1, 1), t_interactattnmask.repeat(1, time_size, 1, 1)],
                dim=0)

            hidden_state = hidden_state.repeat(1, 3, 1, 1)

            batch_size *= 3

        # A single transformer encoder.
        # TODO: Test multiple structure of attention.
        x = self.encoding(entities)  # TODO: Maybe not useful because all information has already been embedded.
        x = self.norm1(x[..., 0, :] + self.attn(x, obs_mask))
        x = self.norm2(x + self.feedforward(x))
        # TODO: After the first entity attention layer, the rest should be self attention layer. Not implemented.

        # Output.   attn_dim -> n_actions
        # TODO: RNN might not be advanced. Transformer decoder seems to work here.
        x = self.rnn_proj(x)  # batch * time * n_agents * hidden_dim

        x = x.transpose(1, 2).reshape(batch_size * n_agents, time_size,
                                      self.hidden_dim)  # b * t * n * d -> b * n * t * d -> (b * n) * t * d
        h = hidden_state.reshape(self.gru_layers, batch_size * n_agents,
                                 self.hidden_dim)  # layers * (batch * n_agents) * hidden_dim

        x, h = self.rnn(x, h)  # GRU forward.

        x = x.reshape(batch_size, n_agents, time_size, self.hidden_dim).transpose(1,
                                                                                  2)  # (b * n) * t * d -> b * n * t * d -> b * t * n * d
        h = h.reshape(self.gru_layers, batch_size, n_agents, self.hidden_dim)

        q = self.decoding(x)

        q = q.reshape(batch_size, time_size, self.args.n_agents, -1)
        if not imagine:
            return q, h

        return q, h, (t_withinattnmask.repeat(1, time_size, 1, 1), t_interactattnmask.repeat(1, time_size, 1, 1))
