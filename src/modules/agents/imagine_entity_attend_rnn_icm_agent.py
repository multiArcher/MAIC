from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import EntityAttentionLayer
from modules.layers.rms_norm import RMSNorm
from utils.custom_logging import PyMARLLogger
from .agent import Agent


class FiniteDist(nn.Module):
    def __init__(self, in_dim, out_dim, device, limit):
        super(FiniteDist, self).__init__()
        assert out_dim % 2 == 0
        self.limit = limit
        self.msg_dim = out_dim // 2
        self.fc1 = nn.Linear(in_dim, self.msg_dim)
        self.fc2 = nn.Linear(in_dim, self.msg_dim)
        self.max_logvar = nn.Parameter((th.ones((1, self.msg_dim)).float() / 2).to(device), requires_grad=False)
        self.min_logvar = nn.Parameter((-th.ones((1, self.msg_dim)).float() * 10).to(device), requires_grad=False)

    def forward(self, x, limit=True):
        mean = self.fc1(x)
        if self.limit:
            mean = th.tanh(mean)
        logvar = self.fc2(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return th.cat([mean, logvar], dim=-1)


class EntityAttnRNNAgentICM(Agent):
    def __init__(self, input_scheme, args: SimpleNamespace):
        super().__init__(input_scheme, args)
        self.args = args
        self.device = args.device
        self.n_heads: int = getattr(args, "agent_attn_heads")  # Attention heads.
        self.hidden_dim: int = getattr(args, "agent_hidden_dim")  # Dimension of embedding andRNN.
        self.attn_dim: int = getattr(args, "agent_attn_dim")  # Dimension of full attention layer
        self.gru_layers: int = getattr(args, "agent_gru_layers")  # Number of GRU layers.

        # Embedding layers: scheme -> hidden_dim
        self.embedding_layers = nn.ModuleList()
        for feat_name, feat_shape in input_scheme[0].items():
            self.embedding_layers.append(
                nn.Linear(feat_shape[1], self.hidden_dim, bias=False, device=self.device)
            )

        # Embedding layers: 1 -> hidden_dim
        for feat_name, feat_shape in input_scheme[1].items():
            self.embedding_layers.append(
                nn.Embedding(feat_shape[1], self.hidden_dim, device=self.device)
            )

        # Encoding layers: hidden_dim -> attn_dim
        self.encoding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim, device=self.device),
            nn.LeakyReLU(inplace=True),
        )

        self.attn = EntityAttentionLayer(self.attn_dim, self.attn_dim, self.attn_dim, args)
        self.norm1 = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True).to(self.device)
        self.feedforward = nn.Linear(self.attn_dim, self.attn_dim, bias=False, device=self.device)
        self.norm2 = RMSNorm(self.attn_dim, eps=1e-5, elementwise_affine=True).to(self.device)

        # Output layers: attn_dim -> hidden_dim -> n_actions q
        self.rnn_proj = nn.Linear(self.attn_dim, self.hidden_dim, device=self.device)
        self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=self.gru_layers, batch_first=True,
                          device=self.device)

        self.fc_a1 = nn.Linear(self.attn_dim * (1 + args.self_loc) * 2, self.hidden_dim)
        self.fc_a2 = nn.Linear(self.hidden_dim, args.n_actions)

        self.decoding = nn.Linear(self.hidden_dim, args.n_actions)  # h -> q

        # new network used to cama
        if not args.sp_use_same_attn:
            self.sp_encoding = nn.Sequential(
                nn.Linear(self.hidden_dim, self.attn_dim),
                nn.LeakyReLU(),
            )  # hidden_dim -> attn_dim
            self.sp_attn = EntityAttentionLayer(self.attn_dim, self.attn_dim, self.attn_dim, args)
        self.fc_private = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attn_dim),
            nn.LeakyReLU(),
        )

        # coach message used
        # self.fc1_coach = nn.Linear(input_shape, args.attn_embed_dim)
        # self.coach_attn = EntityAttentionLayer(args, self.attn_dim, self.n_heads)

        attn_out_dim = self.attn_dim * (1 + args.self_loc + args.reserve_ori_f + args.double_attn)
        # mi message used
        if args.rnn_message:
            self.rnn_mess_b = nn.Linear(attn_out_dim * (1 + self.args.club_mi), args.rnn_hidden_dim)
            self.rnn_message = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
            self.fc_msg = FiniteDist(args.rnn_hidden_dim, args.msg_dim * 2, args.device, args.limit_msg)

        else:
            self.fc_msg = FiniteDist(attn_out_dim * (1 + self.args.club_mi), args.msg_dim * 2, args.device,
                                     args.limit_msg)

        self.fc_q = nn.Sequential(nn.Linear(attn_out_dim + args.club_mi * args.agent_attn_dim, args.agent_attn_dim),
                                  # self.fc_q = nn.Sequential(nn.Linear(attn_out_dim, args.attn_embed_dim),
                                  nn.ReLU(),
                                  FiniteDist(args.agent_attn_dim, args.msg_dim * 2, args.device, args.limit_msg))
        if args.add_q:
            self.adhoc_q_net = nn.Sequential(nn.Linear(args.msg_dim, args.agent_attn_dim),
                                             nn.ReLU(),
                                             nn.Linear(args.agent_attn_dim, args.n_actions))

        if self.args.group == "dpp":
            self.cos = nn.CosineSimilarity(dim=3)
            self.ally = None

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.own_embed.weight.new(1, self.args.agent_hidden_dim).zero_()

        self.attn_weights = None
        if self.args.rnn_message:
            return torch.zeros(self.gru_layers, self.args.agent_hidden_dim, device=self.device), torch.zeros(
                self.gru_layers, self.args.agent_hidden_dim, device=self.device)
        self.msg = np.zeros([1, 0, self.args.n_agents, self.args.msg_dim])
        return torch.zeros(self.gru_layers, self.args.agent_hidden_dim, device=self.device)

    def get_club_message(self, f1, f2, bs, ts, randidx=None, hidden_state=None):
        # f1->f_i f2->f_-i
        hs = None
        if self.args.rnn_message:
            x3_coach = F.relu(self.rnn_mess_b(th.cat([f1, f2], dim=-1)))
            x3_coach = x3_coach.reshape(bs, ts, self.args.n_agents, -1)
            h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hs = []
            for t in range(ts):
                curr_x3 = x3_coach[:, t].reshape(-1, self.args.rnn_hidden_dim)
                h = self.rnn_message(curr_x3, h)
                hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
            hs = th.stack(hs, dim=1)  # Concat over time
            zt_logits = self.fc_msg(hs)
        else:
            zt_logits = self.fc_msg(th.cat([f1, f2], dim=-1))
        zt_logits = zt_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim * 2)
        mean = zt_logits[:, :, :, :self.args.msg_dim]
        if self.args.save_entities_and_msg:
            self.msg = np.concatenate([self.msg, mean.detach().cpu().numpy()], axis=1)
        logvar = zt_logits[:, :, :, self.args.msg_dim:]
        zt_dis = th.distributions.Normal(mean, logvar.exp().sqrt())
        if self.training:
            zt = zt_dis.rsample()
        else:
            zt = zt_dis.mean
        if randidx is not None:
            flattern_f1 = f1.reshape(bs, ts, self.args.n_agents, -1)
            flattern_f1 = flattern_f1[:, :-1].reshape(bs * (ts - 1) * self.args.n_agents,
                                                      -1)  # del the last t corresponding to the mask.
            flattern_f2 = f2.reshape(bs, ts, self.args.n_agents, -1)
            flattern_f2 = flattern_f2[:, :-1].reshape(bs * (ts - 1) * self.args.n_agents, -1)
            oriidx = randidx.chunk(2, dim=0)[0]
            oriidx = oriidx.repeat(2)
            if self.args.rnn_message:  # need a q to estimate p.
                msg_q_logits_club = self.fc_q(th.cat([flattern_f1[randidx], flattern_f2[oriidx]], dim=-1))  # for club
                msg_q_logits_logq = self.fc_q(th.cat([f1.detach(), f2.detach()], dim=-1))
                msg_q_logits = [msg_q_logits_club, msg_q_logits_logq]
            else:
                msg_q_logits = self.fc_msg(th.cat([flattern_f1[randidx], flattern_f2[oriidx]], dim=-1))
        else:
            msg_q_logits = zt_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim * 2)
        return zt, zt_logits, msg_q_logits, hs

    def get_coach_message(self, inputs, feature, randidx=None, hidden_state=None):
        # unused code
        entities, obs_mask, entity_mask = inputs
        bs, ts, na, ne, ed = entities.shape
        entities = entities.reshape(bs * ts * na, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts * na, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        attn_mask = 1 - th.bmm((1 - agent_mask.to(th.float)).unsqueeze(2),
                               (1 - entity_mask.to(th.float)).unsqueeze(1))
        x1_coach = F.relu(self.fc1_coach(entities))
        x2_coach = self.coach_attn(x1_coach, pre_mask=attn_mask.to(th.uint8),
                                   post_mask=agent_mask)
        hs = None
        if self.args.rnn_message:
            x3_coach = F.relu(self.rnn_mess_b(x2_coach))
            x3_coach = x3_coach.reshape(bs, ts, self.args.n_agents, -1)
            h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            hs = []
            for t in range(ts):
                curr_x3 = x3_coach[:, t].reshape(-1, self.args.rnn_hidden_dim)
                h = self.rnn_message(curr_x3, h)
                hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
            hs = th.stack(hs, dim=1)  # Concat over time
            zt_logits = self.fc_msg(hs)
        else:
            zt_logits = self.fc_msg(x2_coach)
        zt_logits = zt_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim * 2)
        mean = zt_logits[:, :, :, :self.args.msg_dim]
        logstd = self.max_logvar - F.softplus(self.max_logvar - zt_logits[:, :, :, self.args.msg_dim:])
        logstd = self.min_logvar + F.softplus(logstd - self.min_logvar)
        zt_dis = th.distributions.Normal(mean, logstd.exp().sqrt())
        if self.training:
            if randidx is not None:
                zt = zt_dis.rsample()
            else:
                zt = zt_dis.sample()
        else:
            zt = zt_dis.mean
        if self.args.club_mi:
            if randidx is not None:
                flattern_feature = feature.reshape(bs * ts * self.args.n_agents, -1)
                flattern_x2_coach = x2_coach.reshape(bs * ts * self.args.n_agents, -1)
                oriidx = randidx.chunk(2, dim=0)[0]
                oriidx = oriidx.repeat(2)
                # msg_q_logits = self.fc_q(flattern_feature[randidx])
                msg_q_logits = self.fc_q(th.cat([flattern_feature[randidx], flattern_x2_coach[oriidx]], dim=-1))
            else:
                msg_q_logits = self.fc_q(th.cat([feature, x2_coach], dim=-1))
                # msg_q_logits = self.fc_q(feature)
                msg_q_logits = msg_q_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim * 2)
        else:
            msg_q_logits = self.fc_q(feature)
            msg_q_logits = msg_q_logits.reshape(bs, ts, self.args.n_agents, self.args.msg_dim * 2)
        return zt, zt_logits, msg_q_logits, hs

    def get_feature(self, inputs, return_F=False, only_F=False, rank_percent=None, pre_obs_mask=False,
                    ret_attn_weights=False):
        # use MHA get f_i
        # entities, obs_mask, entity_mask = inputs

        entities, obs_mask, entity_mask = inputs
        bs, ts, na, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, na, ne, ed)
        if pre_obs_mask:
            obs_mask = obs_mask.reshape(bs * ts * self.args.agent_attn_heads, self.args.n_agents, ne)
        else:
            obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]

        if only_F and not self.args.sp_use_same_attn:  # means it is s_p and needs other nets
            cur_encoding = self.sp_encoding
            cur_attn = self.sp_attn
        else:
            cur_encoding = self.encoding
            cur_attn = self.attn
        x1 = cur_encoding(entities)  # bs * ts * na * ne * hd
        # use multihead attention
        if rank_percent is not None:
            x2, true_pre_mask = cur_attn(x1, pre_mask=obs_mask, post_mask=agent_mask, rank_percent=rank_percent,
                                         entity_mask=entity_mask, ret_attn_weights=ret_attn_weights)
        else:
            x2 = cur_attn(x1, pre_mask=obs_mask, post_mask=agent_mask, ret_attn_weights=ret_attn_weights)
        if ret_attn_weights:
            x2, attn_weights = x2
            if self.attn_weights is None:
                self.attn_weights = attn_weights
            else:
                self.attn_weights = th.cat([self.attn_weights, attn_weights], dim=0)

        if return_F and only_F:
            return x2
        if rank_percent is not None:
            return x2, bs, ts, true_pre_mask.reshape(bs, ts, self.args.agent_attn_heads, self.args.n_agents, ne)
        # why return batch_size and time_size?
        return x2, bs, ts

    def get_q(self, x2, bs, ts, hidden_state, return_F=False, zt=None, h1=None):
        # get q value use rnn
        if zt is not None and not self.args.add_q:
            _, _, na, edim = zt.shape
            if self.args.no_msg:
                x3 = F.relu(self.rnn_proj(th.cat([x2, th.zeros(bs * ts, na, edim).to(x2.device)], dim=-1)))
            else:
                x3 = F.relu(self.rnn_proj(th.cat([x2, zt.reshape(bs * ts, na, edim)], dim=-1)))
        else:
            x3 = F.relu(self.rnn_proj(x2))

        x = x3.transpose(1, 2).reshape(bs * self.args.n_agents, ts,
                                       self.hidden_dim)  # b * t * n * d -> b * n * t * d -> (b * n) * t * d
        h = hidden_state.reshape(self.gru_layers, bs * self.args.n_agents,
                                 self.hidden_dim)  # layers * (batch * n_agents) * hidden_dim

        x, h = self.rnn(x, h)  # GRU forward.

        x = x.reshape(bs, self.args.n_agents, ts, self.hidden_dim).transpose(1,
                                                                             2)  # (b * n) * t * d -> b * n * t * d -> b * t * n * d
        h = h.reshape(self.gru_layers, bs, self.args.n_agents, self.hidden_dim)

        q = self.decoding(x)
        # zero out output for inactive agents
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        # q = q.reshape(bs * self.args.n_agents, -1)
        # arm q plus
        if zt is not None and self.args.add_q:
            self.adhoc_q = self.adhoc_q_net(zt)
            if not self.args.no_msg:
                q += self.adhoc_q
        if h1 is not None:
            h = [h1, h]
        if return_F:
            return q, h, x2
        return q, h

    def get_inputs_m(self, inputs, true_pre_mask=None):
        entities, obs_mask, entity_mask = inputs
        if true_pre_mask is not None:
            c_mask = self.logical_not(true_pre_mask)  # bs, ts, n_head, na, ne
            inputs_m = (entities, c_mask, entity_mask)
        else:
            c_mask = self.logical_not(obs_mask)
            entities = entities.repeat(2, 1, 1, 1, 1)
            obs_mask = th.cat([obs_mask, c_mask], dim=0)
            entity_mask = entity_mask.repeat(2, 1, 1)
            inputs_m = (entities, obs_mask, entity_mask)
        return inputs_m

    def forward(self, inputs, hidden_state, imagine=False, return_F=False, only_F=False, randidx=None,
                ret_attn_weights=False):
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
        tmp_entity_mask = th.ones(batch_size, time_size, n_agents, n_entities)
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
            # attn_mask = th.cat([withinattnmask, interactattnmask], dim=0)
            # for i in range(n_agents):
            #     tmp_mask[:, :, i, :] = attn_mask[:, :, i, i, :]
            # # obs_mask = tmp_mask.repeat(1, time_size, 1, 1)
            # obs_mask = th.cat([tmp_mask, obs_mask], dim=0)
            # tmp_mask = attn_mask[:, :, 0, :, :]
            # # obs_mask = tmp_mask.repeat(1, time_size, 1, 1)
            # obs_mask = th.cat([tmp_mask, obs_mask], dim=0)

            # # na * ne * ne -> ne * ne, every agent use 1 * ne
            t_withinattnmask = withinattnmask[:, :, 0, :, :]
            t_interactattnmask = interactattnmask[:, :, 0, :, :]
            for i in range(n_agents):
                t_withinattnmask[:, :, i, :] = withinattnmask[:, :, i, i, :]
                t_interactattnmask[:, :, i, :] = interactattnmask[:, :, i, i, :]
            obs_mask = th.cat(
                [obs_mask, t_withinattnmask.repeat(1, time_size, 1, 1), t_interactattnmask.repeat(1, time_size, 1, 1)],
                dim=0)

            if self.args.rnn_message:
                hidden_state = [h.repeat(1, 3, 1, 1) for h in hidden_state]
            else:
                hidden_state = hidden_state.repeat(1, 3, 1, 1)

        inputs = (entities, obs_mask, entity_mask)

        if return_F and only_F:
            return self.get_feature(inputs, return_F=return_F, only_F=only_F, ret_attn_weights=ret_attn_weights)
        if self.args.mi_message:
            if self.args.rnn_message:
                h1, h2 = hidden_state
            else:
                h1, h2 = None, hidden_state
            if self.args.club_mi:
                # msg_q_logits_i=f(s^g_i, s^l_j)
                x2, batch_size, time_size, true_pre_mask = self.get_feature(inputs, return_F=return_F, only_F=only_F,
                                                                            rank_percent=self.args.rank_percent,
                                                                            ret_attn_weights=ret_attn_weights)
                inputs_m = self.get_inputs_m(inputs, true_pre_mask=true_pre_mask)
                # bs * ts * ne * ed
                x2_m, _, _ = self.get_feature(inputs_m, return_F=return_F, only_F=only_F, pre_obs_mask=True)
                # zeta t
                zt, zt_logits, msg_q_logits, h1 = self.get_club_message(x2, x2_m, batch_size, time_size,
                                                                        randidx=randidx,
                                                                        hidden_state=h1)
            else:
                inputs_m = self.get_inputs_m(inputs)
                x2, batch_size, time_size = self.get_feature(inputs_m, return_F=return_F, only_F=only_F,
                                                             ret_attn_weights=ret_attn_weights)
                x2, x2_m = x2.chunk(2, dim=0)
                batch_size //= 2
                zt, zt_logits, msg_q_logits, h1 = self.get_coach_message(inputs, x2_m, hidden_state=h1)
            # q, h, f = self.get_q(x2, batch_size, time_size, h2, return_F=return_F, zt=zt, h1=h1)
            if imagine:
                q, h, f = self.get_q(x2, batch_size, time_size, h2, return_F=return_F, zt=zt,
                                     h1=h1)
                return q, h, f, zt, zt_logits, msg_q_logits, (
                    t_withinattnmask.repeat(1, time_size, 1, 1), t_interactattnmask.repeat(1, time_size, 1, 1))
            else:
                q, h = self.get_q(x2, batch_size, time_size, h2, return_F=return_F, zt=zt,
                                  h1=h1)
                return q, h, zt, zt_logits, msg_q_logits
        else:
            x2, batch_size, time_size = self.get_feature(inputs, return_F=return_F, only_F=only_F,
                                                         ret_attn_weights=ret_attn_weights)
            # q, h, f = self.get_q(x2, batch_size, time_size, hidden_state, return_F=return_F)
            if imagine:
                return self.get_q(x2, batch_size, time_size, hidden_state, return_F=return_F), (
                    withinattnmask.repeat(1, time_size, 1, 1), interactattnmask.repeat(1, time_size, 1, 1))
            else:
                return self.get_q(x2, batch_size, time_size, hidden_state, return_F=return_F)
            # return q, h, f, (withinattnmask.repeat(1, time_size, 1, 1), interactattnmask.repeat(1, time_size, 1, 1))

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


class ImagineEntityAttnRNNAgentICM(EntityAttnRNNAgentICM):
    def __init__(self, *args, **kwargs):
        super(ImagineEntityAttnRNNAgentICM, self).__init__(*args, **kwargs)
        if self.args.group == "dpp":
            self.cos = nn.CosineSimilarity(dim=3)
            self.ally = None

    def forward(self, inputs, hidden_state, inputs_p=None, imagine=True, return_F=False, only_F=False, randidx=None,
                ret_attn_logits=None, msg=None,
                ret_attn_weights=False):
        # instead of ICM and MI_ICM
        if self.args.mi_message:
            if imagine:
                q, hs, xs, zt, zt_logits, msg_q_logits, m = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs,
                                                                                                              hidden_state,
                                                                                                              imagine=imagine,
                                                                                                              return_F=True,
                                                                                                              only_F=False,
                                                                                                              randidx=randidx)
                xs = xs.chunk(3, dim=0)[0]
            else:
                q, hs, xs, zt, zt_logits, msg_q_logits = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs,
                                                                                                           hidden_state,
                                                                                                           imagine=imagine,
                                                                                                           return_F=True,
                                                                                                           only_F=False,
                                                                                                           randidx=randidx)
            bs, ts, _, _, _ = inputs[0].shape
            xs = xs.reshape(bs, ts, self.args.n_agents, self.args.agent_attn_dim)
            xsp = th.cat([xs[:, 1:], xs[:, -1:]], dim=1)
            # xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
            x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
            logits = self.fc_a2(x)
            logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
            if imagine:
                return q, hs, m, logits, zt, zt_logits, msg_q_logits
            else:
                return q, hs, logits, zt, zt_logits, msg_q_logits
        else:
            if imagine:
                q, hs, xs, m = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs, hidden_state, imagine=imagine,
                                                                                 return_F=True, only_F=False)
                xs = xs.chunk(3, dim=0)[0]
            else:
                q, hs, xs = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs, hidden_state, imagine=imagine,
                                                                              return_F=True, only_F=False)
            xsp = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs_p, None, return_F=True, only_F=True)
            x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
            logits = self.fc_a2(x)
            bs, ts, _, _ = inputs[0].shape
            logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
            if imagine:
                return q, hs, m, logits
            else:
                return q, hs, logits

            # q, hs, xs, zt, zt_logits, msg_q_logits, m = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs,
            #                                                                                               hidden_state,
            #                                                                                               return_F=return_F,
            #                                                                                               only_F=only_F,
            #                                                                                               randidx=randidx,
            #                                                                                               ret_attn_weights=ret_attn_weights,
            #                                                                                               imagine=imagine)
            # xs = xs.chunk(3, dim=0)[0]
            # bs, ts, _, _ = inputs[0].shape
            # xs = xs.reshape(bs, ts, self.args.n_agents, self.args.agent_attn_dim)
            # xsp = th.cat([xs[:, 1:], xs[:, -1:]], dim=1)
            # # xsp = self.forward(inputs_sp, None, return_F=True, only_F=True)
            # x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
            # logits = self.fc_a2(x)
            # logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
            # return q, hs, m, logits, zt, zt_logits, msg_q_logits

            # q, hs, xs, m = super(ImagineEntityAttnRNNAgentICM, self).forward(inputs, hidden_state, return_F=return_F,
            #                                                                              only_F=only_F,
            #                                                                              ret_attn_weights=ret_attn_weights,
            #                                                                              imagine=imagine)
            #             xs = xs.chunk(3, dim=0)[0]
            #             xsp = self.forward(inputs_p, None, return_F=True, only_F=True)
            #             x = F.relu(self.fc_a1(th.cat([xs, xsp], dim=-1)))
            #             logits = self.fc_a2(x)
            #             bs, ts, _, _ = inputs[0].shape
            #             logits = logits.reshape(bs, ts, self.args.n_agents, self.args.n_actions)
            #             return q, hs, m, logits
