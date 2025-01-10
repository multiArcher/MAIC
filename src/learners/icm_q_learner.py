import copy

from modules.mixers.flex_qmix import FlexQMixer
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from learners.learner import Learner
# from modules.mixers.vdn import VDNMixer
# from modules.mixers.qmix import QMixer
from utils.maker import MixerMaker
import torch as th
from torch.nn.functional import cross_entropy as CE
from torch.optim import RMSprop, Adam
import torch.distributions as D
import torch.nn.functional as F


class ICMQLearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None

        if args.mixer is not None:
            self.mixer = MixerMaker.make(args.mixer, args)

            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.params_logq = list(self.mac.agent.fc_q.parameters())
        if self.args.club_mi:
            self.cur_max_log_q = True
            self.club_count = 0
            self.loss_club = th.tensor(0)
            self.optimiser_logq = Adam(params=self.params_logq, lr=args.lr, weight_decay=args.weight_decay)
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=args.weight_decay)
        # self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = args.device
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)

    def _get_mixer_ins(self, batch, repeat_batch=1, keep_last_dim=False):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1),
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            if not keep_last_dim:
                return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                         batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                        (entities[:, 1:],
                         batch["entity_mask"][:, 1:]))
            return ((entities.repeat(repeat_batch, 1, 1, 1),
                     batch["entity_mask"].repeat(repeat_batch, 1, 1)),
                    (entities[:, 1:],
                     batch["entity_mask"][:, 1:]))

    def get_dist_from_logits(self, logits, return_mv=False):
        logits_mean = logits[..., :self.args.msg_dim]
        logits_logvar = logits[..., self.args.msg_dim:]
        if return_mv:
            return logits_mean, logits_logvar.exp()
        dis = D.Normal(logits_mean, logits_logvar.exp().sqrt())
        return dis

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.mac.train()
        self.mixer.train()
        self.target_mac.eval()
        self.target_mixer.eval()
        randidx = None
        if self.args.mi_message and self.args.club_mi:
            club_mask = mask.expand(-1, -1, self.args.n_agents)
            idx = club_mask.reshape(-1).nonzero(as_tuple=True)[0]
            randidx = idx[th.randperm(len(idx))]
            randidx = th.cat([idx, randidx], dim=0)

        if 'imagine' in self.args.agent:
            if self.args.mi_message:
                all_mac_out, groups, logits, zt, zt_logits, msg_q_logits = self.mac.forward(batch, t=None, imagine=True,
                                                                                            train_mode=True,
                                                                                            randidx=randidx)
                zt = zt.chunk(3, dim=0)[0]
                zt_logits = zt_logits.chunk(3, dim=0)[0]
                if not self.args.club_mi:
                    msg_q_logits = msg_q_logits.chunk(3, dim=0)[0]
                if self.args.add_q:
                    adhoc_q = self.mac.agent.adhoc_q.chunk(3, dim=0)[0]
                    adhoc_q = th.gather(adhoc_q, dim=3, index=actions).squeeze(3)
            else:
                all_mac_out, groups, logits = self.mac.forward(batch, t=None, imagine=True, train_mode=True)
            # Pick the Q-Values for the actions taken by each agent
            rep_actions = actions.repeat(3, 1, 1, 1)
            all_chosen_action_qvals = th.gather(all_mac_out, dim=3, index=rep_actions).squeeze(3)  # Remove the last dim
            mac_out = all_mac_out.chunk(3, dim=0)[0]
            chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)
            caqW, caqI = caqW[:, :-1], caqI[:, :-1]
            # caqW, caqI = caqW, caqI
            caq_imagine = th.cat([caqW, caqI], dim=2)
        else:
            if self.args.mi_message:
                mac_out, logits, zt, zt_logits, msg_q_logits = self.mac.forward(batch, t=None, train_mode=True,
                                                                                randidx=randidx)
                if self.args.add_q:
                    adhoc_q = th.gather(self.mac.agent.adhoc_q, dim=3, index=actions).squeeze(3)
            else:
                mac_out, logits = self.mac.forward(batch, t=None, train_mode=True)
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out, dim=3, index=actions).squeeze(
                3)  # Remove the last dim, but not here, for icm.
        logits = logits[:, :-1]
        if self.args.mi_message:
            zt, zt_logits, = zt[:, :-1], zt_logits[:, :-1]
            if not self.args.club_mi:
                msg_q_logits = msg_q_logits[:, :-1]
        self.target_mac.init_hidden(batch.batch_size)

        target_mac_out = self.target_mac.forward(batch, t=None)
        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        loss = 0

        batch_size, time_size, _ = target_max_qvals.shape
        agent_mask = mask.expand_as(target_max_qvals)  # available 1, others 0, [bs, t, n_agent]
        # Mix
        if self.mixer is not None:
            if self.args.global_icm:
                if 'imagine' in self.args.agent:
                    # mix_ins, targ_mix_ins = self._get_mixer_ins(batch,
                    #                                             keep_last_dim=True)  # mix_ins has one more dim on t.
                    # global_action_qvals, x2 = self.mixer(chosen_action_qvals, mix_ins, return_f=True)
                    global_action_qvals, x2 = self.mixer(
                        chosen_action_qvals, batch["state"], return_f=True
                    )
                    # don't need last timestep
                    groups = [gr[:, :-1] for gr in groups]
                    # mix_ins, _ = self._get_mixer_ins(batch)
                    # caq_imagine = self.mixer(caq_imagine, mix_ins, imagine_groups=groups)
                    caq_imagine = self.mixer(
                        caq_imagine, batch["state"][:, :-1], imagine_groups=groups
                    )
                else:
                    # mix_ins, targ_mix_ins = self._get_mixer_ins(batch, keep_last_dim=True)
                    # global_action_qvals, x2 = self.mixer(chosen_action_qvals, mix_ins, return_f=True)
                    global_action_qvals, x2 = self.mixer(
                        chosen_action_qvals, batch["state"], return_f=True
                    )
                global_action_qvals = global_action_qvals[:, :-1]
                a_logits = self.mixer.predict_action(x2[:, :-1], x2[:, 1:])  # [bs,t,n_agents, na]
                bs, ts, n_agent, na = a_logits.shape
                gce = CE(a_logits.reshape(-1, na), actions[:, :-1].reshape(-1), reduction="none")
                lg_mask = mask.expand_as(a_logits[:, :, :, 0])
                gce = gce.reshape(bs, ts, n_agent) * lg_mask * agent_mask
                gce_loss = gce.sum() / agent_mask.sum() * self.args.gce_weight
                loss += gce_loss
            else:
                if 'imagine' in self.args.agent:
                    # mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                    # global_action_qvals = self.mixer(chosen_action_qvals[:, :-1],
                    #                                  mix_ins)
                    global_action_qvals = self.mixer(chosen_action_qvals[:, :-1], batch["state"][:, :-1])
                    # don't need last timestep
                    groups = [gr[:, :-1] for gr in groups]
                    # caq_imagine = self.mixer(caq_imagine, mix_ins, imagine_groups=groups)
                    caq_imagine = self.mixer(caq_imagine, batch["state"][:, :-1], imagine_groups=groups)
                else:
                    # mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                    # global_action_qvals = self.mixer(chosen_action_qvals[:, :-1], mix_ins)
                    global_action_qvals = self.mixer(
                        chosen_action_qvals[:, :-1], batch["state"][:, :-1]
                    )

            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        l_mask = mask.expand_as(logits[:, :, :, 0])  # bs,ts,n_agent
        if self.args.mi_message:
            mi_mask = mask.expand_as(zt[:, :, :, 0])
            bs, t, na = mi_mask.shape
            mi_mask = mi_mask.permute(1, 0, 2).reshape(t, bs * na)
        # Td-error
        td_error = (global_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        q_loss = self.args.q_weight * (masked_td_error ** 2).sum() / mask.sum()

        if 'imagine' in self.args.agent:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            q_loss = (1 - im_prop) * q_loss + im_prop * im_loss

        # [bs,t,n_agent, na]
        bs, ts, n_agent, na = logits.shape
        ce = CE(logits.reshape(-1, na), actions[:,:-1].reshape(-1), reduction="none")

        ce = ce.reshape(bs, ts, n_agent) * l_mask * agent_mask
        ce_loss = ce.sum() / agent_mask.sum() * self.args.ce_weight
        loss += q_loss + ce_loss
        if self.args.mi_message:
            bs, t, ne, msg_d = zt.shape
            zt_ori = zt.detach()
            valid_t_mask = mi_mask.sum(-1) / ne >= 1  # t
            zt_dis = self.get_dist_from_logits(zt_logits.permute(1, 0, 2, 3).reshape(t, 1, bs * ne, msg_d * 2))
            if self.args.club_mi:
                if self.args.rnn_message:
                    msg_q_logits_club, msg_q_logits_logq = msg_q_logits
                    if 'imagine' in self.args.agent:
                        msg_q_logits_logq = msg_q_logits_logq.chunk(3, dim=0)[0]
                    msg_q_logits_logq = msg_q_logits_logq.reshape(bs, t + 1, ne, msg_d * 2)
                    msg_q_dis = self.get_dist_from_logits(
                        msg_q_logits_logq[:, :-1].permute(1, 0, 2, 3).reshape(t, 1, bs * ne, msg_d * 2))
                    zt_dis_detach = self.get_dist_from_logits(
                        zt_logits.detach().permute(1, 0, 2, 3).reshape(t, 1, bs * ne, msg_d * 2))
                    loss_logq = self.args.logq_weight * (
                            D.kl_divergence(zt_dis_detach, msg_q_dis).sum(-1) * mi_mask.unsqueeze(
                        1)).sum() / mi_mask.sum()
                    loss += loss_logq
                    self.club_count += 1
                    if self.club_count >= self.args.club_ratio:
                        msg_q_logits_pos, msg_q_logits_neg = msg_q_logits_club.detach().chunk(2, dim=0)
                        zt = zt.reshape(bs * ts * self.args.n_agents, -1)
                        zt = zt[idx]
                        pos = self.get_dist_from_logits(msg_q_logits_pos)
                        neg = self.get_dist_from_logits(msg_q_logits_neg)
                        loss_club = self.args.club_weight * (pos.log_prob(zt) - neg.log_prob(zt)).sum(-1).mean()
                        self.loss_club = loss_club.detach()
                        loss += loss_club
                        self.club_count = 0
                else:
                    msg_q_logits_pos, msg_q_logits_neg = msg_q_logits.chunk(2, dim=0)
                    zt = zt.reshape(bs * ts * self.args.n_agents, -1)
                    zt = zt[idx].detach()

                    # pos_mean, pos_var = self.get_dist_from_logits(msg_q_logits_pos, return_mv=True)
                    # neg_mean, neg_var = self.get_dist_from_logits(msg_q_logits_neg, return_mv=True)
                    # negative = - (neg_mean - zt)**2 / neg_var
                    # positive = - (pos_mean - zt)**2 / pos_var
                    # loss_club = self.args.club_weight*(positive-negative).sum(-1).mean()
                    pos = self.get_dist_from_logits(msg_q_logits_pos)
                    neg = self.get_dist_from_logits(msg_q_logits_neg)
                    self.loss_club = self.args.club_weight * (pos.log_prob(zt) - neg.log_prob(zt)).sum(-1).mean()
                    loss += self.loss_club
            else:
                # zt, zt_logits, msg_q_logits
                msg_q_dis = self.get_dist_from_logits(
                    msg_q_logits.permute(1, 0, 2, 3).reshape(t, 1, bs * ne, msg_d * 2))
                zt = zt.permute(1, 0, 2, 3).reshape(t, 1, bs * ne, msg_d)
                pm = zt_dis.log_prob(zt)
                qm = msg_q_dis.log_prob(zt)
                lib = self.args.ib_weight * ((pm - qm).sum(-1) * mi_mask.unsqueeze(1)).sum() / mi_mask.sum()
                loss += lib
            zt = zt_ori.permute(1, 0, 2, 3).reshape(t, bs * ne, 1, msg_d)
            mi_logits = zt_dis.log_prob(zt)  # t*(bs*ne)*(bs*ne)*msg_d
            mi_logits = mi_logits.sum(-1)  # t*(bs*ne)*(bs*ne)
            mi_logits = mi_logits.masked_fill((1 - mi_mask).unsqueeze(1).bool(), -1e8)
            mi_logits = mi_logits.masked_fill(th.logical_not(valid_t_mask).unsqueeze(1).unsqueeze(1),
                                              0)  # In case the whole batch is -inf.
            ince = D.Categorical(logits=mi_logits)  # t*(bs*ne)
            inds = th.arange(bs * ne, device=batch.device).unsqueeze(0).repeat(t, 1)  # t*(bs*ne)
            unnormed_mi_upper = ince.log_prob(inds)
            unnormed_mi_upper = unnormed_mi_upper.masked_fill((1 - mi_mask).bool(), 0)
            unnormed_mi_upper = unnormed_mi_upper[valid_t_mask].sum(-1) + th.log(mi_mask[valid_t_mask].sum(-1))
            mi_upper = unnormed_mi_upper / mi_mask[valid_t_mask].sum(-1)
            if self.args.club_mi:
                lia = -self.args.club_weight * (1 - self.args.beta) * mi_upper.mean()
            else:
                lia = -self.args.ia_weight * mi_upper.mean()
            loss += lia
            entropy_loss = self.args.entropy_weight * (
                    -zt_dis.entropy().sum(dim=-1) * mi_mask.unsqueeze(1)).sum() / mi_mask.sum()
            loss += entropy_loss
        if self.args.club_mi and self.args.rnn_message:
            self.optimiser_logq.zero_grad()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # try:
        #     grad_norm = grad_norm.item()
        # except:
        #     pass
        self.optimiser.step()
        if self.args.club_mi and self.args.rnn_message:
            self.optimiser_logq.step()

        self.training_steps += 1
        if (
                self.args.target_update_interval_or_tau > 1
                and (self.training_steps - self.last_target_update_step)
                / self.args.target_update_interval_or_tau
                >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals[:, :-1] * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
                self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                    self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.to(self.args.device)
        self.target_mac.to(self.args.device)
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
