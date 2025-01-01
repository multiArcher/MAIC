from collections import OrderedDict
from types import SimpleNamespace

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch
from .entity_controller import EntityMAC
from modules.agents.imagine_entity_attend_rnn_icm_agent import ImagineEntityAttnRNNAgentICM


class ICMMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super(ICMMAC, self).__init__(scheme, groups, args)
    
    def forward(self, ep_batch, t, test_mode=False, train_mode=False, imagine=False, randidx=None, ret_attn_weights=False):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True
        if self.args.mi_message:
            if train_mode:
                agent_inputs, agent_inputs_sp = self._build_inputs(ep_batch, t, sp=True)
                if imagine:
                    agent_outs, self.hidden_states, groups, logits, zt, zt_logits, msg_q_logits = self.agent(agent_inputs, self.hidden_states, agent_inputs_sp, randidx=randidx)
                else:
                    agent_outs, self.hidden_states, logits, zt, zt_logits, msg_q_logits = self.agent(agent_inputs, self.hidden_states, agent_inputs_sp, randidx=randidx)
            else:
                agent_inputs = self._build_inputs(ep_batch, t)
                agent_outs, self.hidden_states, _, _, _ = super(ImagineEntityAttnRNNAgentICM, self.agent).forward(agent_inputs, self.hidden_states)
            if int_t:
                return agent_outs.squeeze(1)
            if train_mode:
                if imagine:
                    return agent_outs, groups, logits, zt, zt_logits, msg_q_logits 
                else:
                    return agent_outs, logits, zt, zt_logits, msg_q_logits 
            return agent_outs
        else:
            if train_mode:
                agent_inputs, agent_inputs_sp = self._build_inputs(ep_batch, t, sp=True)
                if imagine:
                    agent_outs, self.hidden_states, groups, logits = self.agent.ICM(agent_inputs, agent_inputs_sp, self.hidden_states)
                else:
                    agent_outs, self.hidden_states, logits = self.agent.ICM(agent_inputs, agent_inputs_sp, self.hidden_states)
            else:
                agent_inputs = self._build_inputs(ep_batch, t)
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            if int_t:
                return agent_outs.squeeze(1)
            if train_mode:
                if imagine:
                    return agent_outs, groups, logits
                else:
                    return agent_outs, logits
            return agent_outs
    
    def init_hidden(self, batch_size):
        if self.args.rnn_message:
            single_hidden_state = self.agent.init_hidden().unsqueeze(1).unsqueeze(1)
            self.hidden_states = single_hidden_state.expand(-1, batch_size, self.n_agents, -1).contiguous()
            self.hidden_states = [x.unsqueeze(0).expand(batch_size, self.n_agents, -1) for x in self.hidden_states]
        else:
            single_hidden_state = self.agent.init_hidden().unsqueeze(1).unsqueeze(1)
            self.hidden_states = single_hidden_state.expand(-1, batch_size, self.n_agents, -1).contiguous()  # bav

    def _build_inputs(self, batch, t, sp=False):
        # Assumes homogenous agents with entity + observation mask inputs.
        batch_size, max_length, n_agents, obs_size = batch["obs"].shape
        obs_data = batch["obs"][:, t]
        time_size = obs_data.shape[1]
        inputs = []

        # Split obs by the mapping indices.
        split_obs = torch.split(obs_data, self.input_splits, dim=-1)
        for i, feat_shape in enumerate(self.input_scheme[0].values()):
            inputs.append(
                split_obs[i].reshape(batch_size, time_size, n_agents, *feat_shape)
            )

        if self.args.obs_last_action:
            # Add a one dim last_action. This is not one-hot. The Agent will handle it.
            last_actions = self._get_last_actions(batch, t, batch_size, n_agents)
            inputs.append(last_actions)

        if self.args.obs_agent_id:
            # Add a one dim agent_id. This is not one-hot. The Agent will handle it.
            agent_id = torch.arange(  # [1, 2, 3]
                self.n_agents, dtype=torch.int, device=batch.device,
            ).repeat(  # b * t * n_agents * 1
                batch_size,
                time_size,
                1,
            ).unsqueeze(-1)
            inputs.append(agent_id)

        if not sp:
            return inputs

        inputs_p = (torch.cat([inputs[0][:, 1:], inputs[0][:, -1:]], dim=1),
                    torch.cat([inputs[1][:, 1:], inputs[1][:, -1:]], dim=1),
                    torch.cat([inputs[2][:, 1:], inputs[2][:, -1:]], dim=1),)

        return inputs, inputs_p

    def _get_input_shape(self, scheme) -> (OrderedDict[str, tuple[int, int]], OrderedDict[str, tuple[int, int]]):
        """Compute and return the input scheme for agents.

        The input scheme includes observation components (e.g., own features, enemy features, ally features)
        and additional embedding information based on configuration.

        The embedding scheme includes last action and agent ID features. And
        other features need embedding before encoding.

        Args:
            scheme (dict): The scheme that defines the structure and shapes of the input data.

        Returns:
            tuple:
                - OrderedDict[str, tuple[int, int]]: Observation components with their shapes.
                - OrderedDict[str, tuple[int, int]]: Embedding scheme including last action and agent ID features.
    """
        # Input from env.
        input_scheme = self.obs_components
        embedding_scheme = OrderedDict()

        # Additional embedding features.
        if self.args.obs_last_action:
            embedding_scheme["embedding.last_action"] = (1, scheme["avail_actions"]["vshape"][0])
        if self.args.obs_agent_id:
            embedding_scheme["embedding.agent_id"] = (1, self.n_agents)

        return input_scheme, embedding_scheme