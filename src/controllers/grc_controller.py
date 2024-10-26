import torch

from modules.agents import get_agent
from components.action_selectors import get_action_selector
# from typing import Dict, Tuple, List


# This multi-agent controller shares parameters between agents
class GRCMAC():
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_states = None

        self.action_selector = get_action_selector(args.action_selector, args)

        self._build_agents(self._get_obs_shape(scheme))

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_obs = self._build_obs(ep_batch, t)    # batch * n_agent * dim
        messages = self._build_messages(agent_obs, ep_batch, t)  # batch * n_agent * dim

        # Message info is lost where the mask is True.
        missing_feature_mask = self._get_missing_feature_mask(messages)
        masked_message = messages.clone()
        masked_message[~missing_feature_mask] = float("nan")

        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_obs, masked_message, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.args.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs

    def _get_missing_feature_mask(self, message: torch.Tensor) -> torch.Tensor:
        """
        Return mask with the same shape of message indicating whether each feature is present or missing.
        If `type`='uniform', then each feature of each node is missing uniformly at random with probability `rate`.
        Instead, if `type`='structural', either we observe all features for a node, or we observe none. For each node
        there is a probability of `rate` of not observing any feature.
        True is existing data and False is missing data.
        """
        # Simulate message loss in communication.
        if type == "structural":  # either remove all of a nodes features or none
            prob = torch.ones(message.shape[:-1]) * (1 - self.args.message_missing_rate)
            mask = torch.bernoulli(prob).unsqueeze(-1).expand(message.shape)
        else:
            prob = torch.ones_like(message) * (1 - self.args.message_missing_rate)
            mask = torch.bernoulli(prob)

        return mask.bool().to(self.args.device)

    def _build_messages(self, agent_obs, ep_batch, t: int) -> torch.Tensor:
        # Should return (batch_size * n_agents) * (n_agents) * message_feature_dim
        # Representing every agent has (n_agents - 1) messages from teammates.
        messages = self.agent.generate_message(agent_obs)  # Every agent generate the message it sends.

        message_received = self._fully_connected_communication(messages)
        # Use a mean of all obs as message
        return message_received

    @staticmethod
    def _fully_connected_communication(messages: torch.Tensor) -> torch.Tensor:
        """
        Fully connected communication function to represent communication between agents.
        Args:
            messages: messages in shape (batch, n_agents, dim)

        Returns:
            received messages in shape (batch, n_agents, n_agents, dim)

        """
        batch_size, n_agents, message_dim = messages.shape\

        # (batch, n_agents, message_dim) -> (batch, n_agents, n_agents - 1, message_dim)
        messages_expanded = messages.unsqueeze(1).repeat(1, n_agents, 1, 1)

        # mask self message
        # mask = torch.eye(n_agents).bool().unsqueeze(0).repeat(batch_size, 1, 1).to(messages.device)
        #
        # received_messages = communication_matrix_expanded.masked_fill(mask.unsqueeze(-1), 0)
        # received_messages = received_messages[~mask].view(batch_size, n_agents, n_agents - 1, dim)

        return messages_expanded

# region MAC functions
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.to(self.args.device)

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        # self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.agent = get_agent(self.args.agent, input_shape, self.args)

    def _get_obs_shape(self, scheme: dict) -> int:
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _build_obs(self, batch, t) -> torch.Tensor:
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        obs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                obs.append(torch.zeros_like(batch["actions_onehot"][:, t]))
            else:
                obs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            obs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        obs = torch.cat([x for x in obs], dim=2)
        return obs
    # endregion
