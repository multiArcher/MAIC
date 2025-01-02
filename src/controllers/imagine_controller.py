from collections import OrderedDict
from types import SimpleNamespace

import torch

from .mac import MAC
from utils.maker import AgentMaker, ActionSelectorMaker
from components.action_selectors.action_selector import ActionSelector
from utils.custom_logging import PyMARLLogger


# This multi-agent controller shares parameters between agents
class ImagineMAC(MAC):
    """Controller for entity wise env.

    Differences from MAC:
    - input_slices: list of slices to extract input data from obs into entity inputs.
    - input_scheme: tuple of input shape of agents in a tuple including
    (own_feats_dim, enemy_feats_dim, ally_feats_dim, Optional last_action_dim, Optional agent_id_dim)

    TODO: Testing inheriting from torch.nn.Module to enable features of nn.Module.
    TODO: breakpoints functions to be added in the future.
    """
    def __init__(self, scheme: dict, groups, args: SimpleNamespace):
        # Check if the env is in entity scheme
        if getattr(args, "entity_scheme", False) is False:
            class_name = self.__class__.__name__
            PyMARLLogger("main").get_child_logger(f"{class_name}").critical(f"{class_name} only works in entity scheme.")
            raise RuntimeError(f"{class_name} only works in entity scheme.")

        super(ImagineMAC, self).__init__(scheme, groups, args)

        self.args = args
        self.device: torch.device = args.device
        self.n_agents: int = args.n_agents

        # Obs components in a dict. Calculated in env_wrapper.
        self.obs_components: dict[str: tuple[int, int]] = args.env_info["obs_components"]
        # self.obs_partsL: int = len(self.obs_components) # number of obs parts
        self.input_scheme = self._get_input_shape(scheme)   # Scheme including normal features and embedding features.
        self.input_splits = self._init_entity_splits(self.input_scheme[0])  # A list for splitting obs data into entities.

        self.agent_output_type: str = args.agent_output_type
        self.action_selector: ActionSelector = ActionSelectorMaker.make(args.action_selector, args)

        self._build_agents(self.input_scheme)
        self.hidden_states = None

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def init_hidden(self, batch_size):
        single_hidden_state = self.agent.init_hidden().unsqueeze(1).unsqueeze(1)
        self.hidden_states = single_hidden_state.expand(-1, batch_size, self.n_agents, -1).contiguous()

    def forward(self, ep_batch, t, test_mode=False, *args, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)  # two list of tensor features.
        avail_actions = ep_batch["avail_actions"][:, t]

        if kwargs.get('imagine', False):
            agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, **kwargs)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)   # Agent forward

        # For politic action selection. Not Implemented yet.
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs[avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = avail_actions.sum(dim=-1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + torch.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[avail_actions == 0] = 0.0

        if int_t:
            return agent_outs.squeeze(1)
        if kwargs.get('imagine', False):
            return agent_outs, groups
        return agent_outs

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)

        return chosen_actions

    @staticmethod
    def _init_entity_splits(input_scheme: OrderedDict[str, tuple[int, int]]):
        # Mapping indices.
        split = []
        for feat_name, feat_shape in input_scheme.items():
            if feat_name.startswith("embedding."):
                # split.append(feat_shape[1])
                continue
            else:
                split.append(feat_shape[0] * feat_shape[1])
        return split

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

    def _build_inputs(self, batch, t: slice):
        """Build input for every agent.

        The input of entity agents is organized in a list of tensors.
        - First are entity states defined in env.env_info. Read fomr obs_components.
        - Then are embedding features defined in config.
        - Last are agent_id and last_action.
        """
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
            agent_id = torch.arange(    # [1, 2, 3]
                self.n_agents, dtype=torch.int, device=batch.device,
            ).repeat(       # b * t * n_agents * 1
                batch_size,
                time_size,
                1,
            ).unsqueeze(-1)
            inputs.append(agent_id)

        return inputs   # TODO: It's not elegant to return a list. Change to a NamedTuple or Dataclass.

    def _build_agents(self, input_shape):
        self.agent = AgentMaker.make(self.args.agent, input_shape, self.args)

    def _get_last_actions(self, batch, t: slice, batch_size, n_agents):
        """
        Return last actions of time slice t。

        Args:
            batch: PyMARL batch。
            t (slice): time slice。
            batch_size (int): batch size。
            n_agents (int): agent number。

        Returns:
            torch.Tensor: last actions of time slice t。
        """
        if t.start == 0:
            zeros = torch.zeros(batch_size, 1, n_agents, 1, device=self.device)
            sliced_actions = batch["actions"][:, slice(0, t.stop - 1)]
            return torch.cat([zeros, sliced_actions], dim=1).int()
        else:
            return batch["actions"][:, slice(t.start - 1, t.stop - 1)].int()
