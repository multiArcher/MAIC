from types import SimpleNamespace

import torch

from .mac import MAC
from utils.maker import AgentMaker
from utils.maker import ActionSelectorMaker
from utils.custom_logging import PyMARLLogger


class EntityMAC(MAC, torch.nn.Module):
    """Controller for entity wise env.

    Differences from MAC:
    - input_slices: list of slices to extract input data from obs into entity inputs.
    - input_shape: tuple of input shape of agents in a tuple including
    (own_feats_dim, enemy_feats_dim, ally_feats_dim, Optional last_action_dim, Optional agent_id_dim)

    TODO: Testing inheriting from torch.nn.Module to enable features
     of breakpoints functions to be added in the future.
    """
    def __init__(self, scheme: dict, groups, args: SimpleNamespace):
        if getattr(args, "entity_scheme", False) is False:
            # Check if the env is in entity scheme
            class_name = self.__class__.__name__
            PyMARLLogger("main").get_child_logger(f"{class_name}").critical(f"{class_name} only works in entity scheme.")
            raise RuntimeError(f"{class_name} only works in entity scheme.")

        super(MAC, self).__init__()

        self.args = args
        self.n_agents: int = args.n_agents
        self.obs_shape: int = args.env_info["obs_shape"]
        self.obs_components: dict = args.env_info["obs_components"]   # Obs components in a dict. Calculated in env_wrapper.
        self.n_enemies, self.n_enemy_feats_dim = self.obs_components["n_enemy_feats"]   # int, int
        self.n_ally, self.n_ally_feats_dim = self.obs_components["n_ally_feats"]    # int, int
        self.move_feats_size: int = self.obs_components["move_feats_size"]
        self.own_feats_size: int = self.obs_components["own_feats_size"]

        self.input_slices: list[slice] = []  # Slices to extract input data from obs into entity inputs.
        self._inti_entity_mapping_slices()

        self.input_shape = self._get_input_shape(scheme)
        self.agent_output_type = args.agent_output_type
        self.action_selector = ActionSelectorMaker.make(args.action_selector, args)

        self._build_agents(self.input_shape)
        self.hidden_states = None

    def _get_input_shape(self, scheme) -> tuple[int, int, int, int, int]:
        """Return input shape of agents in a tuple including (own_feats_dim, enemy_feats_dim, ally_feats_dim)"""
        obs_components: dict = self.args.env_info["obs_components"]

        move_feats_dim: int = obs_components["move_feats_size"]
        enemy_feats_dim: int = obs_components["n_enemy_feats"][1]
        ally_feats_dim: int = obs_components["n_ally_feats"][1]
        own_feats_dim: int = obs_components["own_feats_size"]
        own_feats_dim += move_feats_dim

        last_action_dim = 1 if self.args.obs_last_action else 0
        agent_id_dim = 1 if self.args.obs_agent_id else 0

        input_shape = (own_feats_dim, enemy_feats_dim, ally_feats_dim, last_action_dim, agent_id_dim)

        return input_shape

    def _build_inputs(self, batch, t):
        """The input of entity agents have 5 parts:

        own_feats: batch * time * n_agents * own_feats_dim
        ally_feats: batch * time * n_agents * (n_agents - 1) * ally_feats_dim
        enemy_feats: batch * time * n_agents * n_enemies * enemy_feats_dim
        Optional last_action: batch * time * n_agents * n_actions
        Optional agent_id: batch * time * n_agents * n_agents

        They will return in a tuple in order. If the option is not set, the corresponding part will be None.
        TODO: The return is in a list. This is not always compatible with agents.
              Maybe a named_tuple or else.
        """
        batch_size, time_steps, n_agents, obs_size = batch["obs"].shape
        obs_data = batch["obs"][:, t]
        time_size = obs_data.shape[1]

        # Split obs by the mapping indices.
        move_feats, enemy_feats, ally_feats, own_feats = [obs_data[...,input_slice] for input_slice in self.input_slices]
        # split enemies and allies.
        enemy_feats = enemy_feats.reshape(batch_size, time_size, self.n_agents, self.n_enemies, self.n_enemy_feats_dim)
        ally_feats = ally_feats.reshape(batch_size, time_size, self.n_agents, self.n_agents - 1, self.n_ally_feats_dim)

        own_feats_catted = torch.cat([own_feats, move_feats], dim=-1)   # own_feats is own_feats + move_feats.

        last_actions = None
        agent_id = None

        if self.args.obs_last_action:
            # Add a one dim last_action. This is not one-hot. The Agent will handle it.
            last_actions_data = torch.roll(batch["actions"], shifts=1, dims=1).int()
            last_actions_data[:, 0] = 0
            last_actions = last_actions_data[:, t].squeeze(-1)

        if self.args.obs_agent_id:
            # Add a one dim agent_id. This is not one-hot. The Agent will handle it.
            agent_id = torch.arange(    # [1, 2, 3]
                self.n_agents,
                dtype=torch.int,
                device=batch.device,
            ).repeat(       # b * t * n_agents * 1
                batch_size,
                time_size,
                1,
            )
        # TODO: It's not elegant to return a list. Change to a named_tuple or else.
        return own_feats_catted, ally_feats, enemy_feats, last_actions, agent_id

    def _build_agents(self, input_shape):
        self.agent = AgentMaker.make(self.args.agent, input_shape, self.args)

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def forward(self, ep_batch, t, test_mode=False, *args, **kwargs):
        # TODO: The situation that t is a slice never happens. Consider removing it.
        if int_t:= isinstance(t, int):
            t = slice(t, t + 1)
        else:
            t = slice(0, ep_batch["avail_actions"].shape[1])

        agent_inputs = self._build_inputs(ep_batch, t)  # own_feats, ally_feats, enemy_feats, last_actions, agent_id
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)   # Agent forward

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

        return agent_outs

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)

        return chosen_actions

    def _inti_entity_mapping_slices(self):
        # Mapping indices.  TODO: slice might be slow. Try torch.split instead.
        bit_count = 0
        self.input_slices.append(slice(bit_count, bit_count:= bit_count + self.move_feats_size))
        self.input_slices.append(slice(bit_count, bit_count:= bit_count + self.n_enemies * self.n_enemy_feats_dim))
        self.input_slices.append(slice(bit_count, bit_count:= bit_count + self.n_ally * self.n_ally_feats_dim))
        self.input_slices.append(slice(bit_count, bit_count:= bit_count + self.own_feats_size))

        # For debugging  # 92 in p 5v5 map
        assert bit_count == self.obs_shape, "The mapping is not correct."
