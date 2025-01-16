import yaml
from pathlib import Path
from collections import OrderedDict

from typing import Union

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from .multiagentenv import MultiAgentEnv


# TODO: This should not be here!
SMACv2_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"


def get_scenario_names():
    return [p.name for p in SMACv2_CONFIG_DIR.iterdir()]


def load_scenario(map_name, **kwargs):
    scenario_path = SMACv2_CONFIG_DIR / f"{map_name}.yaml"
    with open(scenario_path, "r") as f:
        scenario_args = yaml.load(f, Loader=yaml.FullLoader)
    scenario_args.update(kwargs)    # TODO: Bug Seed, episode_limit and other kwargs are not passed to the SC2Env.
    # Work around: pass them to the SC2Env directly.
    scenario_args["env_args"]["seed"] = kwargs.get("seed", None)
    scenario_args["env_args"]["window_size_x"] = kwargs.get("window_size_x", None)
    scenario_args["env_args"]["window_size_y"] = kwargs.get("window_size_y", None)
    scenario_args["env_args"]["fully_observable"] = kwargs.get("fully_observable", False)
    scenario_args["env_args"]["state_timestep_number"] = kwargs.get("state_timestep_number", False)

    # Default to None if not provided.
    if replay_dir := kwargs.get("replay_dir", None):
        scenario_args["env_args"]["replay_dir"] = replay_dir
    if replay_prefix := kwargs.get("replay_prefix", None):
        scenario_args["env_args"]["replay_prefix"] = replay_prefix

    return StarCraftCapabilityEnvWrapper(**scenario_args["env_args"])


class SMACv2Wrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.env = load_scenario(map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit

    def get_obs_components(self) -> OrderedDict[str: Union[int, tuple[int, int]]]:
        """
        This function needs to keep synchronization with StarCraft2Env.get_obs_size()
        The order is defined in StarCraft2Env.get_obs_agent()
        """
        components = OrderedDict()
        components["move_feats_size"] = (1, self.env.get_obs_move_feats_size())
        components["n_enemy_feats"] = self.env.get_obs_enemy_feats_size()
        components["n_ally_feats"] = self.env.get_obs_ally_feats_size()
        components["own_feats_size"] = (1, self.env.get_obs_own_feats_size())
        return components

    def get_state_components(self) -> OrderedDict[str: Union[int, tuple[int, int]]]:
        """
        This function needs to keep synchronization with StarCraft2Env.get_state_size()
        The order is defined in StarCraft2Env.get_state()
        """
        if self.env.obs_instead_of_state:
            return self.get_obs_components()

        components = OrderedDict()
        components["n_ally_feats"] = (self.env.n_agents, self.env.get_ally_num_attributes())
        components["n_enemy_feats"] = (self.env.n_enemies, self.env.get_enemy_num_attributes())

        if self.state_last_action:
            components["last_action"] = (self.env.n_agents, self.env.n_actions)
        if self.env.state_timestep_number:
            components["timestep_number"] = 1

        return components

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info["n_enemies"] = self.n_enemies

        #TODO : This is not a part of env info. This should be implemented by exposing env function,
        # but the init of env is in runner, blocking reading map params.
        # This is a workaround of passing params to run() func.
        # Move this arguments maybe to a singleton configuration class.
        # The old Config class in epic7_automator may be useful.
        env_info["state_feature_names"] = self.env.get_state_feature_names()
        env_info["obs_feature_names"] = self.env.get_obs_feature_names()
        env_info["obs_components"] = self.get_obs_components()
        env_info["state_components"] = self.get_state_components()
        env_info["state_last_action"] = self.env.state_last_action

        return env_info

    def get_stats(self):
        return self.env.get_stats()

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError


if __name__ == "__main__":
    for scenario in get_scenario_names():
        env = load_scenario(scenario)
        env_info = env.get_env_info()
        # print name of config, number of agents, state shape, observation shape, action shape
        print(
            scenario,
            env_info["n_agents"],
            env_info["state_shape"],
            env_info["obs_shape"],
            env_info["n_actions"],
        )
        print()
