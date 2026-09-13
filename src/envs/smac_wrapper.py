import copy

from smac.env import StarCraft2Env

from .multiagentenv import MultiAgentEnv


class SMACWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        init_config = copy.deepcopy(kwargs)
        init_config.pop("common_reward", None)
        init_config.pop("reward_scalarisation", None)    
        init_config.pop("args", None)
        
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **init_config)
        self.episode_limit = self.env.episode_limit

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
        env_info.update(
            {
                "n_enemies": self.env.n_enemies,
                "n_allies": self.env.n_agents - 1,
                "n_actions_move": self.env.n_actions_move,
                "obs_move_feats_size": self.env.get_obs_move_feats_size(),
                "obs_enemy_feats_size": self.env.get_obs_enemy_feats_size(),
                "obs_ally_feats_size": self.env.get_obs_ally_feats_size(),
                "obs_own_feats_size": self.env.get_obs_own_feats_size(),
                "unit_type_bits": self.env.unit_type_bits,
                "obs_pathing_grid": self.env.obs_pathing_grid,
                "obs_terrain_height": self.env.obs_terrain_height,
                "obs_timestep_number": self.env.obs_timestep_number,
                "env_obs_last_action": self.env.obs_last_action,
            }
        )
        env_info["obs_components"] = (
            env_info["obs_move_feats_size"],
            env_info["obs_enemy_feats_size"],
            env_info["obs_ally_feats_size"],
            env_info["obs_own_feats_size"],
        )
        return env_info

    def get_stats(self):
        return self.env.get_stats()
