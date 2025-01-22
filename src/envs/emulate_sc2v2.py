import yaml
import torch

from typing import Union
from pathlib import Path
from collections import OrderedDict

import numpy
from tqdm import tqdm

from envs.smacv2_wrapper import load_scenario, SMACv2Wrapper, SMACv2_CONFIG_DIR
from .multiagentenv import MultiAgentEnv

# TODO: This should not be here!
SMACv2_CONFIG_DIR = Path(__file__).parent.parent / "config" / "envs" / "smacv2_configs"
HISTORY_DATA_PATH = Path(r"C:\Users\loren\WorkSpace\epymarl_based\history_data")


class EmulateSMACv2(SMACv2Wrapper):
    def __init__(self, map_name, seed, **kwargs):
        super().__init__(map_name, seed, **kwargs)
        self.available_actions_data_path = HISTORY_DATA_PATH / f"available_actions_data.npy"
        self.obs_data_path = HISTORY_DATA_PATH / f"obs_data.npy"

        self.available_actions_data = numpy.load(self.available_actions_data_path, allow_pickle=True)
        self.obs_data = numpy.load(self.obs_data_path, allow_pickle=True)

        self.game_index = -1
        self.max_game_index = len(self.available_actions_data) - 1
        self.game_step = 0
        self.current_game_trace_max_step = 0

        self.available_actions_trace = None
        self.obs_trace = None

        self.trace_actions_record = []
        self.actions_record = []
        self.progress_bar = tqdm(total=self.max_game_index+1)

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        # actions = torch.ones(5)
        # rewards, terminated, info = self.env.step(actions)
        self.trace_actions_record.append(actions.detach().cpu().numpy())
        obss = self.get_obs()
        truncated = False
        rewards = 0.0

        terminated = self.game_step >= self.current_game_trace_max_step - 2
        info = {
            "battle_won": False,
            "dead_allies": 0,
            "dead_enemies": 0,
        }
        self.game_step += 1

        if terminated:
            self.actions_record.append(numpy.array(self.trace_actions_record))
            self.trace_actions_record = []

        if self.game_index == self.max_game_index and terminated:
            actions_record_numpy = numpy.array(self.actions_record, dtype=object)
            numpy.save(HISTORY_DATA_PATH / f"entity_attend_agent_qmix.npy", actions_record_numpy)
            raise StopIteration("End of data. Inferencing complete.")

        return obss, rewards, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        obs_array = self.obs_trace[self.game_step][0]
        # print(self.game_step)
        obs_list = [obs_array[i] for i in range(len(obs_array))]
        return obs_list

    def get_avail_actions(self):
        avail_actions_array = self.available_actions_trace[self.game_step][0]
        # print(self.game_step)
        avail_actions_list = [avail_actions_array[i] for i in range(len(avail_actions_array))]
        return avail_actions_list

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        self.game_index += 1
        self.game_step = 0

        if seed is not None:
            self.env.seed(seed)
        if self.game_index == 0:
            obss, _ = self.env.reset()


        self.available_actions_trace = self.available_actions_data[self.game_index]
        self.obs_trace = self.obs_data[self.game_index]
        self.current_game_trace_max_step = len(self.obs_trace)

        obss = self.get_obs()
        self.progress_bar.update(1)

        return obss, {}

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError
