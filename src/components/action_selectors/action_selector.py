from abc import ABC, abstractmethod


class ActionSelector(ABC):
    @abstractmethod
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        pass