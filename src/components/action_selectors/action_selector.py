from abc import ABC, abstractmethod


class ActionSelector(ABC):
    """Abstract class for action selectors."""

    @abstractmethod
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        pass
