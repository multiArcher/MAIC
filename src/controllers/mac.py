from abc import ABC, abstractmethod
from types import SimpleNamespace

from torch.nn import Module

class MAC(Module, ABC):
    """
    Interface for Multi Agent Controllers(MAC).
    """
    @abstractmethod
    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super(MAC, self).__init__()

    @abstractmethod
    def select_actions(self, ep_batch, t_ep, t_env, bs, test_mode):
        pass

    @abstractmethod
    def forward(self, ep_batch, t, test_mode, **kwargs):
        pass

    @abstractmethod
    def init_hidden(self, batch_size):
        pass

    @abstractmethod
    def load_state(self, other_mac):
        pass

    @abstractmethod
    def save_models(self, path):
        pass

    @abstractmethod
    def load_models(self, path):
        pass

    @abstractmethod
    def _build_agents(self, input_shape):
        pass

    @abstractmethod
    def _build_inputs(self, batch, t):
        pass

    @abstractmethod
    def _get_input_shape(self, scheme):
        pass
