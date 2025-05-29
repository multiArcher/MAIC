from abc import ABC, abstractmethod
from typing import Any

import torch


class Agent(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super(Agent, self).__init__()

    @abstractmethod
    def init_hidden(self) -> Any:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @property
    def size(self):
        return str(sum(p.numel() for p in self.parameters()) / 1000) + 'K'
