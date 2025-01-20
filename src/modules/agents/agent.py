from abc import ABC, abstractmethod

import torch


class Agent(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, input_shape: any, args: any):
        super(Agent, self).__init__()

    @abstractmethod
    def init_hidden(self):
        pass

    @abstractmethod
    def forward(self, inputs, hidden_state):
        pass

    @property
    def size(self):
        return str(sum(p.numel() for p in self.parameters()) / 1000) + 'K'
