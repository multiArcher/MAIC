from abc import ABC, abstractmethod

import torch


class Agent(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self, input_shape, args):
        super(Agent, self).__init__()

    @abstractmethod
    def init_hidden(self):
        pass

    @abstractmethod
    def forward(self, inputs, hidden_state):
        pass
