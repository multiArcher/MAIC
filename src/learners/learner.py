from abc import ABC, abstractmethod


class Learner(ABC):
    @abstractmethod
    def train(self, batch, t_env, episode_num):
        pass

    @abstractmethod
    def _update_targets_hard(self):
        pass

    @abstractmethod
    def _update_targets_soft(self, tau):
        pass

    @abstractmethod
    def cuda(self):
        # TODO: cuda can inherit from torch.nn.Module.
        pass

    def to(self, device):
        # TODO: Implement supplement of different devices.
        self.cuda()

    @abstractmethod
    def save_models(self, path):
        pass

    @abstractmethod
    def load_models(self, path):
        pass
