from abc import ABCMeta, abstractmethod


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def train(self, batch, t_env, episode_num):
        pass

    @abstractmethod
    def _update_targets_hard(self):
        pass

    @abstractmethod
    def _update_targets_soft(self, tau):
        pass

    # def to(self, device):
    #     self.mac.to(device)
    #     self.target_mac.to(device)
    #     if self.mixer is not None:
    #         self.mixer.to(device)
    #         self.target_mixer.to(device)

    @abstractmethod
    def cuda(self):
        pass

    def to(self, device):
        # TODO Implement suppliment of different devices.
        self.cuda()

    @abstractmethod
    def save_models(self, path):
        pass

    @abstractmethod
    def load_models(self, path):
        pass
