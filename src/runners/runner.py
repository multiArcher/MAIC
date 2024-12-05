from abc import ABCMeta, abstractmethod


class Runner(metaclass=ABCMeta):
    @abstractmethod
    def setup(self, scheme, groups, preprocess, mac):
        pass

    @abstractmethod
    def get_env_info(self):
        pass

    @abstractmethod
    def save_replay(self):
        pass

    @abstractmethod
    def close_env(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run(self, test_mode):
        pass

    @abstractmethod
    def _log(self, returns, stats, prefix):
        pass