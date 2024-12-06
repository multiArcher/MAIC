from .runner import Runner
from utils.maker import Maker


class RunnerMaker(Maker):
    """
    Factory class for creating runners.
    """
    @staticmethod
    def make_episode(args, logger) -> Runner:
        from .episode_runner import EpisodeRunner
        return EpisodeRunner(args, logger)

    @staticmethod
    def make_parallel(args, logger) -> Runner:
        from .parallel_runner import ParallelRunner
        return ParallelRunner(args, logger)
