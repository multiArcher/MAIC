from .maker import Maker

from runners.runner import Runner


class RunnerMaker(Maker):
    """Factory class for creating runners."""

    @staticmethod
    def make_episode(args, logger) -> Runner:
        from runners.episode_runner import EpisodeRunner
        return EpisodeRunner(args, logger)

    @staticmethod
    def make_delayed_episode(args, logger) -> Runner:
        from runners.delayed_episode_runner import DelayedEpisodeRunner
        return DelayedEpisodeRunner(args, logger)

    @staticmethod
    def make_parallel(args, logger) -> Runner:
        from runners.parallel_runner import ParallelRunner
        return ParallelRunner(args, logger)
