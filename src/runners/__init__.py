from typing import Union

from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner


REGISTRY = {
    "episode": EpisodeRunner,
    "parallel": ParallelRunner
}


def get_runner(runner_name: str, args, logger) \
        -> Union[
            EpisodeRunner,
            ParallelRunner
        ]:
    if runner_name in REGISTRY.keys():
        return REGISTRY[runner_name](args=args, logger=logger)
    else:
        raise ValueError(f"Invalid runner name: {runner_name}.")
