from utils.custom_logging import PyMARLLogger
from envs.multiagentenv import MultiAgentEnv
from .maker import Maker


class EnvMaker(Maker):
    """Factory class for creating environments."""

    @staticmethod
    def _check_and_prepare_smac_kwargs(kwargs):
        """Check and prepare kwargs for SMAC environments."""
        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        assert kwargs[
            "common_reward"
        ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
        # del kwargs["common_reward"]
        # del kwargs["reward_scalarisation"]
        assert "map_name" in kwargs, "Please specify the map_name in the env_args"
        return kwargs

    @staticmethod
    def make_gymma(*args, **kwargs) -> MultiAgentEnv:
        from envs.gymma import GymmaWrapper

        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        return GymmaWrapper(*args, **kwargs)

    @staticmethod
    def make_smaclite(*args, **kwargs) -> MultiAgentEnv:
        from envs.smaclite_wrapper import SMACliteWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACliteWrapper(*args, **kwargs)

    @staticmethod
    def make_sc2(*args, **kwargs) -> MultiAgentEnv:
        from envs.smac_wrapper import SMACWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(*args, **kwargs)

    @staticmethod
    def make_sc2v2(*args, **kwargs) -> MultiAgentEnv:
        if len(args) != 0:
            PyMARLLogger(
                "main", PyMARLLogger.DEBUG
            ).warning(
                "SMACv2 should NOT be initialized by placeholder args. Unwanted behavior may occur."
            )

        from envs.smacv2_wrapper import SMACv2Wrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)

        return SMACv2Wrapper(*args, **kwargs)
