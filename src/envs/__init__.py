import os
import sys

from .multiagentenv import MultiAgentEnv
from utils.maker import Maker

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

class EnvMaker(Maker):
    """Factory class for creating environments."""
    @staticmethod
    def _check_and_prepare_smac_kwargs(kwargs):
        """Check and prepare kwargs for SMAC environments."""
        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        assert kwargs[
            "common_reward"
        ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
        del kwargs["common_reward"]
        del kwargs["reward_scalarisation"]
        assert "map_name" in kwargs, "Please specify the map_name in the env_args"
        return kwargs

    @staticmethod
    def make_gymma(*args, **kwargs) -> MultiAgentEnv:
        from .gymma import GymmaWrapper

        assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
        return GymmaWrapper(*args, **kwargs)

    @staticmethod
    def make_smaclite(*args, **kwargs) -> MultiAgentEnv:
        from .smaclite_wrapper import SMACliteWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACliteWrapper(*args, **kwargs)

    @staticmethod
    def make_sc2(*args, **kwargs) -> MultiAgentEnv:
        # registering both smac and smacv2 causes a pysc2 error
        from .smac_wrapper import SMACWrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(*args, **kwargs)

    @staticmethod
    def make_sc2v2(*args, **kwargs) -> MultiAgentEnv:
        # registering both smac and smacv2 causes a pysc2 error
        from .smacv2_wrapper import SMACv2Wrapper

        kwargs = EnvMaker._check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(*args, **kwargs)
