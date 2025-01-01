from torch.nn import Module

from .maker import Maker


class MixerMaker(Maker):
    """Factory class for creating Mixer networks."""

    @staticmethod
    def check_common_reward(func):
        def wrapper(*args, **kwargs):
            if "args" in kwargs:
                common_reward = kwargs["args"].common_reward
            else:
                common_reward = args[0].common_reward

            if common_reward is not True:
                raise ValueError("VDN and QMixer only support common reward setting")
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    @check_common_reward
    def make_qmix(*args, **kwargs) -> Module:
        from modules.mixers.qmix import QMixer
        return QMixer(*args, **kwargs)

    @staticmethod
    def make_qtran(*args, **kwargs) -> Module:
        from modules.mixers.qtran import QTranBase
        return QTranBase(*args, **kwargs)    # TODO migrate RNNNSAgent to Agent

    @staticmethod
    @check_common_reward
    def make_vdn(*args, **kwargs) -> Module:
        from modules.mixers.vdn import VDNMixer
        return VDNMixer(*args, **kwargs)   # TODO migrate RNNFeatureAgent to Agent

    @staticmethod
    def make_flex_qmix(*args, **kwargs) -> Module:
        from modules.mixers.flex_qmix import FlexQMixer
        return FlexQMixer(*args, **kwargs)

    @staticmethod
    def make_pymarl2_qmix_mixer(*args, **kwargs) -> Module:
        """QMIX implementation in PyMARL2."""
        from modules.mixers.pymarl2_qmix_mixer import Mixer
        return Mixer(*args, **kwargs)

    @staticmethod
    def make_icm_qmix(*args, **kwargs) -> Module:
        """QMIX implementation in PyMARL2."""
        from modules.mixers.icm_qmix import ICMQMixer
        return ICMQMixer(*args, **kwargs)