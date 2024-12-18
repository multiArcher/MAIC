from torch.nn import Module

from .maker import Maker

class CriticMaker(Maker):
    """Factory class for creating Critic Networks."""

    @staticmethod
    def make_coma_critic(*args, **kwargs) -> Module:
        from modules.critics.coma import COMACritic
        return COMACritic(*args, **kwargs)

    @staticmethod
    def make_cv_critic(*args, **kwargs) -> Module:
        from modules.critics.centralV import CentralVCritic
        return CentralVCritic(*args, **kwargs)

    @staticmethod
    def make_coma_critic_ns(*args, **kwargs) -> Module:
        from modules.critics.coma_ns import COMACriticNS
        return COMACriticNS(*args, **kwargs)

    @staticmethod
    def make_cv_critic_ns(*args, **kwargs) -> Module:
        from modules.critics.centralV_ns import CentralVCriticNS
        return CentralVCriticNS(*args, **kwargs)

    @staticmethod
    def make_maddpg_critic(*args, **kwargs) -> Module:
        from modules.critics.maddpg import MADDPGCritic
        return MADDPGCritic(*args, **kwargs)

    @staticmethod
    def make_maddpg_critic_ns(*args, **kwargs) -> Module:
        from modules.critics.maddpg_ns import MADDPGCriticNS
        return MADDPGCriticNS(*args, **kwargs)

    @staticmethod
    def make_ac_critic(*args, **kwargs) -> Module:
        from modules.critics.ac import ACCritic
        return ACCritic(*args, **kwargs)

    @staticmethod
    def make_ac_critic_ns(*args, **kwargs) -> Module:
        from modules.critics.ac_ns import ACCriticNS
        return ACCriticNS(*args, **kwargs)

    @staticmethod
    def make_pac_critic(*args, **kwargs) -> Module:
        from modules.critics.pac_ac import PACCritic
        return PACCritic(*args, **kwargs)

    @staticmethod
    def make_pac_critic_ns(*args, **kwargs) -> Module:
        from modules.critics.pac_ac_ns import PACCriticNS
        return PACCriticNS(*args, **kwargs)

    @staticmethod
    def make_pac_dcg_critic_ns(*args, **kwargs) -> Module:
        from modules.critics.pac_dcg_ns import DCGCriticNS
        return DCGCriticNS(*args, **kwargs)
