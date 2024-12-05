from utils.maker import Maker
from .learner import Learner

class LearnerMaker(Maker):
    """Factory class for creating various types of learners."""
    # TODO, migrate Multiple Learners to Learner.
    @staticmethod
    def make_q_learner(*args, **kwargs) -> 'Learner':
        from .q_learner import QLearner
        return QLearner(*args, **kwargs)

    @staticmethod
    def make_coma_learner(*args, **kwargs) -> 'Learner':
        from .coma_learner import COMALearner
        return COMALearner(*args, **kwargs)

    @staticmethod
    def make_qtran_learner(*args, **kwargs) -> 'Learner':
        from .qtran_learner import QLearner as QTranLearner
        return QTranLearner(*args, **kwargs)

    @staticmethod
    def make_actor_critic_learner(*args, **kwargs) -> 'Learner':
        from .actor_critic_learner import ActorCriticLearner
        return ActorCriticLearner(*args, **kwargs)

    @staticmethod
    def make_pac_learner(*args, **kwargs) -> 'Learner':
        from .actor_critic_pac_learner import PACActorCriticLearner
        return PACActorCriticLearner(*args, **kwargs)

    @staticmethod
    def make_pac_dcg_learner(*args, **kwargs) -> 'Learner':
        from .actor_critic_pac_dcg_learner import PACDCGLearner
        return PACDCGLearner(*args, **kwargs)

    @staticmethod
    def make_maddpg_learner(*args, **kwargs) -> 'Learner':
        from .maddpg_learner import MADDPGLearner
        return MADDPGLearner(*args, **kwargs)

    @staticmethod
    def make_ppo_learner(*args, **kwargs) -> 'Learner':
        from .ppo_learner import PPOLearner
        return PPOLearner(*args, **kwargs)
