from .maker import Maker
from learners.learner import Learner

class LearnerMaker(Maker):
    """Factory class for creating various types of learners."""

    # TODO, migrate Multiple Learners to Learner.
    @staticmethod
    def make_q_learner(*args, **kwargs) -> 'Learner':
        from learners.q_learner import QLearner
        return QLearner(*args, **kwargs)

    @staticmethod
    def make_coma_learner(*args, **kwargs) -> 'Learner':
        from learners.coma_learner import COMALearner
        return COMALearner(*args, **kwargs)

    @staticmethod
    def make_qtran_learner(*args, **kwargs) -> 'Learner':
        from learners.qtran_learner import QLearner as QTranLearner
        return QTranLearner(*args, **kwargs)

    @staticmethod
    def make_actor_critic_learner(*args, **kwargs) -> 'Learner':
        from learners.actor_critic_learner import ActorCriticLearner
        return ActorCriticLearner(*args, **kwargs)

    @staticmethod
    def make_pac_learner(*args, **kwargs) -> 'Learner':
        from learners.actor_critic_pac_learner import PACActorCriticLearner
        return PACActorCriticLearner(*args, **kwargs)

    @staticmethod
    def make_pac_dcg_learner(*args, **kwargs) -> 'Learner':
        from learners.actor_critic_pac_dcg_learner import PACDCGLearner
        return PACDCGLearner(*args, **kwargs)

    @staticmethod
    def make_maddpg_learner(*args, **kwargs) -> 'Learner':
        from learners.maddpg_learner import MADDPGLearner
        return MADDPGLearner(*args, **kwargs)

    @staticmethod
    def make_ppo_learner(*args, **kwargs) -> 'Learner':
        from learners.ppo_learner import PPOLearner
        return PPOLearner(*args, **kwargs)

    @staticmethod
    def make_pymarl2_q_learner(*args, **kwargs) -> 'Learner':
        """Q-learner from PyMARL2."""
        from learners.pymarl2_q_learner import NQLearner
        return NQLearner(*args, **kwargs)

    @staticmethod
    def make_maic_learner(*args, **kwargs) -> 'Learner':
        from learners.maic_learner import MAICLearner
        return MAICLearner(*args, **kwargs)

    @staticmethod
    def make_icm_q_learner(*args, **kwargs) -> 'Learner':
        """Q-learner from PyMARL2."""
        from learners.icm_q_learner import ICMQLearner
        return ICMQLearner(*args, **kwargs)

    @staticmethod
    def make_refil_q_learner(*args, **kwargs) -> 'Learner':
        """Q-learner from PyMARL2."""
        from learners.refil_q_learning import RefilQLearner
        return RefilQLearner(*args, **kwargs)

    @staticmethod
    def make_new_q_learner(*args, **kwargs) -> 'Learner':
        """A more efficient Q-learner implementation."""
        from learners.new_q_learner import QLearner
        return QLearner(*args, **kwargs)
