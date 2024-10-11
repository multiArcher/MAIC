from typing import Union


from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .grc_learner import GRCLearner
from .reward_shaping_learner import RewardShapingLearner


REGISTRY = {
    "q_learner": QLearner,
    "coma_learner": COMALearner,
    "qtran_learner": QTranLearner,
    "actor_critic_learner": ActorCriticLearner,
    "maddpg_learner": MADDPGLearner,
    "ppo_learner": PPOLearner,
    "pac_learner": PACActorCriticLearner,
    "pac_dcg_learner": PACDCGLearner,
    "grc_learner": GRCLearner,
    "rsq_learner": RewardShapingLearner
}


def get_learner(learner_name: str, mac, scheme, logger, args) \
        -> Union[
            QLearner,
            COMALearner,
            QTranLearner,
            ActorCriticLearner,
            MADDPGLearner,
            PPOLearner,
            PACActorCriticLearner,
            PACDCGLearner,
            GRCLearner,
            RewardShapingLearner
        ]:
    if learner_name in REGISTRY.keys():
        return REGISTRY[learner_name](mac, scheme, logger, args)
    else:
        raise ValueError(f"Invalid learner name: {learner_name}.")
