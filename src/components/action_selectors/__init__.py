from utils.maker import Maker
from .action_selector import ActionSelector
from .multinomial_action_selector import MultinomialActionSelector
from .epsilon_greedy_action_selector import EpsilonGreedyActionSelector
from .soft_policies_selector import SoftPoliciesSelector


class ActionSelectorMaker(Maker):
    """Factory class for creating Action Selectors."""
    @staticmethod
    def make_multinomial(*args, **kwargs) -> ActionSelector:
        return MultinomialActionSelector(*args, **kwargs)

    @staticmethod
    def make_epsilon_greedy(*args, **kwargs) -> ActionSelector:
        return EpsilonGreedyActionSelector(*args, **kwargs)

    @staticmethod
    def make_soft_policies(*args, **kwargs) -> ActionSelector:
        return SoftPoliciesSelector(*args, **kwargs)
