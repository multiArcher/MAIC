from .maker import Maker
from components.action_selectors.action_selector import ActionSelector


class ActionSelectorMaker(Maker):
    """Factory class for creating Action Selectors."""

    @staticmethod
    def make_multinomial(*args, **kwargs) -> ActionSelector:
        from components.action_selectors.multinomial_action_selector import MultinomialActionSelector
        return MultinomialActionSelector(*args, **kwargs)

    @staticmethod
    def make_epsilon_greedy(*args, **kwargs) -> ActionSelector:
        from components.action_selectors.epsilon_greedy_action_selector import EpsilonGreedyActionSelector
        return EpsilonGreedyActionSelector(*args, **kwargs)

    @staticmethod
    def make_soft_policies(*args, **kwargs) -> ActionSelector:
        from components.action_selectors.soft_policies_selector import SoftPoliciesSelector
        return SoftPoliciesSelector(*args, **kwargs)
