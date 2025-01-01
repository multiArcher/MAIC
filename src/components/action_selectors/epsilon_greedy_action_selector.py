import torch
from torch.distributions import Categorical

from .action_selector import ActionSelector
from .epsilon_schedules import DecayThenFlatSchedule

class EpsilonGreedyActionSelector(ActionSelector):
    """Implements epsilon-greedy action selection.

    Epsilon-greedy action selection chooses between exploration (taking a random action) and exploitation (taking the
    action with the highest Q-value). The probability of exploration is decayed over time, starting from a high value
    and annealing to a low value over a fixed period of time.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = getattr(self.args, "evaluation_epsilon", 0.0)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()  # Agent inputs are actually the output of Actor net.
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = torch.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions
