import numpy as np


class DecayThenFlatSchedule():
    """A schedule that starts with a linear decay and then stays flat.
    Args:
        start (float): initial value
        finish (float): final value
        time_length (int): time in number of steps over which to linearly decay
        decay (str): type of decay, either "linear" or "exp" (exponential)
    Returns:
        float: the current value of the schedule at time T
    """
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass
