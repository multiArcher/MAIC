import torch
from torch import nn


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def get_parameters_num(param_list):
    numel = sum(p.numel() for p in param_list)
    
    if numel < 1e6:
        return str(round(numel / 1e3, 2)) + 'K'
    elif numel < 1e9:
        return str(round(numel / 1e6, 2)) + 'M'
    else:
        return str(round(numel / 1e9, 2)) + 'B'
    
    
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_,
             lambda x: nn.init.constant_(x, 0), gain=gain)
