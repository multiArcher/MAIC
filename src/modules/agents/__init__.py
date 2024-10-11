from typing import Union

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .grc_agent import GRCAgent


REGISTRY = {
    "rnn": RNNAgent, 
    "rnn_ns": RNNNSAgent, 
    "rnn_feat": RNNFeatureAgent, 
    "grc": GRCAgent
}


def get_agent(agent_name: str, input_shape, args) \
        -> Union[
            RNNAgent, 
            RNNNSAgent, 
            RNNFeatureAgent, 
            GRCAgent
        ]:
    if agent_name in REGISTRY.keys():
        return REGISTRY[agent_name](input_shape, args)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}.")
