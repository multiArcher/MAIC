from torch.nn import Module
from utils.maker import Maker


class AgentMaker(Maker):
    """Factory class for creating Agents."""
    @staticmethod
    def make_rnn(*args, **kwargs) -> Module:
        from .rnn_agent import RNNAgent
        return RNNAgent(*args, **kwargs)

    @staticmethod
    def make_rnn_ns(*args, **kwargs) -> Module:
        from .rnn_ns_agent import RNNNSAgent
        return RNNNSAgent(*args, **kwargs)    # TODO migrate RNNNSAgent to Agent

    @staticmethod
    def make_rnn_feat(*args, **kwargs) -> Module:
        from .rnn_feature_agent import RNNFeatureAgent
        return RNNFeatureAgent(*args, **kwargs)   # TODO migrate RNNFeatureAgent to Agent
