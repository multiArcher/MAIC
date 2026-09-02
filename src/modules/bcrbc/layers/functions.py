from torch import Tensor


def softclamp(x: Tensor, value: float = 50.) -> Tensor:
    """Softclamping function for attention."""
    return (x / value).tanh() * value
