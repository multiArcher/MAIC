import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Linear, RMSNorm, Identity


class SwiGLUFeedForward(Module):
    """ SwiGLU Feedforward Module.

    Shazeer, Noam. "Glu variants improve transformer." arXiv preprint arXiv:2002.05202 (2020).
    https://arxiv.org/pdf/2002.05202

    FFN_SwiGLU(x, W, V, W2) = (Swish1(xW) ⊙ (xV)) W2
    where Swish1(z) = z * sigmoid(z)
    """
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        pre_rmsnorm: bool = True,
        device = None
    ):
        super().__init__()
        self.norm = RMSNorm(dim, device=device) if pre_rmsnorm else Identity()  # TODO: Move to Transformer Block.

        # PaLM / LLaMA logic: minimize parameter count increase while widening the layer
        # Standard FFN has 2 matrices (dim -> 4dim -> dim).
        # SwiGLU has 3 matrices (Gate, Val, Out). To keep params roughly same, reduce width by 2/3.
        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.proj_in = Linear(dim, dim_inner * 2, device=device)
        self.proj_out = Linear(dim_inner, dim, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        x, gates = self.proj_in(x).chunk(2, dim = -1)
        x = x * F.silu(gates)

        return self.proj_out(x)
