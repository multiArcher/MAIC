import torch
from torch.nn.functional import normalize
from torch.nn import Module, Parameter


class MultiHeadRMSNorm(Module):
    """Multi-Head RMS Normalization Layer."""
    def __init__(
        self,
        dim_head,
        heads = 8,
        device=None
    ):
        super().__init__()
        self.scale = torch.tensor(dim_head ** 0.5, device=device)
        self.gamma = Parameter(torch.zeros(heads, 1, dim_head, device=device))

    def forward(
        self,
        x: torch.Tensor,    # (..., heads, _, dim_head)
        dim: int = -1
    ) -> torch.Tensor:
        """Normalize over the last dimension by default."""
        normed = normalize(x, dim = dim, p = 2)
        scale = (self.gamma + 1.) * self.scale  # (heads, 1, dim_head)

        return normed * scale.expand_as(normed)  # (..., heads, _, dim_head)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(heads={self.gamma.shape[0]}, dim_head={self.gamma.shape[2]})"
