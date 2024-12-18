from collections import OrderedDict
from abc import ABC, abstractmethod

import torch


class Transform(ABC):
    @abstractmethod
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform input to output"""
        pass

    @abstractmethod
    def infer_output_info(self, vshape_in: tuple[int], dtype_in: torch.dtype):
        """Infer output shape and type from input shape and type"""
        pass


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32
