import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as functional

from utils.custom_logging import PyMARLLogger


class EntityPoolingLayer(nn.Module):
    """
    pooling Layer for entity scheme multi-agent systems.
    Args:
        args: args initialized in pymarl.
        embed_dim: size of embedding feature.
    """
    def __init__(
            self,
            args: SimpleNamespace,
            embed_dim: int,
    ):
        super(EntityPoolingLayer, self).__init__()
        self.args = args

        self.embed_dim = embed_dim

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}, heads={self.num_heads}"

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entities (torch.Tensor): Entity representations.
                [batch size, time size, n_agents, n_entities, embedding dimension]

        Return:
            pooled_entities (torch.Tensor): attend entities state., [batch size, n_agents, embedding dimension]
        """
        pooled_entities = torch.mean(entities, dim=2)

        return pooled_entities
