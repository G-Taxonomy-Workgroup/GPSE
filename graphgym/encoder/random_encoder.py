from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_node_encoder


class RandomNodeEncoder(nn.Module, ABC):
    """Random node encoders.

    This is an abstract class that is not to be used directly. Use the derived
    class NormalRENodeEncoder, UniformRENodeEncoder, and BernoulliRENodeEncoder
    instead.

    Args:
        dim_emb: Size of the final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x: bool = False):
        super().__init__()
        self.dim_emb = dim_emb

        if expand_x:
            raise NotImplementedError
        self.expand_x = expand_x

    def __repr__(self) -> str:
        dim_emb, expand_x = self.dim_emb, self.expand_x
        return f"{self.__class__.__name__}({dim_emb=!r}, {expand_x=!r})"

    @abstractmethod
    def generator(self, num_nodes, int, device: str) -> torch.Tensor:
        ...

    def forward(self, batch):
        batch.x = self.generator(batch.num_nodes, batch.edge_index.device)
        return batch


@register_node_encoder("NormalRE")
class NormalRENodeEncoder(RandomNodeEncoder):

    def generator(self, num_nodes: int, device: str) -> torch.Tensor:
        return torch.normal(0, 1, (num_nodes, self.dim_emb), device=device)


@register_node_encoder("UniformRE")
class UniformRENodeEncoder(RandomNodeEncoder):

    def generator(self, num_nodes: int, device: str) -> torch.Tensor:
        return torch.rand(num_nodes, self.dim_emb, device=device)


@register_node_encoder("BernoulliRE")
class BernoulliRENodeEncoder(RandomNodeEncoder):

    def generator(self, num_nodes: int, device: str) -> torch.Tensor:
        return torch.rand(num_nodes, self.dim_emb).float().to(device)


@register_node_encoder("NormalFixedRE")
class NormalFixedRENodeEncoder(RandomNodeEncoder):

    def generator(self, num_nodes, int, device: str) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch):
        batch.x = batch.pestat_NormalFixedRE
        return batch
