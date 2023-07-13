import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


def _apply_index(batch):
    pred, true = batch.x, batch.y
    if cfg.virtual_node:
        # Remove virtual node
        idx = torch.concat([
            torch.where(batch.batch == i)[0][:-1]
            for i in range(batch.batch.max().item() + 1)
        ])
        pred, true = pred[idx], true[idx]
    return pred, true


@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, true = _apply_index(batch)
        return pred, true


@register_head('inductive_node_multi')
class GNNInductiveNodeMultiHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks (one MLP per task).

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        layer_config = new_layer_config(dim_in, 1, cfg.gnn.layers_post_mp,
                                        has_act=False, has_bias=True, cfg=cfg)
        if cfg.gnn.multi_head_dim_inner is not None:
            layer_config.dim_inner = cfg.gnn.multi_head_dim_inner

        self.layer_post_mp = nn.ModuleList([MLP(layer_config)
                                            for _ in range(dim_out)])

    def forward(self, batch):
        batch.x = torch.hstack([m(batch.x) for m in self.layer_post_mp])
        pred, true = _apply_index(batch)
        return pred, true
