import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head, pooling_dict


def _pad_and_stack(x1: torch.Tensor, x2: torch.Tensor, pad1: int, pad2: int):
    padded_x1 = nn.functional.pad(x1, (0, pad2))
    padded_x2 = nn.functional.pad(x2, (pad1, 0))
    return torch.vstack([padded_x1, padded_x2])


def _apply_index(batch, virtual_node: bool, pad_node: int, pad_graph: int):
    graph_pred, graph_true = batch.graph_feature, batch.y_graph
    node_pred, node_true = batch.node_feature, batch.y
    if virtual_node:
        # Remove virtual node
        idx = torch.concat([
            torch.where(batch.batch == i)[0][:-1]
            for i in range(batch.batch.max().item() + 1)
        ])
        node_pred, node_true = node_pred[idx], node_true[idx]

    # Stack node predictions on top of graph predictions and pad with zeros
    pred = _pad_and_stack(node_pred, graph_pred, pad_node, pad_graph)
    true = _pad_and_stack(node_true, graph_true, pad_node, pad_graph)

    return pred, true


@register_head('inductive_hybrid')
class GNNInductiveHybridHead(nn.Module):
    """
    GNN prediction head for inductive node and graph prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. Not used. Use share.num_node_targets
            and share.num_graph_targets instead.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.node_target_dim = cfg.share.num_node_targets
        self.graph_target_dim = cfg.share.num_graph_targets
        self.virtual_node = cfg.virtual_node
        num_layers = cfg.gnn.layers_post_mp

        self.node_post_mp = MLP(
            new_layer_config(dim_in, self.node_target_dim, num_layers,
                             has_act=False, has_bias=True, cfg=cfg))

        self.graph_pooling = pooling_dict[cfg.model.graph_pooling]
        self.graph_post_mp = MLP(
            new_layer_config(dim_in, self.graph_target_dim, num_layers,
                             has_act=False, has_bias=True, cfg=cfg))

    def forward(self, batch):
        batch.node_feature = self.node_post_mp(batch.x)
        graph_emb = self.graph_pooling(batch.x, batch.batch)
        batch.graph_feature = self.graph_post_mp(graph_emb)
        return _apply_index(batch, self.virtual_node, self.node_target_dim,
                            self.graph_target_dim)


@register_head('inductive_hybrid_multi')
class GNNInductiveHybridMultiHead(nn.Module):
    """
    GNN prediction head for inductive node and graph prediction tasks using
    individual MLP for each task.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. Not used. Use share.num_node_targets
            and share.num_graph_targets instead.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.node_target_dim = cfg.share.num_node_targets
        self.graph_target_dim = cfg.share.num_graph_targets
        self.virtual_node = cfg.virtual_node
        num_layers = cfg.gnn.layers_post_mp

        layer_config = new_layer_config(dim_in, 1, num_layers,
                                        has_act=False, has_bias=True, cfg=cfg)
        if cfg.gnn.multi_head_dim_inner is not None:
            layer_config.dim_inner = cfg.gnn.multi_head_dim_inner
        self.node_post_mps = nn.ModuleList([MLP(layer_config) for _ in
                                            range(self.node_target_dim)])

        self.graph_pooling = pooling_dict[cfg.model.graph_pooling]
        self.graph_post_mp = MLP(
            new_layer_config(dim_in, self.graph_target_dim, num_layers,
                             has_act=False, has_bias=True, cfg=cfg))

    def forward(self, batch):
        batch.node_feature = torch.hstack([m(batch.x)
                                           for m in self.node_post_mps])
        graph_emb = self.graph_pooling(batch.x, batch.batch)
        batch.graph_feature = self.graph_post_mp(graph_emb)
        return _apply_index(batch, self.virtual_node, self.node_target_dim,
                            self.graph_target_dim)
