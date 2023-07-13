import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import GATConv, GATv2Conv


@register_layer('gateconv')
class GATEConvGraphGymLayer(nn.Module):
    """Edge attr aware GAT convolution layer.
    """

    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.model = GATConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
            heads=cfg.gnn.att_heads,
            dropout=cfg.gnn.att_dropout,
            concat=cfg.gnn.att_concat_proj,
            edge_dim=cfg.dataset.edge_dim,
        )
        if cfg.gnn.att_concat_proj:
            proj_dim_out = layer_config.dim_out
            proj_dim_in = proj_dim_out * cfg.gnn.att_heads
            self.proj = nn.Linear(proj_dim_in, proj_dim_out)
        else:
            self.proj = nn.Identity()

    def forward(self, batch):
        x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        batch.x = self.proj(x)
        return batch


@register_layer('gatev2conv')
class GATEv2ConvGraphGymLayer(nn.Module):
    """Edge attr aware GAT convolution layer.
    """

    def __init__(self, layer_config: LayerConfig):
        super().__init__()
        self.model = GATv2Conv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
            heads=cfg.gnn.att_heads,
            dropout=cfg.gnn.att_dropout,
            concat=cfg.gnn.att_concat_proj,
            edge_dim=cfg.dataset.edge_dim,
        )
        if cfg.gnn.att_concat_proj:
            proj_dim_out = layer_config.dim_out
            proj_dim_in = proj_dim_out * cfg.gnn.att_heads
            self.proj = nn.Linear(proj_dim_in, proj_dim_out)
        else:
            self.proj = nn.Identity()

    def forward(self, batch):
        x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        batch.x = self.proj(x)
        return batch
