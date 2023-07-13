import torch.nn as nn
import torch_geometric.graphgym.register as register

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv


@register_layer('gatv2conv')
class GATv2ConvGraphGymLayer(nn.Module):
    """GATv2 convolution layer
    for forward(), we ignore `edge_attr` since GraphGym does so for GATConv as well.
    Both can be extended to account for `edge_attr` if necessary.
    """

    def __init__(self, layer_config: LayerConfig):
        super(GATv2ConvGraphGymLayer, self).__init__()
        self.model = GATv2Conv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
            heads=cfg.gnn.att_heads,
            dropout=cfg.gnn.att_dropout,
            concat=cfg.gnn.att_concat_proj,
        )
        if cfg.gnn.att_concat_proj:
            proj_dim_out = layer_config.dim_out
            proj_dim_in = proj_dim_out * cfg.gnn.att_heads
            self.proj = nn.Linear(proj_dim_in, proj_dim_out)
        else:
            self.proj = nn.Identity()

    def forward(self, batch):
        x = self.model(batch.x, batch.edge_index)
        batch.x = self.proj(x)
        return batch


class GATv2ConvLayer(nn.Module):
    """GATv2 convolution layer
    for forward(), we ignore `edge_attr` since GraphGym does so for GATConv as well.
    Both can be extended to account for `edge_attr` if necessary.
    """

    def __init__(self, in_dim, out_dim, dropout, residual, act='relu', **kwargs):
        super(GATv2ConvLayer, self).__init__()
        self.model = GATv2Conv(in_dim,
                               out_dim,
                               dropout=cfg.gnn.att_dropout,
                               act=register.act_dict[act](),
                               residual=residual,
                               heads=cfg.gnn.att_heads,
                               concat=cfg.gnn.att_concat_proj,
                               **kwargs
                               )
        if cfg.gnn.att_concat_proj:
            proj_dim_out = out_dim
            proj_dim_in = proj_dim_out * cfg.gnn.att_heads
            self.proj = nn.Linear(proj_dim_in, proj_dim_out)
        else:
            self.proj = nn.Identity()

    def forward(self, batch):
        x = self.model(batch.x, batch.edge_index)
        batch.x = self.proj(x)
        return batch
