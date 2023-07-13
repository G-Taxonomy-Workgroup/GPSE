from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False

    # Attention dropout ratios
    cfg.gnn.att_dropout = 0.0

    # Concatenate embeddings from multihead attentions, followed by a lin proj
    cfg.gnn.att_concat_proj = False
