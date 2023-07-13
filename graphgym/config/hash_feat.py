from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config("cfg_hash_feat")
def set_cfg_hash_feat(cfg):
    """Feature hashing GNN configuration."""
    cfg.hash_feat = CN()

    # Enable feature hashing
    cfg.hash_feat.enable = False

    # Name of the node attributes to be hashed and the name to set
    cfg.hash_feat.name_in = "y"
    cfg.hash_feat.name_out = "y"
    cfg.hash_feat.graph_format = "edge_index"

    # Hashing GNN model settings
    cfg.hash_feat.gnn_name = "GIN"
    cfg.hash_feat.gnn_num_layers = 5
    cfg.hash_feat.gnn_act = "relu"
    cfg.hash_feat.gnn_dim_inner = 128


@register_config("cfg_graph_norm")
def set_cfg_graph_norm(cfg):
    """Graph normalization configuration."""
    cfg.graph_norm = CN()

    # Enable feature hashing
    cfg.graph_norm.enable = False

    # Name of the node attributes to be hashed and the name to set
    cfg.graph_norm.name_in = "y"
    cfg.graph_norm.name_out = "y"

    # Target node encoders normalization settings
    cfg.graph_norm.name = "GraphNorm"
    cfg.graph_norm.eps = 1e-05
    # set features with variance below this threshold to be zeros
    cfg.graph_norm.clip_var = 1e-05


@register_config("cfg_virtual_node")
def set_cfg_virtual_node(cfg):
    """Graph normalization configuration."""
    cfg.virtual_node = False
