from typing import Optional

from torch_geometric import nn as pygnn
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode


class RandomGNNHash:
    def __init__(
        self,
        hash_gnn_cfg: Optional[CfgNode] = None,
        device: Optional[str] = None,
    ):
        self.hash_gnn_cfg = hash_gnn_cfg or cfg.hash_feat
        self.device = device or cfg.accelerator

    @property
    def model(self):
        return getattr(self, "_model", None)

    def init_model(self, num_feat):
        gnn_configs = {
            "in_channels": num_feat,
            "out_channels": num_feat,
            "hidden_channels": self.hash_gnn_cfg.gnn_dim_inner,
            "num_layers": self.hash_gnn_cfg.gnn_num_layers,
            "act": self.hash_gnn_cfg.gnn_act,
        }  # yapf: disable
        mdl_cls = getattr(pygnn, self.hash_gnn_cfg.gnn_name)
        self._model = mdl_cls(**gnn_configs).to(self.device)

    def __call__(self, data):
        if not self.model:
            feat_name = self.hash_gnn_cfg.name_in
            try:
                num_feat = getattr(data, feat_name).shape[1]
            except AttributeError:
                raise AttributeError("Trying to access non-existing feature "
                                     f"{feat_name} from {data} during hashing "
                                     "GNN model initialization.")
            self.init_model(num_feat)

        x = getattr(data, self.hash_gnn_cfg.name_in).to(self.device)
        g = getattr(data, self.hash_gnn_cfg.graph_format).to(self.device)
        setattr(data, self.hash_gnn_cfg.name_out, self.model(x, g).detach().cpu())

        return data


class GraphNormalizer:
    def __init__(self):
        self._normalizer = None

    @property
    def normalizer(self):
        return self._normalizer

    def init_model(self, num_feat):
        gnn_configs = {
            "in_channels": num_feat,
            "eps": cfg.graph_norm.eps
        }  # yapf: disable
        mdl_cls = getattr(pygnn.norm, cfg.graph_norm.name)
        self._normalizer = mdl_cls(**gnn_configs).to(cfg.accelerator)

    def __call__(self, data):
        if not self.normalizer:
            feat_name = cfg.graph_norm.name_in
            try:
                num_feat = getattr(data, feat_name).shape[1]
            except AttributeError:
                raise AttributeError("Trying to access non-existing feature "
                                     f"{feat_name} from {data} during "
                                     "normalizer initialization.")
            self.init_model(num_feat)

        x = getattr(data, cfg.graph_norm.name_in).to(cfg.accelerator)
        x[:, x.var(0) < cfg.graph_norm.clip_var] = 0
        setattr(data, cfg.graph_norm.name_out, self.normalizer(x).detach().cpu())

        return data
