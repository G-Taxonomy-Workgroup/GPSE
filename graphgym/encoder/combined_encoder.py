import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('CombinedPSE')
class CombinedPSENodeEncoder(torch.nn.Module):
    """Combined Positional and Structural Encoding node encoder.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()

        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_CombinedPSE
        dim_pe = pecfg.dim_pe
        raw_dim = pecfg._raw_dim
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if raw_dim is None:
            raise ValueError("CombinedPSE raw dimension "
                             "'cfg.posenc_GPSE.raw_dim' not set up properly, "
                             "this should have been done automatically within "
                             "the compute_posenc_stats() function. Make sure "
                             "'cfg.dataset.combine_output_pestat' is True.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(raw_dim)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(raw_dim, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(raw_dim, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(raw_dim, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        if not hasattr(batch, "pestat_CombinedPSE"):
            raise ValueError("Precomputed 'pestat_CombinedPSE' variable is "
                             f"required for {self.__class__.__name__}; set "
                             "config 'posenc_CombinedPSE.enable' to True, and "
                             "also set 'dataset.combine_output_pestat' to True")

        pos_enc = batch.pestat_CombinedPSE  # (Num nodes) x (Raw dim)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_CombinedPSE = pos_enc
        return batch
