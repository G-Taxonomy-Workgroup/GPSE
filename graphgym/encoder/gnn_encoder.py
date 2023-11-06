from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.loader import NeighborLoader
from tqdm import trange

from graphgym.utils import get_device


@torch.no_grad()
def gpse_process_batch(model, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        batch: A PyG batch object holding a batch of graphs

    Returns:
        A two tuple of tensors correspond to the stacked GPSE encodings and
        the pointers indicating individual graphs
    """
    # Generate random features for the encoder
    n = batch.num_nodes
    rand_type = cfg.posenc_GPSE.rand_type
    if (dim_in := cfg.share.get("pt_dim_in")) is None:
        raise AttributeError("cfg.share.pt_dim_in has not been set up yet, "
                             "please make sure load_pretrained_gnn function "
                             "has been called appropriately (usually it "
                             "should be given that cfg.posenc_GPSE.model_dir "
                             "is specified correctly)")

    # Prepare input distributions for GPSE
    if rand_type == "NormalSE":
        rand = np.random.normal(loc=0, scale=1.0, size=(n, dim_in))
    elif rand_type == "UniformSE":
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
    elif rand_type == "BernoulliSE":
        rand = np.random.uniform(low=0.0, high=1.0, size=(n, dim_in))
        rand = (rand < cfg.randenc_BernoulliSE.threshold)
    else:
        raise ValueError(f"Unknown {rand_type=!r}")
    batch.x = torch.from_numpy(rand.astype("float32"))

    if cfg.posenc_GPSE.virtual_node:
        # HACK: We need to reset virtual node features to zeros to match the
        # pretraining setting (virtual node applied after random node features
        # are set, and the default node features for the virtual node are all
        # zeros). Can potentially test if initializing virtual node features to
        # random features is better than setting them to zeros.
        for i in batch.ptr[1:]:
            batch.x[i - 1] = 0

    # Generate encodings using the pretrained encoder
    device = get_device(cfg.posenc_GPSE.accelerator, cfg.accelerator)
    if cfg.posenc_GPSE.loader.type == "neighbor":
        num_neighbors = cfg.posenc_GPSE.loader.num_neighbors
        fillval = cfg.posenc_GPSE.loader.fill_num_neighbors
        diff = cfg.posenc_GPSE.gnn_cfg.layers_mp - len(num_neighbors)
        if fillval > 0 and diff > 0:
            num_neighbors += [fillval] * diff

        loader = NeighborLoader(batch, num_neighbors=num_neighbors,
                                batch_size=cfg.posenc_GPSE.loader.batch_size,
                                shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=True)

        out_list = []
        pbar = trange(batch.num_nodes, position=2)
        for i, batch in enumerate(loader):
            out, _ = model(batch.to(device))
            out = out[:batch.batch_size].to("cpu", non_blocking=True)
            out_list.append(out)
            pbar.update(batch.batch_size)
        out = torch.vstack(out_list)
    elif cfg.posenc_GPSE.loader.type == "full":
        out, _ = model(batch.to(device))
        out = out.to("cpu")
    else:
        raise ValueError(f"Unknown loader: {cfg.posenc_GPSE.loader.type!r}")

    return out, batch.ptr


@register_node_encoder("GPSE")
class GNNNodeEncoder(nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        """Pre-trained GNN P/SE predictor encoder.

        Args:
            dim_emb: Size of final node embedding
            expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)

        """
        super().__init__()

        pecfg = cfg.posenc_GPSE
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        gpse_dim_out = cfg.share.pt_dim_out
        dim_pe = pecfg.dim_pe
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model

        if expand_x:
            dim_in = cfg.share.dim_in  # expected original input node feat dim
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        self.raw_norm = None
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(gpse_dim_out)

        self.dropout_be = nn.Dropout(p=pecfg.input_dropout_be)
        self.dropout_ae = nn.Dropout(p=pecfg.input_dropout_ae)

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(gpse_dim_out, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(gpse_dim_out, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(gpse_dim_out, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        if not hasattr(batch, "pestat_GPSE"):
            raise ValueError("Precomputed 'pestat_GPSE' variable is "
                             "required for GNNNodeEncoder; set config "
                             "'posenc_pestat_GPSE.enable' to True")

        pos_enc = batch.pestat_GPSE

        pos_enc = self.dropout_be(pos_enc)
        pos_enc = self.raw_norm(pos_enc) if self.raw_norm else pos_enc
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe
        pos_enc = self.dropout_ae(pos_enc)

        # Expand node features if needed
        h = self.linear_x(batch.x) if self.expand_x else batch.x

        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)

        return batch
