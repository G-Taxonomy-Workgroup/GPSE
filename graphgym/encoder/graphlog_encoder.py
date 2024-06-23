"""GraphLog GINE model adpated from https://github.com/DeepGraphLearning/GraphLoG/tree/main

Used for extracting representations from the pre-trained model.

Adaptations

1. self-loop token last, misc second last
2. set bond directions to none for all (override the bond stereo channel from ogb)

"""
import os
import logging
import time

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import trange

URL = "https://github.com/DeepGraphLearning/GraphLoG/raw/main/models/graphlog.pth"

GRAPHLOG_LAYERS = 5
GRPPHLOG_DIM = 300
GRAPHLOG_PRECOMP_BATCHSIZE = 128

NUM_ATOM_TYPE = 120  # including the extra mask tokens
NUM_CHIRALITY_TAG = 3

NUM_BOND_TYPE = 6  # single, double, triple, aromatic, misc, selfloop
NUM_BOND_DIRECTION = 3


@torch.no_grad()
def precompute_graphlog(cfg, dataset):
    path = cfg.posenc_GraphLog.model_dir
    if not os.path.isfile(path):
        logging.info(f"Downloading GraphLog pre-trained weights from {URL}")
        with requests.get(URL) as r, open(path, "wb") as f:
            if not r.ok:
                raise requests.RequestException(
                    f"Failed to download weights from {URL} ({r!r})")
            f.write(r.content)

    # Load pretrained GraphLog model
    model = GraphLog(GRAPHLOG_LAYERS, GRPPHLOG_DIM)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(cfg.accelerator)

    # Temporarily replace the transformation
    orig_dataset_transform = dataset.transform
    dataset.transform = None

    # Remove split indices, to be recovered at the end of the precomputation
    tmp_store = {}
    for name in ["train_mask", "val_mask", "test_mask", "train_graph_index",
                 "val_graph_index", "test_graph_index", "train_edge_index",
                 "val_edge_index", "test_edge_index"]:
        if (name in dataset.data) and (dataset.slices is None
                                       or name in dataset.slices):
            tmp_store_data = dataset.data.pop(name)
            tmp_store_slices = dataset.slices.pop(name) if dataset.slices else None
            tmp_store[name] = (tmp_store_data, tmp_store_slices)

    loader = DataLoader(dataset, batch_size=GRAPHLOG_PRECOMP_BATCHSIZE,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0)

    # Batched GraphLog precomputation loop
    data_list = []
    curr_idx = 0
    pbar = trange(len(dataset), desc="Pre-computing GraphLog embeddings")
    tic = time.perf_counter()
    for batch in loader:
        # batch_out, batch_ptr = graphlog_process_batch(model, batch)
        batch_out = model(batch.to(cfg.accelerator)).to("cpu")
        batch_ptr = batch.ptr.to("cpu")

        for start, end in zip(batch_ptr[:-1], batch_ptr[1:]):
            data = dataset.get(curr_idx)
            data.pestat_GraphLog = batch_out[start:end]
            data_list.append(data)
            curr_idx += 1

        pbar.update(len(batch_ptr) - 1)
    pbar.close()

    # Collate dataset and reset indicies and data list
    dataset.transform = orig_dataset_transform
    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)

    # Recover split indices
    for name, (tmp_store_data, tmp_store_slices) in tmp_store.items():
        dataset.data[name] = tmp_store_data
        if tmp_store_slices is not None:
            dataset.slices[name] = tmp_store_slices
    dataset._data_list = None

    timestr = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - tic))
    logging.info(f"Finished GraphLog embedding pre-computation, took {timestr}")


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = 5  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # NOTE: overwrite bond directions to NONE since this is not available
        # from OGB processed graphs
        edge_attr[:, 1] = 0

        edge_embeddings = (self.edge_embedding1(edge_attr[:, 0])
                           + self.edge_embedding2(edge_attr[:, 1]))

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GraphLog(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, drop_ratio=0):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.relu(h) if layer == self.num_layer - 1 else h
            h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation


@register_node_encoder("GraphLog")
class GraphLogNodeEncoder(nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        """Pre-trained GNN P/SE predictor encoder.

        Args:
            dim_emb: Size of final node embedding
            expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)

        """
        super().__init__()

        pecfg = cfg.posenc_GraphLog
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        dim = GRPPHLOG_DIM
        dim_pe = pecfg.dim_pe
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model

        if expand_x:
            dim_in = cfg.share.dim_in  # expected original input node feat dim
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        self.raw_norm = None
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(dim)

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(dim, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(dim, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        if not hasattr(batch, "pestat_GraphLog"):
            raise ValueError("Precomputed 'pestat_GraphLog' variable is "
                             "required for GNNNodeEncoder; set config "
                             "'posenc_pestat_GraphLog.enable' to True")

        pos_enc = batch.pestat_GraphLog

        pos_enc = self.raw_norm(pos_enc) if self.raw_norm else pos_enc
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        h = self.linear_x(batch.x) if self.expand_x else batch.x

        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)

        return batch
