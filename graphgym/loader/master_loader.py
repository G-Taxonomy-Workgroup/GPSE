import logging
import os
import os.path as osp
import time
import zipfile
from copy import deepcopy
from functools import partial
from typing import List, Optional
from urllib.parse import urljoin

import numpy as np
import requests
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (ZINC, GNNBenchmarkDataset, Planetoid,
                                      TUDataset, WikipediaNetwork)
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.loader import (load_ogb, load_pyg,
                                             set_dataset_attr)
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_loader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch_scatter import scatter_sum
from tqdm import tqdm, trange
from yacs.config import CfgNode as CN

from graphgym.encoder.gnn_encoder import gpse_process_batch
from graphgym.head.identity import IdentityHead
from graphgym.loader.dataset.aqsol_molecules import AQSOL
from graphgym.loader.dataset.coco_superpixels import COCOSuperpixels
from graphgym.loader.dataset.er_dataset import ERDataset
from graphgym.loader.dataset.spectre import SPECTREDataset
from graphgym.loader.dataset.malnet_tiny import MalNetTiny
from graphgym.loader.dataset.open_mol_graph import OpenMolGraph
from graphgym.loader.dataset.synthetic_wl import SyntheticWL
from graphgym.loader.dataset.voc_superpixels import VOCSuperpixels
from graphgym.loader.split_generator import prepare_splits, set_dataset_splits
from graphgym.transform.gnn_hash import GraphNormalizer, RandomGNNHash
from graphgym.transform.posenc_stats import compute_posenc_stats
from graphgym.transform.transforms import (VirtualNodePatchSingleton,
                                           clip_graphs_to_size,
                                           concat_x_and_pos,
                                           pre_transform_in_memory, typecast_x)
from graphgym.utils import get_device
from graphgym.wl_dataset import ToyWLDataset

import networkx as nx


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


def load_pretrained_gnn(cfg) -> Optional[GraphGymModule]:
    if cfg.posenc_GPSE.enable:
        assert cfg.posenc_GPSE.model_dir is not None
        return load_pretrained_gpse(cfg)
    else:
        return None, lambda: None


def load_pretrained_gpse(cfg) -> Optional[GraphGymModule]:
    if cfg.posenc_GPSE.model_dir is None:
        return None, lambda: None

    logging.info("[*] Setting up GPSE")
    path = cfg.posenc_GPSE.model_dir
    logging.info(f"    Loading pre-trained weights from {path}")
    model_state = torch.load(path, map_location="cpu")["model_state"]
    # Input dimension of the first module in the model weights
    cfg.share.pt_dim_in = dim_in = model_state[list(model_state)[0]].shape[1]
    logging.info(f"    Input dim of the pre-trained model: {dim_in}")
    # Hidden (representation) dimension and final output dimension
    if cfg.posenc_GPSE.gnn_cfg.head.startswith("inductive_hybrid"):
        # Hybrid head dimension inference
        cfg.share.num_graph_targets = model_state[list(model_state)[-1]].shape[0]
        node_head_bias_name = [
            i for i in model_state
            if i.startswith("model.post_mp.node_post_mp")][-1]
        if cfg.posenc_GPSE.gnn_cfg.head.endswith("multi"):
            head_idx = int(
                node_head_bias_name.split("node_post_mps.")[1].split(".model")[0])
            dim_out = head_idx + 1
        else:
            dim_out = model_state[node_head_bias_name].shape[0]
        cfg.share.num_node_targets = dim_out
        logging.info(f"    Graph emb outdim: {cfg.share.num_graph_targets}")
    elif cfg.posenc_GPSE.gnn_cfg.head == "inductive_node_multi":
        dim_out = len([
            1 for i in model_state
            if ("layer_post_mp" in i) and ("layer.model.weight" in i)
        ])
    else:
        dim_out = model_state[list(model_state)[-2]].shape[0]
    if cfg.posenc_GPSE.use_repr:
        cfg.share.pt_dim_out = cfg.posenc_GPSE.gnn_cfg.dim_inner
    else:
        cfg.share.pt_dim_out = dim_out
    logging.info(f"    Outdim of the pre-trained model: {cfg.share.pt_dim_out}")

    # HACK: Temporarily setting global config to default and overwrite GNN
    # configs using the ones from GPSE. Currently, there is no easy way to
    # repurpose the GraphGymModule to build a model using a specified cfg other
    # than completely overwriting the global cfg. [PyG v2.1.0]
    orig_gnn_cfg = CN(cfg.gnn.copy())
    orig_dataset_cfg = CN(cfg.dataset.copy())
    orig_model_cfg = CN(cfg.model.copy())
    plain_cfg = CN()
    set_cfg(plain_cfg)
    # Temporarily replacing the GNN config with the pre-trained GNN predictor
    cfg.gnn = cfg.posenc_GPSE.gnn_cfg
    # Resetting dataset config for bypassing the encoder settings
    cfg.dataset = plain_cfg.dataset
    # Resetting model config to make sure GraphGymModule uses the default GNN
    cfg.model = plain_cfg.model
    logging.info(f"Setting up GPSE using config:\n{cfg.posenc_GPSE.dump()}")

    # Construct model using the patched config and load trained weights
    model = GraphGymModule(dim_in, dim_out, cfg)
    model.load_state_dict(model_state)
    # Set the final linear layer to identity if we want to use the hidden repr
    if cfg.posenc_GPSE.use_repr:
        if cfg.posenc_GPSE.repr_type == "one_layer_before":
            model.model.post_mp.layer_post_mp.model[-1] = torch.nn.Identity()
        elif cfg.posenc_GPSE.repr_type == "no_post_mp":
            model.model.post_mp = IdentityHead()
        else:
            raise ValueError(f"Unknown repr_type {cfg.posenc_GPSE.repr_type!r}")
    model.eval()
    device = get_device(cfg.posenc_GPSE.accelerator, cfg.accelerator)
    model.to(device)
    logging.info(f"Pre-trained model constructed:\n{model}")

    # HACK: Construct bounded function to recover the original configurations
    # to be called right after the pre_transform_in_memory call with
    # compute_posenc_stats is done. This is necessary because things inside
    # GrapyGymModule checks for global configs to determine the behavior for
    # things like forward. To FIX this in the future, need to seriously
    # make sure modules like this store the fixed value at __init__, instead of
    # dynamically looking up configs at runtime.
    def _recover_orig_cfgs():
        cfg.gnn = orig_gnn_cfg
        cfg.dataset = orig_dataset_cfg
        cfg.model = orig_model_cfg

        # Release pretrained model from CUDA memory
        model.to("cpu")
        torch.cuda.empty_cache()

    return model, _recover_orig_cfgs


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'MalNetTiny':
            dataset = preformat_MalNetTiny(dataset_dir, feature_set=name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'VOCSuperpixels':
            dataset = preformat_VOCSuperpixels(dataset_dir, name,
                                               cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'COCOSuperpixels':
            dataset = preformat_COCOSuperpixels(dataset_dir, name,
                                                cfg.dataset.slic_compactness)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError("crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)

        elif pyg_dataset_id == 'AQSOL':
            dataset = preformat_AQSOL(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    elif format == 'er':
        def set_xy(data):
            data.x = torch.ones(data.num_nodes, 1)
            data.y = 1
            return data
        dataset = ERDataset(osp.join(dataset_dir, 'er'))
        pre_transform_in_memory(dataset, set_xy, show_progress=True)

    elif format == 'SPECTRE':
        dataset_dir = osp.join(dataset_dir, 'SPECTRE')
        def set_xy(data):
            data.x = torch.ones(data.num_nodes, 1)
            data.y = 1
            return data
        dataset = SPECTREDataset(dataset_dir, name)
        pre_transform_in_memory(dataset, set_xy, show_progress=True)


    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbg'):
            dataset = preformat_OGB_Graph(dataset_dir, name.replace('_', '-'))
            dataset = dataset.index_select(range(int(len(dataset)*cfg.dataset.subset_ratio)))

        elif name.startswith('ogbn'):
            dataset = preformat_OGB_Node(dataset_dir, name.replace('_', '-'))

        elif name.startswith('PCQM4Mv2-'):
            subset = name.split('-', 1)[1]
            dataset = preformat_OGB_PCQM4Mv2(dataset_dir, subset)

        elif name.startswith('peptides-'):
            dataset = preformat_Peptides(dataset_dir, name)

        ### Link prediction datasets.
        elif name.startswith('ogbl-'):
            # GraphGym default loader.
            dataset = load_ogb(name, dataset_dir)

            # OGB link prediction datasets are binary classification tasks,
            # however the default loader creates float labels => convert to int.
            def convert_to_int(ds, prop):
                tmp = getattr(ds.data, prop).int()
                set_dataset_attr(ds, prop, tmp, len(tmp))

            convert_to_int(dataset, 'train_edge_label')
            convert_to_int(dataset, 'val_edge_label')
            convert_to_int(dataset, 'test_edge_label')

        elif name.startswith('PCQM4Mv2Contact-'):
            dataset = preformat_PCQM4Mv2Contact(dataset_dir, name)

        else:
            raise ValueError(f"Unsupported OGB(-derived) dataset: {name}")

    elif format == 'OpenMolGraph':
        dataset = preformat_OpenMolGraph(dataset_dir, name=name)

    elif format == 'SyntheticWL':
        dataset = preformat_SyntheticWL(dataset_dir, name=name)

    elif format == 'ToyWL':
        dataset = preformat_ToyWL(dataset_dir, name=name)

    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Preprocess for reducing the molecular dataset to unique structured graphs
    if cfg.dataset.unique_mol_graphs:
        dataset = get_unique_mol_graphs_via_smiles(dataset,
                                                   cfg.dataset.umg_train_ratio,
                                                   cfg.dataset.umg_val_ratio,
                                                   cfg.dataset.umg_test_ratio,
                                                   cfg.dataset.umg_random_seed)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if (key.startswith(('posenc_', 'graphenc_')) and pecfg.enable
                and key != "posenc_GPSE"):  # GPSE handled separately
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True)
        if hasattr(dataset.data, "y") and len(dataset.data.y.shape) == 2:
            cfg.share.num_node_targets = dataset.data.y.shape[1]
        if hasattr(dataset.data, "y_graph"):
            cfg.share.num_graph_targets = dataset.data.y_graph.shape[1]
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    if cfg.hash_feat.enable:  # TODO: Improve handling here
        try:
            pre_transform_in_memory(dataset, RandomGNNHash(), show_progress=True)
        except:
            logging.info("Hashing to be computed later")

    if cfg.graph_norm.enable:
        pre_transform_in_memory(dataset, GraphNormalizer(), show_progress=True)

    dataset.transform_list = None
    randse_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('randenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            randse_enabled_list.append(pe_name)
    if randse_enabled_list:
        set_random_se(dataset, randse_enabled_list)

    if cfg.virtual_node:
        set_virtual_node(dataset)

    if dataset.transform_list is not None:
        dataset.transform = T.Compose(dataset.transform_list)

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNA') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])
        # print(f"Indegrees: {cfg.gt.pna_degrees}")
        # print(f"Avg:{np.mean(cfg.gt.pna_degrees)}")

    # Precompute GPSE if it is enabled
    if cfg.posenc_GPSE.enable:
        precompute_gpse(cfg, dataset)

    # Precompute GraphLog embeddings if it is enabled
    if cfg.posenc_GraphLog.enable:
        from graphgym.encoder.graphlog_encoder import precompute_graphlog
        precompute_graphlog(cfg, dataset)

    logging.info(f"Finished processing data:\n  {dataset.data}")

    return dataset


def gpse_io(
    dataset,
    mode: str = "save",
    name: Optional[str] = None,
    tag: Optional[str] = None,
    auto_download: bool = True,
):
    assert tag, "Please provide a tag for saving/loading GPSE (e.g., '1.0')"
    pse_dir = dataset.processed_dir
    gpse_data_path = osp.join(pse_dir, f"gpse_{tag}_data.pt")
    gpse_slices_path = osp.join(pse_dir, f"gpse_{tag}_slices.pt")

    def maybe_download_gpse():
        is_complete = osp.isfile(gpse_data_path) and osp.isfile(gpse_slices_path)
        if is_complete or not auto_download:
            return

        if name is None:
            raise ValueError("Please specify the dataset name for downloading.")

        if tag != "1.0":
            raise ValueError(f"Invalid tag {tag!r}, currently only support '1.0")
        # base_url = "https://sandbox.zenodo.org/record/1219850/files/"  # 1.0.dev
        base_url = "https://zenodo.org/record/8145344/files/"  # 1.0
        fname = f"{name}_{tag}.zip"
        url = urljoin(base_url, fname)
        save_path = osp.join(pse_dir, fname)

        # Stream download
        with requests.get(url, stream=True) as r:
            if r.ok:
                total_size_in_bytes = int(r.headers.get("content-length", 0))
                pbar = tqdm(
                    total=total_size_in_bytes,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading {url}",
                )
                with open(save_path, "wb") as file:
                    for data in r.iter_content(1024):
                        pbar.update(len(data))
                        file.write(data)
                pbar.close()

            else:
                meta_url = base_url.replace("/record/", "/api/records/")
                meta_url = meta_url.replace("/files/", "")
                meta_r = requests.get(meta_url)
                if meta_r.ok:
                    files = meta_r.json()["files"]
                    opts = [i["key"].rsplit(".zip")[0] for i in files]
                else:
                    opts = []

                opts_str = "\n".join(sorted(opts))
                raise requests.RequestException(
                    f"Fail to download from {url} ({r!r}). Available options "
                    f"for {tag=!r} are:\n{opts_str}",
                )

        # Unzip files and cleanup
        logging.info(f"Extracting {save_path}")
        with zipfile.ZipFile(save_path, "r") as f:
            f.extractall(pse_dir)
        os.remove(save_path)

    if mode == "save":
        torch.save(dataset.data.pestat_GPSE, gpse_data_path)
        torch.save(dataset.slices["pestat_GPSE"], gpse_slices_path)
        logging.info(f"Saved pre-computed GPSE ({tag}) to {pse_dir}")

    elif mode == "load":
        maybe_download_gpse()
        dataset.data.pestat_GPSE = torch.load(gpse_data_path, map_location="cpu")
        dataset.slices["pestat_GPSE"] = torch.load(gpse_slices_path, map_location="cpu")
        logging.info(f"Loaded pre-computed GPSE ({tag}) from {pse_dir}")

    else:
        raise ValueError(f"Unknown io mode {mode!r}.")


@torch.no_grad()
def precompute_gpse(cfg, dataset):
    dataset_name = f"{cfg.dataset.format}-{cfg.dataset.name}"
    tag = cfg.posenc_GPSE.tag
    if cfg.posenc_GPSE.from_saved:
        gpse_io(dataset, "load", name=dataset_name, tag=tag)
        cfg.share.pt_dim_out = dataset.data.pestat_GPSE.shape[1]
        return

    # Load GPSE model and prepare bounded method to recover original configs
    gpse_model, _recover_orig_cfgs = load_pretrained_gpse(cfg)

    # Temporarily replace the transformation
    orig_dataset_transform = dataset.transform
    dataset.transform = None
    if cfg.posenc_GPSE.virtual_node:
        dataset.transform = VirtualNodePatchSingleton()

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

    loader = DataLoader(dataset, batch_size=cfg.posenc_GPSE.loader.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0)

    # Batched GPSE precomputation loop
    data_list = []
    curr_idx = 0
    pbar = trange(len(dataset), desc="Pre-computing GPSE")
    tic = time.perf_counter()
    for batch in loader:
        batch_out, batch_ptr = gpse_process_batch(gpse_model, batch)

        batch_out = batch_out.to("cpu", non_blocking=True)
        # Need to wait for batch_ptr to finish transfering so that start and
        # end indices are ready to use
        batch_ptr = batch_ptr.to("cpu", non_blocking=False)

        for start, end in zip(batch_ptr[:-1], batch_ptr[1:]):
            data = dataset.get(curr_idx)
            if cfg.posenc_GPSE.virtual_node:
                end = end - 1
            data.pestat_GPSE = batch_out[start:end]
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

    if cfg.posenc_GPSE.save:
        gpse_io(dataset, "save", tag=tag)

    timestr = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - tic))
    logging.info(f"Finished GPSE pre-computation, took {timestr}")

    # Release resource and recover original configs
    del gpse_model
    torch.cuda.empty_cache()
    _recover_orig_cfgs()


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    elif name == "CSL":
        # CSL does have predefined split. Need to use cv or random splits.
        dataset = GNNBenchmarkDataset(root=dataset_dir, name=name,
                                      split="train")
        pre_transform_in_memory(dataset, T.Constant(cat=False))
        return dataset
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_OpenMolGraph(dataset_dir, name):
    """Load and preformat Open Molecular Graph datasets.

    Args:
        dataset_dir: path where to store the cached dataset.
        name: name of the specific dataset in the OpenMolGraph collection.

    Returns:
        PyG dataset object

    Notes:
        This dataset does not come with pre-defined splits. Need to use split
        generation functionalities such as umg to set up splits.

    """
    dataset = OpenMolGraph(dataset_dir, name=name,
                           n_jobs=max(1, cfg.num_workers))
    return dataset


def preformat_SyntheticWL(dataset_dir, name):
    """Load and preformat synthetic WL graph datasets.

    Args:
        dataset_dir: path where to store the cached dataset.
        name: name of the specific dataset in the SyntheticWL collection.
            Available options are: 'exp', 'cexp', and 'sr25'.

    Returns:
        PyG dataset object

    """
    dataset = SyntheticWL(dataset_dir, name=name)
    if name.lower() == "sr25":
        # Evaluate on training, so train/val/test are the same split
        dataset = join_dataset_splits([deepcopy(dataset) for _ in range(3)])
    return dataset


def preformat_ToyWL(dataset_dir, name=None):
    """Load and preformat toy WL graph datasets.

    Args:
        dataset_dir: path where to store the cached dataset.
        name: name of the specific dataset in the SyntheticWL collection.
            Available options are: 'exp', 'cexp', and 'sr25'.

    Returns:
        PyG dataset object

    """
    dataset = ToyWLDataset(dataset_dir, name)
    dataset = join_dataset_splits([deepcopy(dataset) for _ in range(3)])
    return dataset


def preformat_MalNetTiny(dataset_dir, feature_set):
    """Load and preformat Tiny version (5k graphs) of MalNet

    Args:
        dataset_dir: path where to store the cached dataset
        feature_set: select what node features to precompute as MalNet
            originally doesn't have any node nor edge features

    Returns:
        PyG dataset object
    """
    if feature_set in ['none', 'Constant']:
        tf = T.Constant()
    elif feature_set == 'OneHotDegree':
        tf = T.OneHotDegree()
    elif feature_set == 'LocalDegreeProfile':
        tf = T.LocalDegreeProfile()
    else:
        raise ValueError(f"Unexpected transform function: {feature_set}")

    dataset = MalNetTiny(dataset_dir)
    dataset.name = 'MalNetTiny'
    logging.info(f'Computing "{feature_set}" node features for MalNetTiny.')
    pre_transform_in_memory(dataset, tf)

    split_dict = dataset.get_idx_split()
    dataset.split_idxs = [split_dict['train'],
                          split_dict['valid'],
                          split_dict['test']]

    return dataset


def preformat_OGB_Graph(dataset_dir, name):
    """Load and preformat OGB Graph Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Graph dataset

    Returns:
        PyG dataset object
    """
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    if name == 'ogbg-ppa':
        # ogbg-ppa doesn't have any node features, therefore add zeros but do
        # so dynamically as a 'transform' and not as a cached 'pre-transform'
        # because the dataset is big (~38.5M nodes), already taking ~31GB space
        def add_zeros(data):
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
            return data

        dataset.transform = add_zeros
    elif name == 'ogbg-code2':
        from graphgym.loader.ogbg_code2_utils import idx2vocab, \
            get_vocab_mapping, augment_edge, encode_y_to_arr
        num_vocab = 5000  # The number of vocabulary used for sequence prediction
        max_seq_len = 5  # The maximum sequence length to predict

        seq_len_list = np.array([len(seq) for seq in dataset.data.y])
        logging.info(f"Target sequences less or equal to {max_seq_len} is "
                     f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

        # Building vocabulary for sequence prediction. Only use training data.
        vocab2idx, idx2vocab_local = get_vocab_mapping(
            [dataset.data.y[i] for i in s_dict['train']], num_vocab)
        logging.info(f"Final size of vocabulary is {len(vocab2idx)}")
        idx2vocab.extend(idx2vocab_local)  # Set to global variable to later access in CustomLogger

        # Set the transform function:
        # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
        # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
        dataset.transform = T.Compose(
            [augment_edge,
             lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

        # Subset graphs to a maximum size (number of nodes) limit.
        pre_transform_in_memory(dataset, partial(clip_graphs_to_size,
                                                 size_limit=1000))

    return dataset


def preformat_OGB_Node(dataset_dir, name):
    """Load and preformat OGB Node Property Prediction datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific OGB Node dataset

    Returns:
        PyG dataset object
    """
    dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
    splits = dataset.get_idx_split()
    dataset.split_idxs = [splits[s] for s in ['train', 'valid', 'test']]
    pre_transform_in_memory(dataset, T.ToUndirected())

    if "proteins" in name.lower():
        # Prepare default node features as the summed edge weights (8 dim)
        x = scatter_sum(dataset[0].edge_attr, dataset[0].edge_index[0], dim=0)
        set_dataset_attr(dataset, 'x', x, x.shape)

    return dataset


def preformat_OGB_PCQM4Mv2(dataset_dir, name):
    """Load and preformat PCQM4Mv2 from OGB LSC.

    OGB-LSC provides 4 data index splits:
    2 with labeled molecules: 'train', 'valid' meant for training and dev
    2 unlabeled: 'test-dev', 'test-challenge' for the LSC challenge submission

    We will take random 150k from 'train' and make it a validation set and
    use the original 'valid' as our testing set.

    Note: PygPCQM4Mv2Dataset requires rdkit

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of the training set

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from ogb.lsc import PygPCQM4Mv2Dataset
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2Dataset, '
                      'make sure RDKit is installed.')
        raise e

    dataset = PygPCQM4Mv2Dataset(root=dataset_dir)
    split_idx = dataset.get_idx_split()

    rng = default_rng(seed=42)
    train_idx = rng.permutation(split_idx['train'].numpy())
    train_idx = torch.from_numpy(train_idx)

    # Leave out 150k graphs for a new validation set.
    valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
    if name == 'full':
        split_idxs = [train_idx,  # Subset of original 'train'.
                      valid_idx,  # Subset of original 'train' as validation set.
                      split_idx['valid']  # The original 'valid' as testing set.
                      ]

    elif name == 'subset':
        # Further subset the training set for faster debugging.
        subset_ratio = 0.1
        subtrain_idx = train_idx[:int(subset_ratio * len(train_idx))]
        subvalid_idx = valid_idx[:50000]
        subtest_idx = split_idx['valid']  # The original 'valid' as testing set.

        dataset = dataset[torch.cat([subtrain_idx, subvalid_idx, subtest_idx])]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(subtrain_idx), len(subvalid_idx), len(subtest_idx)
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]

    elif name == 'inference':
        split_idxs = [split_idx['valid'],  # The original labeled 'valid' set.
                      split_idx['test-dev'],  # Held-out unlabeled test dev.
                      split_idx['test-challenge']  # Held-out challenge test set.
                      ]

        dataset = dataset[torch.cat(split_idxs)]
        data_list = [data for data in dataset]
        dataset._indices = None
        dataset._data_list = data_list
        dataset.data, dataset.slices = dataset.collate(data_list)
        n1, n2, n3 = len(split_idxs[0]), len(split_idxs[1]), len(split_idxs[2])
        split_idxs = [list(range(n1)),
                      list(range(n1, n1 + n2)),
                      list(range(n1 + n2, n1 + n2 + n3))]
        # Check prediction targets.
        assert (all([not torch.isnan(dataset[i].y)[0] for i in split_idxs[0]]))
        assert (all([torch.isnan(dataset[i].y)[0] for i in split_idxs[1]]))
        assert (all([torch.isnan(dataset[i].y)[0] for i in split_idxs[2]]))

    else:
        raise ValueError(f'Unexpected OGB PCQM4Mv2 subset choice: {name}')
    dataset.split_idxs = split_idxs
    return dataset


def preformat_PCQM4Mv2Contact(dataset_dir, name):
    """Load PCQM4Mv2-derived molecular contact link prediction dataset.

    Note: This dataset requires RDKit dependency!

    Args:
       dataset_dir: path where to store the cached dataset
       name: the type of dataset split: 'shuffle', 'num-atoms'

    Returns:
       PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary
        from graphgym.loader.dataset.pcqm4mv2_contact import \
            PygPCQM4Mv2ContactDataset, \
            structured_neg_sampling_transform
    except Exception as e:
        logging.error('ERROR: Failed to import PygPCQM4Mv2ContactDataset, '
                      'make sure RDKit is installed.')
        raise e

    split_name = name.split('-', 1)[1]
    dataset = PygPCQM4Mv2ContactDataset(dataset_dir, subset='530k')
    # Inductive graph-level split (there is no train/test edge split).
    s_dict = dataset.get_idx_split(split_name)
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    if cfg.dataset.resample_negative:
        dataset.transform = structured_neg_sampling_transform
    return dataset


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from graphgym.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from graphgym.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    return dataset


def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_AQSOL(dataset_dir):
    """Load and preformat AQSOL datasets.

    Args:
        dataset_dir: path where to store the cached dataset

    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [AQSOL(root=dataset_dir, split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_VOCSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat VOCSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [VOCSuperpixels(root=dataset_dir, name=name,
                        slic_compactness=slic_compactness,
                        split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def preformat_COCOSuperpixels(dataset_dir, name, slic_compactness):
    """Load and preformat COCOSuperpixels dataset.

    Args:
        dataset_dir: path where to store the cached dataset
    Returns:
        PyG dataset object
    """
    dataset = join_dataset_splits(
        [COCOSuperpixels(root=dataset_dir, name=name,
                         slic_compactness=slic_compactness,
                         split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


def smiles_from_graph(
    node_list: List[str],
    adjacency_matrix: np.ndarray,
) -> str:
    """Create a SMILES string from a given graph.

    Modified from https://stackoverflow.com/a/51242251/12519564

    """
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        raise ModuleNotFoundError("rdkit is not installed yet")

    # Create empty editable mol object
    mol = Chem.RWMol()

    # Add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # Add bonds between adjacent atoms
    for i, j in zip(*np.nonzero(adjacency_matrix)):
        # Only traverse half the matrix
        if j <= i:
            continue

        if adjacency_matrix[i, j] >= 1:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[i], node_to_idx[j], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)

    return smiles


def get_unique_mol_graphs_via_smiles(
    dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 0,
):
    if (sum_ratio := train_ratio + val_ratio + test_ratio) > 1:
        raise ValueError("Total ratio (train + val + test) must be below 1 "
                         f"got {sum_ratio:.2f}")

    old_size = len(dataset)
    all_smiles = []
    for i in tqdm(dataset, total=old_size,
                  desc="Extracting unique SMILES (ignoring atom/bond types)"):
        num_nodes = i.num_nodes
        trivial_c_atoms = ["C"] * num_nodes
        adj = torch.sparse_coo_tensor(
            i.edge_index,
            torch.ones(i.edge_index.size(1), dtype=torch.float),
            size=(num_nodes, num_nodes),
        ).to_dense().numpy()
        all_smiles.append(smiles_from_graph(trivial_c_atoms, adj))
    unique_smiles = sorted(set(all_smiles))

    unique_graphs = []
    for smiles in tqdm(unique_smiles, total=len(unique_smiles),
                  desc="Filtering unique graphs based on SMILES"):
        g = from_smiles(smiles)
        if (g.num_nodes > 1) and (g.edge_index.shape[1] > 1):
            delattr(g, "smiles")
            delattr(g, "edge_attr")
            unique_graphs.append(g)

    num_unique = len(unique_graphs)
    split_points = [int(num_unique * train_ratio),
                    int(num_unique * (1 - val_ratio - test_ratio)),
                    int(num_unique * (1 - test_ratio))]
    rng = np.random.default_rng(random_seed)
    new_split_idxs = np.split(rng.permutation(num_unique), split_points)
    new_split_idxs.pop(1)  # pop the fill-in split
    # Reorder graphs into train/val/test (poentially remove the fill-in split)
    unique_graphs = [unique_graphs[i] for i in np.hstack(new_split_idxs)]
    new_size = len(unique_graphs)

    if test_ratio == 1:
        # Evaluation only, pad "training" and "evaluation" set with the first
        # graph
        new_split_idxs[0] = np.array([num_unique])
        new_split_idxs[1] = np.array([num_unique + 1])
        unique_graphs.append(unique_graphs[-1])
        unique_graphs.append(unique_graphs[-1])

    # E.g. [[0, 1], [0, 1, 2], [0]]
    dataset.split_idxs = [torch.arange(idxs.size) for idxs in new_split_idxs]
    if train_ratio != 1:
        # E.g. [[0, 1], [2, 3, 4], [5]]
        for i in range(1, len(dataset.split_idxs)):
            dataset.split_idxs[i] += dataset.split_idxs[i - 1][-1] + 1

    dataset.data, dataset.slices = dataset.collate(unique_graphs)
    # We need to remove _data_list because its presence will bypass the
    # indentded data slicing using the .slices attribute.
    # https://github.com/pyg-team/pytorch_geometric/blob/f0c72186286f257778c1d9293cfd0d35472d30bb/torch_geometric/data/in_memory_dataset.py#L75-L94
    dataset._data_list = [None] * len(dataset)
    dataset._indices = None

    logging.info("[*] Dataset reduced to unique molecular structure graphs\n"
                 f"    Number of graphs before: {old_size:,}\n"
                 f"    Number of graphs after: {new_size:,}\n"
                 f"    Train size: {len(new_split_idxs[0]):,} "
                 f"(first five: {new_split_idxs[0][:5]})\n"
                 f"    Validation size: {len(new_split_idxs[1]):,} "
                 f"(first five: {new_split_idxs[1][:5]})\n"
                 f"    Test size: {len(new_split_idxs[2]):,} "
                 f"(first five: {new_split_idxs[2][:5]})\n"
                 f"    {dataset.data}\n")

    return dataset


def set_random_se(dataset, pe_types):

    if 'FixedSE' in pe_types:
        def randomSE_Fixed(data):
            N = data.num_nodes
            stat = np.full(shape=(N, cfg.randenc_FixedSE.dim_pe), fill_value=1).astype('float32')
            data.x = torch.from_numpy(stat)
            return data

        dataset.transform_list = [randomSE_Fixed]

    if 'NormalSE' in pe_types:
        def randomSE_Normal(data):
            N = data.num_nodes
            rand = np.random.normal(loc=0, scale=1.0, size=(N, cfg.randenc_NormalSE.dim_pe)).astype('float32')
            data.x = torch.from_numpy(rand)
            return data

        dataset.transform_list = [randomSE_Normal]

    if 'UniformSE' in pe_types:
        def randomSE_Uniform(data):
            N = data.num_nodes
            rand = np.random.uniform(low=0.0, high=1.0, size=(N, cfg.randenc_UniformSE.dim_pe)).astype('float32')
            data.x = torch.from_numpy(rand)
            return data

        dataset.transform_list = [randomSE_Uniform]

    if 'BernoulliSE' in pe_types:
        def randomSE_Bernoulli(data):
            N = data.num_nodes
            rand = np.random.uniform(low=0.0, high=1.0, size=(N, cfg.randenc_BernoulliSE.dim_pe))
            rand = (rand < cfg.randenc_BernoulliSE.threshold).astype('float32')
            data.x = torch.from_numpy(rand)
            return data

        dataset.transform_list = [randomSE_Bernoulli]


def set_virtual_node(dataset):
    if dataset.transform_list is None:
        dataset.transform_list = []
    dataset.transform_list.append(VirtualNodePatchSingleton())
