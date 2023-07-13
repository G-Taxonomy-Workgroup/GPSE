import itertools
import logging

import numpy as np
import torch
import torch_geometric.transforms as T
from joblib import Parallel, delayed
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import add_self_loops, remove_self_loops, subgraph
from tqdm import tqdm

from graphgym.utils import grouper

BATCH_SIZE = 100


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def parallel_pre_transform_in_memory(dataset, transform_func,
                                     show_progress=False):
    """Parallel version of pre_transform_in_memory."""
    if transform_func is None:
        return dataset

    parallel = Parallel(n_jobs=max(1, cfg.num_workers))
    func = delayed(get_batched_func(transform_func))
    pbar = tqdm(grouper(dataset, n=BATCH_SIZE),
                total=int(np.ceil(len(dataset) / BATCH_SIZE)),
                disable=not show_progress,
                mininterval=1)
    batched_data_lists = parallel(func(batch) for batch in pbar)
    data_list = list(filter(None, itertools.chain(*batched_data_lists)))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def get_batched_func(func, *, batch_size: int = 1000):

    def batched_func(batched_data_list):
        return [func(data) for data in batched_data_list if data is not None]

    return batched_func


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data


class VirtualNodePatchSingleton(T.VirtualNode):
    def __call__(self, data):
        if data.edge_index.numel() == 0:
            logging.debug(f"Data with empty edge set {data}")
            data.edge_index, data.edge_attr = add_self_loops(
                data.edge_index, data.edge_attr, num_nodes=data.num_nodes)
            data = super().__call__(data)
            if hasattr(data, "y_graph"):  # potentially fix hybrid head
                data.y_graph = data.y_graph[:1]
            data.edge_index, data.edge_attr = remove_self_loops(
                data.edge_index, data.edge_attr)
            logging.debug(f"Fixed data due to empty edge set {data}")
        else:
            data = super().__call__(data)
        return data
