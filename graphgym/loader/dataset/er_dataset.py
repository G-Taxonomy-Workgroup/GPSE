from typing import Optional, Callable, List

import os
import os.path as osp
import pickle

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils.convert import from_networkx


class ERDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = ''

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def create_graph(self):
        n = np.random.randint(cfg.er.n_min, cfg.er.n_max + 1)
        g = nx.fast_gnp_random_graph(n, p=cfg.er.p)
        while not nx.is_connected(g):
            g = nx.fast_gnp_random_graph(n, p=cfg.er.p)

        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_pyg = from_networkx(g)
        return g_pyg

    def process(self):
        # Read data into huge `Data` list.
        data_list = [self.create_graph() for i in range(cfg.er.num_samples)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
