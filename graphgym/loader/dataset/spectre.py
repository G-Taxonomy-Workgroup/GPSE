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


class SPECTREDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return 'data.pt'

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(os.path.join(self.raw_dir, self.raw_file_names))
        nx_graphs = [nx.Graph(A.numpy()) for A in adjs]
        data_list = [from_networkx(g) for g in nx_graphs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
