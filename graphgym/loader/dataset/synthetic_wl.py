import os.path as osp
import shutil
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import trange

LOCAL_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class SyntheticWL(InMemoryDataset):
    r"""Synthetic graphs dataset collected from https://arxiv.org/abs/2010.01179
    and https://arxiv.org/abs/2212.13350.

    Supported datasets:

        - EXP
        - CEXP
        - SR25

    """

    _supported_datasets: List[str] = [
        "exp",
        "cexp",
        "sr25",
    ]

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self):,}, name={self.name!r})"

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got {type(name)}")
        name = name.lower()
        if name in ["exp", "cexp"]:
            self.download = self._download_exp
            self._process_data_list = self._process_data_list_exp
            self._raw_file_names = ["GRAPHSAT.txt"]
        elif name == "sr25":
            self.download = self._download_sr25
            self._process_data_list = self._process_data_list_sr25
            self._raw_file_names = ["sr251256.g6"]
        else:
            raise ValueError(f"Unrecognized dataset name {name!r}, available "
                             f"options are: {self._supported_datasets}")
        self._name = name

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, self.name,
                        "processed")

    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        data_list = self._process_data_list()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print(f"{self.processed_paths[0]=}")

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _download_exp(self):
        filename = self.raw_file_names[0]
        data_path = LOCAL_DATA_DIR / "Abboud2020" / self.name.upper() / filename
        shutil.copyfile(data_path, self.raw_paths[0])

    def _process_data_list_exp(self):
        data_list = []
        with open(self.raw_paths[0]) as f:
            # First line is an integer indicating the total number of graphs
            num_graphs = int(f.readline().rstrip())
            for _ in trange(num_graphs):
                # First line of each block: num_nodes, graph_label
                num_nodes, label = map(int, f.readline().rstrip().split(" "))
                adj = np.zeros((num_nodes, num_nodes))
                x = np.zeros((num_nodes, 1), dtype=int)

                for src, line in zip(range(num_nodes), f):
                    values = list(map(int, line.rstrip().split(" ")))
                    x[src] = values[0]

                    for dst in values[2:]:
                        adj[src, dst] = 1
                edge_index = np.vstack(np.nonzero(adj))

                data = Data(x=torch.LongTensor(x),
                            edge_index=torch.LongTensor(edge_index),
                            y=torch.LongTensor([label]))
                data_list.append(data)

        return data_list

    def _download_sr25(self):
        url = "https://raw.githubusercontent.com/XiaoxinHe/Graph-MLPMixer/48cd68f9e92a7ecbf15aea0baf22f6f338b2030e/dataset/sr25/raw/sr251256.g6"
        download_url(url, self.raw_dir)

    def _process_data_list_sr25(self):
        data_list = []
        for i, g in enumerate(nx.read_graph6(self.raw_paths[0])):
            adj_coo = nx.to_scipy_sparse_array(g).tocoo()
            x = torch.ones(g.size(), 1)
            edge_index = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
            y = torch.LongTensor([i])
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        return data_list


if __name__ == "__main__":
    dataset = SyntheticWL("test_swl", "exp")
    print(f"{dataset}\n  {dataset.data}")

    dataset = SyntheticWL("test_swl", "cexp")
    print(f"{dataset}\n  {dataset.data}")

    dataset = SyntheticWL("test_swl", "sr25")
    print(f"{dataset}\n  {dataset.data}")

    shutil.rmtree("test_swl")
