import itertools
from copy import deepcopy

import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset

ADJ_DICT = {'sq_tri': ([[0, 1, 1, 0, 0, 0],
                        [1, 0, 0, 1, 0, 0],
                        [1, 0, 0, 1, 1, 0],
                        [0, 1, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 1, 1, 0]],
                       [[0, 1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [1, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 1, 1],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 1, 0]]
                       ),
            'pent_hex': ([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]],
                         [[0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                          [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 1, 0, 0, 1, 0]]
                         )}


def generate_adj(name):
    adj_1, adj_2 = ADJ_DICT[name]

    adj_1 = torch.tensor(adj_1)
    ei_1 = adj_1.nonzero().t()
    data_1 = Data(edge_index=ei_1)
    data_1.x = [[1], [1], [1], [1], [1], [1]]
    data_1.y = 0

    adj_2 = torch.tensor(adj_2)
    ei_2 = adj_2.nonzero().t()
    data_2 = Data(edge_index=ei_2)
    data_2.x = [[1], [1], [1], [1], [1], [1]]
    data_2.y = 1
    return data_1, data_2


class ToyWLDataset(InMemoryDataset):
    def __init__(self, root, name, num_reps=20, transform=None):
        data_1, data_2 = generate_adj(name)
        data_list = [deepcopy(data_1) for _ in range(num_reps)], [deepcopy(data_2) for _ in range(num_reps)]
        data_list = list(itertools.chain(*data_list))
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
