import logging
from typing import List, Optional, Set, Tuple

import networkx as nx
import torch
from torch import Tensor


class HamiltonianCycle:
    """Hamiltonian cycle object for that considers invariances of cycles.

    The main goal of this object is to reduce a set of Hamiltonian cycles, each
    in the form of a list of unique node indices, into a unique set of
    Hamiltonian cycles. In particular, there are two types of invariances to be
    considered:

        1. Shift invariance. For example, [0, 1, 2, 3] is considered the same
           as [1, 2, 3, 0].
        2. Reflection invariance. For example, [0, 1, 2, 3] is considered the
           the same as [0, 3, 2, 1].

    To efficiently deal with these two invariances, the :obj:`HamiltonianCycle`
    object stores the path list in a reduced format that follows the following
    two conventions:

        1. The first node index must be the smallest among all indices in the
           path list. If not, we apply a shift operation so that the path list
           start with the smallest node index.
        2. The second node index must be no smaller than the last node index.
           If not, we apply a reflection operation so that the second node
           index in the path list is no smaller than the last node index.

    Example:

        >>> path_list = [10, 4, 1, 6, 2]
        >>> print(HamiltonianCycle(path_list))
        (1, 4, 10, 2, 6)

    """

    def __init__(self, path: List[int]):
        self.data = path

    @property
    def reduced_repr(self) -> Tuple[int]:
        return self._reduced_repr

    @property
    def data(self) -> List[int]:
        return self._data

    @data.setter
    def data(self, val: List[int]):
        if not isinstance(val, list):
            raise TypeError(f"path must be a list of integers, got {type(val)}")
        elif len(set(val)) != len(val):
            raise ValueError(f"path must contain unique elements, got {val}")
        elif len(val) < 2:
            raise ValueError(f"path must be at least of size two, got {val}")
        else:
            self._data = val
            self._reduced_repr = self._get_reduced_repr(val)

    @staticmethod
    def _get_reduced_repr(path: List[int]) -> Tuple[int]:
        min_val = min(path)
        min_val_idx = path.index(min_val)

        path = path[min_val_idx:] + path[:min_val_idx]

        if path[-1] < path[1]:
            path = path[:1] + path[-1:0:-1]

        return tuple(path)

    def __len__(self) -> int:
        return len(self._reduced_repr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.__str__()}"

    def __str__(self) -> str:
        return str(self._reduced_repr)

    def __hash__(self) -> int:
        return hash(self._reduced_repr)

    def __eq__(self, other) -> bool:
        if not isinstance(other, HamiltonianCycle):
            raise TypeError("A HamiltonianCycle object can only be compared "
                            "against another HamiltonianCycle object, got "
                            f"{type(other)}")
        return self._reduced_repr == other._reduced_repr

    def __ne__(self, other) -> bool:
        return not self._reduced_repr.__eq__(other._reduced_repr)


def get_all_k_hamcycles(
    edge_index: Tensor,
    num_nodes: int,
    k: int,
    exact: bool = True,
) -> Set[HamiltonianCycle]:
    """Get unique length k Hamiltonian cycles in the graph.

    Args:
        edge_index: COO representation of the adjacency matrix.
        num_nodes: Total number of nodes in the graph.
        k: Target length of the Hamitonian cycles.
        exact: If set to :obj:`True`, then only return Hamiltonian cycles
            *exactly* of length :obj:`k`. Otherwise, return all Hamiltonian
            cycles round the seed node up to, and including, length :obj:`k`.
            Note that computaiton complexities are exactly the same regardless
            of whether :obj:`exact` is set to :obj:`True` or :obj:`False`.

    """
    # NOTE: force graph to be undirected.
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.detach().clone().cpu().T.numpy())

    all_k_hamcycles = set()
    for vi in range(num_nodes):
        k_hamcycles_around_vi = dfs_k_hamcycles(g, k, vi, exact=exact)
        all_k_hamcycles.update(k_hamcycles_around_vi)

    return all_k_hamcycles


def dfs_k_hamcycles(
    g: nx.Graph,
    k: int,
    seed: int,
    *,
    exact: bool = True,
    _cur_depth: int = 0,
    _cur_path: Optional[List[int]] = None,
    _paths: Optional[Set[HamiltonianCycle]] = None,
) -> Set[HamiltonianCycle]:
    """DFS all Hamiltonian cycles up to length k starting from the seed node.

    Args:
        g: Input graph (node id are assumed to be of type integer, representing
            their corresponding node indices).
        k: Target length of the Hamitonian cycles.
        seed: Seed node id.
        exact: If set to :obj:`True`, then only return Hamiltonian cycles
            *exactly* of length :obj:`k`. Otherwise, return all Hamiltonian
            cycles round the seed node up to, and including, length :obj:`k`.
            Note that computaiton complexities are exactly the same regardless
            of whether :obj:`exact` is set to :obj:`True` or :obj:`False`.

    Returns:
        Set[HamiltonianCycle]: A set of unique Hamiltonian cycles of length k
            around the seed node.

    """
    if (not isinstance(k, int)) or (k < 2):
        raise ValueError(f"k (target depth) must be an int > 1, got {k=!r}")

    _cur_path = _cur_path if _cur_path is not None else []
    _paths = _paths if _paths is not None else set()
    logging.debug(f"{_paths=}, {_cur_depth=}, {_cur_path=}, {seed=}")

    if _cur_path and (seed == _cur_path[0]):
        if (_cur_depth == k) or (not exact):
            _paths.add(HamiltonianCycle(_cur_path))

    elif (_cur_depth < k) and (seed not in _cur_path):
        next_depth = _cur_depth + 1
        next_path = _cur_path + [seed]

        for next_seed in g[seed]:
            dfs_k_hamcycles(g, k, next_seed, exact=exact, _cur_depth=next_depth,
                            _cur_path=next_path, _paths=_paths)

    return _paths


def count_cycles(k_list: List[int], data):
    """Count all cycles of length exactly k for k provided in the list."""
    if not isinstance(k_list, list) or not k_list:
        raise ValueError("k_list must be a non-empty list of integers, "
                         f"got {k_list=!r}")

    hamcycles = get_all_k_hamcycles(data.edge_index, data.num_nodes,
                                    max(k_list), False)

    count_dict = {k: 0 for k in k_list}
    for hc in hamcycles:
        if (size := len(hc)) in count_dict:
            count_dict[size] += 1
    cycle_counts = [count_dict[k] for k in k_list]

    return torch.FloatTensor(cycle_counts).unsqueeze(0)
