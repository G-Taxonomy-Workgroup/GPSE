import itertools

import networkx as nx
import torch
from graphgym.transform import cycle_counts


def test_hamcycle_object():
    path = [10, 4, 1, 6, 2]
    hamcycle = cycle_counts.HamiltonianCycle(path)

    assert hamcycle.reduced_repr == (1, 4, 10, 2, 6)
    assert hamcycle.data == [10, 4, 1, 6, 2]
    assert str(hamcycle) == "(1, 4, 10, 2, 6)"
    assert repr(hamcycle) == "HamiltonianCycle(1, 4, 10, 2, 6)"
    assert len(hamcycle) == 5


def test_get_all_k_hamcycles():
    # A graph with a triangle (0, 1, 2) attached to a rectangle (3, 4, 5, 6)
    edge_list = [
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 6),
        (4, 5),
        (5, 6),
    ]
    num_nodes = len(set(itertools.chain.from_iterable(edge_list)))

    g = nx.from_edgelist(edge_list)
    triangles_around_2 = cycle_counts.dfs_k_hamcycles(g, 3, 2)
    assert len(triangles_around_2) == 1
    assert list(triangles_around_2)[0].reduced_repr == (0, 1, 2)

    triangles_around_3 = cycle_counts.dfs_k_hamcycles(g, 3, 3)
    assert len(triangles_around_3) == 0

    rectangles_around_6 = cycle_counts.dfs_k_hamcycles(g, 4, 6)
    assert len(rectangles_around_6) == 1
    assert list(rectangles_around_6)[0].reduced_repr == (3, 4, 5, 6)

    # NOTE: edge_index is not symmetric here, but get_all_k_hamcycles treat
    # any edge_index as undirected, so this is fine.
    edge_index = torch.LongTensor(edge_list).T

    all_triangles = cycle_counts.get_all_k_hamcycles(edge_index, num_nodes, 3)
    assert len(all_triangles) == 1

    all_rectangles = cycle_counts.get_all_k_hamcycles(edge_index, num_nodes, 4)
    assert len(all_rectangles) == 1

    all_pentagons = cycle_counts.get_all_k_hamcycles(edge_index, num_nodes, 5)
    assert len(all_pentagons) == 0

    all_edges = cycle_counts.get_all_k_hamcycles(edge_index, num_nodes, 2)
    assert len(all_edges) == len(edge_list)


def test_get_all_uptok_hamcycles():
    # A graph with a triangle (0, 1, 2) glued to a rectangle (1, 2, 3, 4)
    # along (1, 2)
    edge_list = [
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 4),
    ]

    # 2-cycles are edges adjacency to the seed node
    g = nx.from_edgelist(edge_list)
    res = cycle_counts.dfs_k_hamcycles(g, 2, 2, exact=False)
    res_list = [i.reduced_repr for i in res]
    assert len(res) == 3
    assert (0, 2) in res_list
    assert (1, 2) in res_list
    assert (2, 3) in res_list

    # 2-cycles + 3-cycles
    g = nx.from_edgelist(edge_list)
    res = cycle_counts.dfs_k_hamcycles(g, 3, 2, exact=False)
    res_list = [i.reduced_repr for i in res]
    assert len(res) == 4
    assert (0, 1, 2) in [i.reduced_repr for i in res]

    # 2-cycles + 3-cycles + 4-cycles
    g = nx.from_edgelist(edge_list)
    res = cycle_counts.dfs_k_hamcycles(g, 4, 2, exact=False)
    res_list = [i.reduced_repr for i in res]
    assert len(res) == 5
    assert (1, 2, 3, 4) in [i.reduced_repr for i in res]

    # 2-cycles + 3-cycles + 4-cycles + ... + 9-cycles
    g = nx.from_edgelist(edge_list)
    res = cycle_counts.dfs_k_hamcycles(g, 9, 2, exact=False)
    res_list = [i.reduced_repr for i in res]
    assert len(res) == 6
    assert (0, 1, 4, 3, 2) in [i.reduced_repr for i in res]
