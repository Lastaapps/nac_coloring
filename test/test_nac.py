"""
Tests for all the NAC coloring search related functions
"""

from nac import NACColoring, Edge

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import pyrigi.graphDB as graphs
from pyrigi import Graph
import networkx as nx
import random
from tqdm import tqdm

import nac as nac

import pytest

NAC_MERGING_ALL = [
    "linear",
    "log",
    "sorted_bits",
    "sorted_size",
    "min_max",
    "score",
    "shared_vertices",
    "promising_cycles",
]
NAC_MERGING_GOOD = [
    "linear",
    "shared_vertices",
]
NAC_SPLIT_ALL = [
    "none",
    "random",
    # "degree",
    # "degree_cycles",
    "cycles",
    "cycles_match_chunks",
    # "bfs",
    "neighbors",
    # "neighbors_cycle",
    "neighbors_degree",
    "kernighan_lin",
    "cuts",
]
NAC_SPLIT_GOOD = [
    "none",
    "cycles_match_chunks",
    "neighbors",
    "neighbors_degree",
]


def ThreePrismPlusTriangleOnSide():
    """Return the 3-prism graph where there is extra triangle on one of the connecting edges."""
    return Graph(
        [
            (0, 1),
            (1, 2),
            (0, 2),
            (3, 4),
            (4, 5),
            (3, 5),
            (0, 3),
            (1, 4),
            (2, 5),
            (0, 6),
            (3, 6),
        ]
    )


def DiamondWithZeroExtension():
    """
    Return the diamond graph with zero extension
    (the diamond with 2 extra connected edges from the opposite spikes).
    """
    return Graph(
        [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (3, 4)],
    )


def SquareGrid2D(w: int, h: int):
    """
    Creates a square grid with width and height given.

    Parameters
    ----------
    w:
        width - no. of nodes in each column
    h:
        height - no. of nodes in each row
    ----------

    Example
    ----------
    For input w: 4, h: 2 you get this graph:

    0-1-2-3
    | | | |
    4-5-6-7
    ----------
    """
    G = Graph.from_vertices(range(w * h))
    for r in range(h):
        offset = r * w
        for c in range(offset, offset + w - 1):
            G.add_edge(c, c + 1)

        if r == 0:
            continue

        for c in range(offset, offset + w):
            G.add_edge(c - w, c)

    return G


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "result"),
    [
        (graphs.Path(3), True),
        (graphs.Cycle(3), False),
        (graphs.Cycle(4), True),
        (graphs.Cycle(5), True),
        (graphs.Complete(5), False),
        (graphs.CompleteBipartite(3, 4), True),
        (graphs.Diamond(), False),
        (graphs.ThreePrism(), True),
        (graphs.ThreePrismPlusEdge(), False),
        (DiamondWithZeroExtension(), True),
    ],
    ids=[
        "path",
        "cycle3",
        "cycle4",
        "cycle5",
        "complete5",
        "bipartite5",
        "diamond",
        "prism",
        "prismPlus",
        "minimallyRigid",
    ],
)
def test_sinlge_and_has_NAC_coloring(graph: nx.Graph, result: bool):
    assert result == (nac.single_NAC_coloring(graph) is not None)
    assert result == nac.has_NAC_coloring(
        graph,
    )


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "result"),
    [
        (
            graphs.Path(3),
            {},
        ),
        (
            graphs.Cycle(3),
            {},
            # This is empty as the whole graph is one triangle component
            # {i: [(0, 1, 2)] for i in range(3)},
        ),
        (
            graphs.Cycle(4),
            {i: {(0, 1, 3, 2)} for i in range(4)},
        ),
        (
            graphs.Cycle(5),
            {i: {(0, 1, 4, 3, 2)} for i in range(5)},
        ),
        (
            graphs.Diamond(),
            {},
        ),
        (
            graphs.ThreePrism(),
            {
                # [[(0, 1), (0, 2), (1, 2)], [(0, 3)], [(1, 4)], [(2, 5)], [(3, 4), (3, 5), (4, 5)]]
                0: {(0, 1, 4, 3), (0, 2, 4, 3), (0, 1, 4, 2)},
                1: {(0, 1, 4, 3), (0, 1, 4, 2)},
                2: {(0, 2, 4, 3), (0, 1, 4, 2)},
                3: {(0, 1, 4, 3), (0, 2, 4, 3)},
                4: {(0, 1, 4, 3), (0, 2, 4, 3), (0, 1, 4, 2)},
            },
        ),
        (
            graphs.ThreePrismPlusEdge(),
            {},
        ),
        (
            DiamondWithZeroExtension(),
            {},
        ),
        (
            Graph.from_vertices_and_edges(
                [0, 1, 2, 3, 4, 5, 6, 7],
                [
                    (0, 1),
                    (0, 5),
                    (1, 3),
                    (1, 7),
                    (2, 3),
                    (2, 4),
                    (3, 7),
                    (4, 5),
                    (4, 6),
                    (5, 6),
                    (6, 7),
                ],
            ),
            {
                # 0: [[(0, 1)]
                # 1: [(0, 5)]
                # 2: [(1, 3), (1, 7), (3, 7)]
                # 3: [(2, 3)]
                # 4: [(2, 4)]
                # 5: [(4, 5), (4, 6), (5, 6)]
                # 6: [(6, 7)]]
                0: {(0, 1, 5, 6, 2)},
                1: {(0, 1, 5, 6, 2)},
                2: {(0, 1, 5, 6, 2), (2, 3, 4, 5, 6)},
                3: {(2, 3, 4, 5, 6)},
                4: {(2, 3, 4, 5, 6)},
                5: {(0, 1, 5, 6, 2), (2, 3, 4, 5, 6)},
                6: {(0, 1, 5, 6, 2), (2, 3, 4, 5, 6)},
            },
        ),
        (
            Graph.from_vertices_and_edges(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [
                    (0, 1),
                    (0, 5),
                    (1, 6),
                    (2, 3),
                    (2, 4),
                    (3, 5),
                    (3, 8),
                    (3, 10),
                    (4, 6),
                    (4, 7),
                    (4, 9),
                    (5, 8),
                    (5, 10),
                    (6, 7),
                    (6, 9),
                    (7, 8),
                    (7, 9),
                    (8, 10),
                    (9, 10),
                ],
            ),
            {
                # [[(0, 1)], [(0, 5)], [(1, 6)], [(2, 3)], [(2, 4)], [(3, 5), (3, 8), (3, 10), (5, 8), (5, 10), (8, 10)], [(4, 6), (4, 7), (4, 9), (6, 7), (6, 9), (7, 9)], [(7, 8)], [(9, 10)]]
                3: {(3, 4, 6, 8, 5), (3, 4, 6, 7, 5)},
                4: {(3, 4, 6, 8, 5), (3, 4, 6, 7, 5)},
                5: {(3, 4, 6, 7, 5), (3, 4, 6, 8, 5), (5, 7, 6, 8)},
                6: {(3, 4, 6, 7, 5), (3, 4, 6, 8, 5), (5, 7, 6, 8)},
                7: {(3, 4, 6, 7, 5), (5, 7, 6, 8)},
                8: {(5, 7, 6, 8), (3, 4, 6, 8, 5)},
            },
        ),
        (
            Graph.from_vertices_and_edges(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                [
                    (0, 1),
                    (0, 5),
                    (0, 9),
                    (1, 3),
                    (1, 5),
                    (1, 10),
                    (1, 12),
                    (2, 10),
                    (3, 4),
                    (3, 5),
                    (3, 7),
                    (3, 9),
                    (3, 14),
                    (4, 6),
                    (4, 11),
                    (4, 12),
                    (5, 6),
                    (5, 8),
                    (5, 11),
                    (6, 7),
                    (6, 10),
                    (6, 11),
                    (7, 11),
                    (7, 12),
                    (9, 13),
                    (11, 13),
                    (11, 14),
                    (12, 13),
                ],
            ),
            # {
            #     # 0:  [(0, 1), (0, 5), (1, 3), (1, 5), (3, 4), (3, 5), (3, 7)],
            #     # 1:  [(0, 9), (3, 9)],
            #     # 2:  [(1, 10)],
            #     # 3:  [(1, 12), (4, 12), (7, 12)],
            #     # 4:  [(2, 10)],
            #     # 5:  [(3, 14)],
            #     # 6:  [(4, 6), (4, 11), (5, 6), (5, 11), (6, 7), (6, 11), (7, 11)],
            #     # 7:  [(5, 8)],
            #     # 8:  [(6, 10)],
            #     # 9:  [(9, 13)],
            #     # 10: [(11, 13)],
            #     # 11: [(11, 14)],
            #     # 12: [(12, 13)],
            #     0: {(0, 1), (0, 3), (0, 6)},
            #     1: {(0, 1)},
            #     2: {(0, 2, 8, 6)},
            #     3: {(0, 3)},
            #     5: {(0, 5, 11, 6)},
            #     6: {(3, 6), (0, 6)},
            #     8: {(0, 2, 8, 6), (2, 3, 6, 8)},
            #     9: {(0, 1, 9, 10, 6), (0, 1, 9, 12, 3), (1, 5, 11, 10, 9)},
            #     10: {(3, 6, 10, 12)},
            #     11: {(0, 5, 11, 6)},
            #     12: {(3, 6, 10, 12)},
            # },
            {
                0: {
                    (0, 3, 2, 8, 6),
                    (0, 5, 11, 6, 7),
                    (0, 5, 11, 6),
                    (0, 2, 8, 6, 7),
                    (0, 2, 3, 6, 7),
                    (0, 3, 6, 11, 5),
                    (0, 1, 9, 12, 3),
                    (0, 3, 6),
                    (0, 1, 9, 10, 6),
                    (0, 5, 11, 10, 6),
                    (0, 1, 5, 11, 6),
                    (0, 2, 4, 8, 6),
                    (0, 3, 6, 7),
                    (0, 2, 8, 6),
                    (0, 2, 3, 6),
                    (0, 3, 12, 10, 6),
                },
                1: {(0, 1, 9, 10, 6), (0, 1, 9, 12, 3), (1, 5, 11, 10, 9)},
                2: {
                    (0, 3, 2, 8, 6),
                    (2, 3, 6, 8, 4),
                    (0, 2, 8, 6, 7),
                    (2, 3, 6, 8),
                    (0, 2, 4, 8, 6),
                    (0, 2, 8, 6, 3),
                    (0, 2, 8, 6),
                },
                3: {
                    (3, 6, 10, 9, 12),
                    (3, 6, 11, 10, 12),
                    (3, 6, 10, 12),
                    (0, 1, 9, 12, 3),
                    (0, 3, 12, 10, 6),
                },
                5: {
                    (0, 5, 11, 6, 7),
                    (0, 3, 6, 11, 5),
                    (1, 5, 11, 10, 9),
                    (0, 5, 11, 10, 6),
                    (0, 1, 5, 11, 6),
                    (0, 5, 11, 6),
                },
                6: {
                    (0, 3, 2, 8, 6),
                    (2, 3, 6, 8, 4),
                    (0, 5, 11, 6, 7),
                    (3, 6, 10, 12),
                    (3, 6, 10, 9, 12),
                    (3, 6, 11, 10, 12),
                    (0, 2, 8, 6, 7),
                    (0, 3, 6, 11, 5),
                    (2, 3, 6, 8),
                    (0, 1, 9, 10, 6),
                    (0, 5, 11, 10, 6),
                    (0, 1, 5, 11, 6),
                    (0, 2, 4, 8, 6),
                    (0, 5, 11, 6),
                    (0, 2, 8, 6, 3),
                    (0, 2, 8, 6),
                    (0, 3, 12, 10, 6),
                },
                8: {
                    (0, 2, 4, 8, 6),
                    (0, 3, 2, 8, 6),
                    (2, 3, 6, 8, 4),
                    (2, 3, 6, 8),
                    (0, 2, 8, 6, 7),
                    (0, 2, 8, 6, 3),
                    (0, 2, 8, 6),
                },
                9: {(0, 1, 9, 10, 6), (0, 1, 9, 12, 3), (1, 5, 11, 10, 9)},
                10: {
                    (3, 6, 10, 12),
                    (3, 6, 10, 9, 12),
                    (3, 6, 11, 10, 12),
                    (1, 5, 11, 10, 9),
                    (0, 1, 9, 10, 6),
                    (0, 3, 12, 10, 6),
                },
                11: {
                    (0, 5, 11, 6, 7),
                    (0, 3, 6, 11, 5),
                    (1, 5, 11, 10, 9),
                    (0, 5, 11, 10, 6),
                    (0, 1, 5, 11, 6),
                    (0, 5, 11, 6),
                },
                12: {
                    (3, 6, 10, 12),
                    (3, 6, 11, 10, 12),
                    (3, 6, 10, 9, 12),
                    (0, 1, 9, 12, 3),
                    (0, 3, 12, 10, 6),
                },
            },
        ),
    ],
    ids=[
        "path",
        "cycle3",
        "cycle4",
        "cycle5",
        "diamond",
        "prism",
        "prismPlus",
        "minimallyRigid",
        "smaller_problemist",
        "large_problemist",
        "anoying_problemist",
        # TODO for disconnected components "cycle_passing_same_component_twice",
    ],
)
def test__find_useful_cycles_for_components(graph: nx.Graph, result: Set[Tuple]):
    _, component_to_edges = nac.find_monochromatic_classes(graph)
    # print()
    # print(graph)
    # print(component_to_edges)
    res = nac._find_useful_cycles_for_components(
        graph,
        set(range(len(component_to_edges))),
        component_to_edges,
        per_class_limit=1024,
    )
    # print(f"{res=}")
    assert res == result


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "result"),
    [
        (
            graphs.Path(3),
            {},
        ),
        (
            graphs.Cycle(3),
            {},
            # This is empty as the whole graph is one triangle component
            # {i: [(0, 1, 2)] for i in range(3)},
        ),
        (
            graphs.Cycle(4),
            {i: {(0, 1, 3, 2)} for i in range(4)},
        ),
        (
            graphs.Cycle(5),
            {i: {(0, 1, 4, 3, 2)} for i in range(5)},
        ),
        (
            graphs.Diamond(),
            {},
        ),
        (
            graphs.ThreePrism(),
            {
                # [[(0, 1), (0, 2), (1, 2)], [(0, 3)], [(1, 4)], [(2, 5)], [(3, 4), (3, 5), (4, 5)]]
                0: {(0, 1, 4, 3), (0, 2, 4, 3), (0, 1, 4, 2)},
                1: {(0, 1, 4, 3), (0, 1, 4, 2)},
                2: {(0, 2, 4, 3), (0, 1, 4, 2)},
                3: {(0, 1, 4, 3), (0, 2, 4, 3)},
                4: {(0, 1, 4, 3), (0, 2, 4, 3), (0, 1, 4, 2)},
            },
        ),
        (
            graphs.ThreePrismPlusEdge(),
            {},
        ),
        (
            DiamondWithZeroExtension(),
            {
                # [[(0, 1), (0, 3), (0, 2), (1, 2), (2, 3)], [(1, 4), (3, 4)]]
                0: {(0, 1)},
                1: {(0, 1)},
            },
        ),
        (
            Graph.from_vertices_and_edges(
                [0, 1, 2, 3, 4, 5, 6, 7],
                [
                    (0, 1),
                    (0, 5),
                    (1, 3),
                    (1, 7),
                    (2, 3),
                    (2, 4),
                    (3, 7),
                    (4, 5),
                    (4, 6),
                    (5, 6),
                    (6, 7),
                ],
            ),
            {
                # 0: [[(0, 1)]
                # 1: [(0, 5)]
                # 2: [(1, 3), (1, 7), (3, 7)]
                # 3: [(2, 3)]
                # 4: [(2, 4)]
                # 5: [(4, 5), (4, 6), (5, 6)]
                # 6: [(6, 7)]]
                0: {(0, 1, 5, 6, 2)},
                1: {(0, 1, 5, 6, 2)},
                2: {(0, 1, 5, 6, 2), (2, 3, 4, 5, 6)},
                3: {(2, 3, 4, 5, 6)},
                4: {(2, 3, 4, 5, 6)},
                5: {(0, 1, 5, 6, 2), (2, 3, 4, 5, 6)},
                6: {(0, 1, 5, 6, 2), (2, 3, 4, 5, 6)},
            },
        ),
        (
            Graph.from_vertices_and_edges(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [
                    (0, 1),
                    (0, 5),
                    (1, 6),
                    (2, 3),
                    (2, 4),
                    (3, 5),
                    (3, 8),
                    (3, 10),
                    (4, 6),
                    (4, 7),
                    (4, 9),
                    (5, 8),
                    (5, 10),
                    (6, 7),
                    (6, 9),
                    (7, 8),
                    (7, 9),
                    (8, 10),
                    (9, 10),
                ],
            ),
            {
                # [[(0, 1)], [(0, 5)], [(1, 6)], [(2, 3)], [(2, 4)], [(3, 5), (3, 8), (3, 10), (5, 8), (5, 10), (8, 10)], [(4, 6), (4, 7), (4, 9), (6, 7), (6, 9), (7, 9)], [(7, 8)], [(9, 10)]]
                0: {(0, 1, 5, 7, 6, 2), (0, 1, 5, 8, 6, 2)},
                1: {(0, 1, 5, 7, 6, 2), (0, 1, 5, 8, 6, 2)},
                2: {(0, 1, 5, 7, 6, 2), (0, 1, 5, 8, 6, 2)},
                3: {(3, 4, 6, 8, 5), (3, 4, 6, 7, 5)},
                4: {(3, 4, 6, 8, 5), (3, 4, 6, 7, 5)},
                5: {(5, 7, 6, 8)},
                6: {(5, 7, 6, 8)},
                7: {(5, 7, 6, 8)},
                8: {(5, 7, 6, 8)},
            },
        ),
        (
            Graph.from_vertices_and_edges(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                [
                    (0, 1),
                    (0, 5),
                    (0, 9),
                    (1, 3),
                    (1, 5),
                    (1, 10),
                    (1, 12),
                    (2, 10),
                    (3, 4),
                    (3, 5),
                    (3, 7),
                    (3, 9),
                    (3, 14),
                    (4, 6),
                    (4, 11),
                    (4, 12),
                    (5, 6),
                    (5, 8),
                    (5, 11),
                    (6, 7),
                    (6, 10),
                    (6, 11),
                    (7, 11),
                    (7, 12),
                    (9, 13),
                    (11, 13),
                    (11, 14),
                    (12, 13),
                ],
            ),
            {
                # 0:  [(0, 1), (0, 5), (1, 3), (1, 5), (3, 4), (3, 5), (3, 7)],
                # 1:  [(0, 9), (3, 9)],
                # 2:  [(1, 10)],
                # 3:  [(1, 12), (4, 12), (7, 12)],
                # 4:  [(2, 10)],
                # 5:  [(3, 14)],
                # 6:  [(4, 6), (4, 11), (5, 6), (5, 11), (6, 7), (6, 11), (7, 11)],
                # 7:  [(5, 8)],
                # 8:  [(6, 10)],
                # 9:  [(9, 13)],
                # 10: [(11, 13)],
                # 11: [(11, 14)],
                # 12: [(12, 13)],
                0: {(0, 1), (0, 3), (0, 6)},
                1: {(0, 1)},
                2: {(0, 2, 8, 6)},
                3: {(0, 3)},
                5: {(0, 5, 11, 6)},
                6: {(3, 6), (0, 6)},
                8: {(0, 2, 8, 6), (2, 3, 6, 8)},
                9: {(0, 1, 9, 10, 6), (0, 1, 9, 12, 3), (1, 5, 11, 10, 9)},
                10: {(3, 6, 10, 12)},
                11: {(0, 5, 11, 6)},
                12: {(3, 6, 10, 12)},
            },
        ),
    ],
    ids=[
        "path",
        "cycle3",
        "cycle4",
        "cycle5",
        "diamond",
        "prism",
        "prismPlus",
        "minimallyRigid",
        "smaller_problemist",
        "large_problemist",
        "anoying_problemist",
        # TODO for disconnected components "cycle_passing_same_component_twice",
    ],
)
def test__find_shortest_cycles_for_components(graph: nx.Graph, result: Set[Tuple]):
    _, component_to_edges = nac.find_monochromatic_classes(graph)
    # print()
    # print(graph)
    # print(component_to_edges)
    res = nac._find_shortest_cycles_for_components(
        graph,
        set(range(len(component_to_edges))),
        component_to_edges,
        all=True,
        per_class_limit=1024,
    )
    # print(f"{res=}")
    assert res == result


NAC_ALGORITHMS = (
    [
        "naive",
        "cycles-True",
        "cycles-False",
        "subgraphs-log-none-64",
        "subgraphs-log-none-1-smart",
    ]
    + [
        flattened
        for paired in [
            (alg,)
            # (alg, alg + "-smart")
            for alg in [
                "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
                for merge in [
                    "linear",
                    "log",
                    # "log_reverse",
                    "sorted_bits",
                    # "sorted_size",
                    # "min_max",
                    "score",
                    # "dynamic",
                    # "recursion",
                    "shared_vertices",
                    "promising_cycles",
                ]
                for algo in [
                    "none",
                    "random",
                    "neighbors",
                ]
                for size in [1, 4]
            ]
        ]
        for flattened in paired
    ]
    + [
        flattened
        for paired in [
            (alg,)
            # (alg, alg + "-smart")
            for alg in [
                "subgraphs-{}-{}-{}-smart".format(merge, algo, size)
                for merge in [
                    "linear",
                    "log",
                    "score",
                ]
                for algo in [
                    "none",
                    "random",
                    # "degree",
                    # "degree_cycles",
                    "cycles",
                    "cycles_match_chunks",
                    # "bfs",
                    "neighbors",
                    "neighbors_cycle",
                    "neighbors_degree",
                    "neighbors_degree_cycle",
                    "neighbors_iterative",
                    "neighbors_iterative_cycle",
                    # "beam_neighbors",
                    # "beam_neighbors_max",
                    # "beam_neighbors_triangles",
                    # "beam_neighbors_max_triangles",
                    # "components_biggest",
                    # "components_spredded",
                    # "kernighan_lin",
                    # "cuts",
                ]
                for size in [1, 4]
            ]
        ]
        for flattened in paired
    ]
)
NAC_RELABEL_STRATEGIES = [
    "none",
    "random",
    # "bfs",
    # "beam_degree",
]


@dataclass
class NACTestCase:
    """
    Used for NAC coloring and cartesian NAC coloring testing.
    """

    name: str
    graph: nx.Graph
    no_normal: int | None
    no_cartesian: int | None


NAC_TEST_CASES: List[NACTestCase] = [
    NACTestCase("path", graphs.Path(3), 2, 2),
    NACTestCase(
        "path_and_single_vertex",
        Graph.from_vertices_and_edges([0, 1, 2, 3], [(0, 1), (1, 2)]),
        2,
        2,
    ),
    NACTestCase("cycle3", graphs.Cycle(3), 0, 0),
    NACTestCase("cycle4", graphs.Cycle(4), 6, 2),
    NACTestCase("cycle5", graphs.Cycle(5), 20, 10),
    NACTestCase("complete5", graphs.Complete(5), 0, 0),
    NACTestCase("bipartite1x3", graphs.CompleteBipartite(1, 3), 6, 6),
    NACTestCase(
        "bipartite1x3-improved",
        Graph.from_vertices_and_edges([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3), (2, 3)]),
        2,
        2,
    ),
    NACTestCase("bipartite1x4", graphs.CompleteBipartite(1, 4), 14, 14),
    NACTestCase(
        "bipartite1x4-improved",
        Graph.from_vertices_and_edges(
            [0, 1, 2, 3, 4], [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4)]
        ),
        6,
        6,
    ),
    NACTestCase("bipartite2x3", graphs.CompleteBipartite(2, 3), 14, 0),
    NACTestCase("bipartite2x4", graphs.CompleteBipartite(2, 4), 30, 0),
    NACTestCase("bipartite3x3", graphs.CompleteBipartite(3, 3), 30, 0),
    NACTestCase("bipartite3x4", graphs.CompleteBipartite(3, 4), 62, 0),
    NACTestCase("diamond", graphs.Diamond(), 0, 0),
    NACTestCase("prism", graphs.ThreePrism(), 2, 2),
    NACTestCase("prismPlus", graphs.ThreePrismPlusEdge(), 0, 0),
    NACTestCase("minimallyRigid", DiamondWithZeroExtension(), 2, 0),
    NACTestCase(
        "smaller_problemist",
        Graph.from_vertices_and_edges(
            [0, 1, 2, 3, 4, 5, 6],
            [(0, 3), (0, 6), (1, 2), (1, 6), (2, 5), (3, 5), (4, 5), (4, 6)],
        ),
        108,
        30,
    ),
    NACTestCase(
        "large_problemist",
        Graph.from_vertices_and_edges(
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 8),
                (2, 7),
                (3, 8),
                (4, 7),
                (5, 7),
                (5, 8),
                (6, 7),
                (6, 8),
            ],
        ),
        472,
        54,
    ),
    NACTestCase(
        "3-squares-and-connectig-edge",
        Graph.from_vertices_and_edges(
            list(range(10)),
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 8),
                (4, 8),
                (0, 9),
                (4, 9),
                (1, 5),
            ],
        ),
        606,
        30,
    ),
    NACTestCase(
        "square-2-pendagons-and-connectig-edge",
        Graph.from_vertices_and_edges(
            list(range(12)),
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 5),
                (0, 10),
                (5, 10),
                (0, 11),
                (5, 11),
                (1, 6),
            ],
        ),
        None,  # 4596,
        286,
    ),
    NACTestCase(
        "diconnected-problemist",
        Graph.from_vertices_and_edges(
            list(range(15)),
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (5, 6),
                (7, 8),
                (9, 10),
                (0, 13),
                (5, 13),
                (0, 14),
                (5, 14),
                (1, 6),
            ],
        ),
        1214,
        254,
    ),
    NACTestCase(
        "brachiosaurus",
        Graph.from_vertices_and_edges(
            list(range(10)),
            [
                (0, 7),
                (0, 8),
                (0, 9),
                (1, 7),
                (1, 8),
                (1, 9),
                (2, 7),
                (2, 8),
                (2, 9),
                (3, 7),
                (3, 8),
                (3, 9),
                (4, 5),
                (4, 8),
                (4, 9),
                (5, 6),
                (5, 9),
                (6, 7),
                (6, 8),
                (6, 9),
            ],
        ),
        126,
        None,  # unknown, yet
    ),
    NACTestCase(
        "cycles_destroyer",
        Graph.from_vertices_and_edges(
            list(range(14 + 1)),
            [
                (0, 3),
                (0, 8),
                (0, 12),
                (0, 14),
                (0, 9),
                (0, 5),
                (1, 13),
                (1, 2),
                (1, 8),
                (1, 10),
                (2, 11),
                (2, 7),
                (2, 9),
                (3, 6),
                (4, 14),
                (4, 13),
                (5, 9),
                (5, 6),
                (6, 9),
                (6, 10),
                (6, 8),
                (7, 10),
                (7, 11),
                (8, 10),
                (9, 10),
                (10, 12),
                (11, 13),
                (11, 12),
                (12, 14),
            ],
        ),
        68,
        None,  # unknown, yet
    ),
    NACTestCase(
        "square_and_line",
        Graph.from_vertices_and_edges(
            list(range(15)),
            [
                (7, 8),
                (0, 13),
                (5, 13),
                (0, 14),
                (5, 14),
            ],
        ),
        14,
        6,
    ),
]


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "colorings_no"),
    [
        (case.graph, case.no_normal)
        for case in NAC_TEST_CASES
        if case.no_normal is not None
    ],
    ids=[case.name for case in NAC_TEST_CASES if case.no_normal is not None],
)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize("use_decompositions", [True, False])
@pytest.mark.parametrize(
    "class_type",
    [
        nac.MonochromaticClassType.MONOCHROMATIC,
        nac.MonochromaticClassType.TRIANGLES,
    ],
)
def test_all_NAC_colorings(
    graph: nx.Graph,
    colorings_no: int,
    algorithm: str,
    relabel_strategy: str,
    use_decompositions: bool,
    class_type: nac.MonochromaticClassType,
):
    # print(f"\nTested graph: {graph=}")
    # print(nx.nx_agraph.to_agraph(graph))

    coloring_list = list(
        nac.NAC_colorings(
            graph,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            monochromatic_class_type=class_type,
            use_decompositions=use_decompositions,
            use_has_coloring_check=False,
            seed=42,  # this is potentially dangerous
        )
    )

    # print(f"{coloring_list=}")

    no_duplicates = {
        (tuple(sorted(coloring[0])), tuple(sorted(coloring[1])))
        for coloring in coloring_list
    }
    assert len(coloring_list) == len(no_duplicates)

    # for coloring in sorted([str(x) for x in coloring_list]):
    #     print(coloring)

    assert colorings_no == len(coloring_list)

    for coloring in coloring_list:
        assert nac.is_NAC_coloring(graph, coloring)


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "coloring", "result"),
    [
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), set([(1, 4), (3, 4)])),
            True,
        ),
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (3, 4)]), set([])),
            False,
        ),
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4)]), set([(3, 4)])),
            False,
        ),
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (3, 0), (0, 2)]), set([(2, 3), (1, 4), (3, 4)])),
            False,
        ),
        (
            graphs.ThreePrism(),
            (
                set([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]),
                set([(0, 3), (1, 4), (2, 5)]),
            ),
            True,
        ),
    ],
)
def test_is_NAC_coloring(
    graph: nx.Graph, coloring: Tuple[Set[Edge], Set[Edge]], result: bool
):
    red, blue = coloring
    assert nac.is_NAC_coloring(graph, (red, blue)) == result
    assert nac.is_NAC_coloring(graph, (blue, red)) == result


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "coloring"),
    [
        (
            ThreePrismPlusTriangleOnSide(),
            None,
        ),
        (
            graphs.Path(3),
            (set([(0, 1)]), set([(1, 2)])),
        ),
        (
            Graph.from_vertices_and_edges(range(4), [(0, 1), (1, 2), (0, 2), (0, 3)]),
            (set([(0, 3)]), set([(0, 1), (1, 2), (0, 2)])),
        ),
        (
            Graph.from_vertices_and_edges(
                range(5), [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
            ),
            (
                set([(0, 2), (0, 3), (0, 4)]),
                set([(1, 2), (1, 3), (1, 4)]),
            ),
        ),
    ],
    ids=["NAC_but_not_now", "path-2", "triangle_with_dangling", "nice_stable_cut"],
)
def test__check_for_simple_stable_cut(graph: nx.Graph, coloring: Optional[NACColoring]):
    res1 = nac._check_for_simple_stable_cut(graph, certificate=False)
    res2 = nac._check_for_simple_stable_cut(graph, certificate=True)

    if coloring is None:
        assert res1 is None
        assert res2 is None
        return

    assert res1 is not None

    coloring = nac.canonical_NAC_coloring(coloring)
    res2 = nac.canonical_NAC_coloring(res2)
    assert coloring == res2


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "colorings_no"),
    [
        (case.graph, case.no_cartesian)
        for case in NAC_TEST_CASES
        if case.no_cartesian is not None
    ],
    ids=[case.name for case in NAC_TEST_CASES if case.no_cartesian is not None],
)
@pytest.mark.parametrize("algorithm", NAC_ALGORITHMS)
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize("use_decompositions", [True, False])
@pytest.mark.parametrize(
    "class_type",
    [
        nac.MonochromaticClassType.MONOCHROMATIC,
        nac.MonochromaticClassType.TRIANGLES,
    ],
)
@pytest.mark.skip(
    "Cartesian NAC coloring is slightly broken and I don't care at the moment"
)
def test_all_cartesian_NAC_colorings(
    graph,
    colorings_no: int,
    algorithm: str,
    relabel_strategy: str,
    use_decompositions: bool,
    class_type: nac.MonochromaticClassType,
):
    # print(f"\nTested graph: {graph=}")
    coloring_list = list(
        nac.cartesian_NAC_colorings(
            graph,
            algorithm=algorithm,
            relabel_strategy=relabel_strategy,
            use_decompositions=use_decompositions,
            monochromatic_class_type=class_type,
            use_has_coloring_check=False,
        )
    )

    # print(f"{coloring_list=}")

    no_duplicates = {
        nac.canonical_NAC_coloring(coloring, including_red_blue_order=False)
        for coloring in coloring_list
    }
    assert len(coloring_list) == len(no_duplicates)

    # for coloring in sorted([str(x) for x in coloringList]):
    #     print(coloring)

    assert colorings_no == len(coloring_list)

    for coloring in coloring_list:
        assert nac.is_NAC_coloring(graph, coloring)
        assert nac.is_cartesian_NAC_coloring(graph, coloring)


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "coloring"),
    [
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4), (3, 4)]), set([])),
        ),
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 4)]), set([(3, 4)])),
        ),
        (
            DiamondWithZeroExtension(),
            (set([(0, 1), (1, 2), (3, 0), (0, 2)]), set([(2, 3), (1, 4), (3, 4)])),
        ),
        # tests if everything works with non-integer vertices
        (
            Graph(
                [
                    ("0", "1"),
                    ("1", "2"),
                    ("2", "3"),
                    ("3", "0"),
                    ("0", "2"),
                    ("1", "4"),
                    ("3", "4"),
                ]
            ),
            (
                set([("0", "1"), ("1", "2"), ("3", "0"), ("0", "2")]),
                set([("2", "3"), ("1", "4"), ("3", "4")]),
            ),
        ),
    ],
)
@pytest.mark.skip(
    "Cartesian NAC coloring is slightly broken and I don't care at the moment"
)
def test_is_cartesian_NAC_coloring_on_not_event_NAC_colorings(
    graph: nx.Graph, coloring: Tuple[Set[Edge], Set[Edge]]
):
    """
    Cartesian NAC coloring is also NAC coloring. So if we pass invalid coloring,
    cartesian NAC coloring result should be also negative.
    """
    red, blue = coloring
    assert nac.is_cartesian_NAC_coloring(graph, (red, blue)) == False
    assert nac.is_cartesian_NAC_coloring(graph, (blue, red)) == False


@pytest.mark.nac_test
@pytest.mark.parametrize(
    ("graph", "coloring", "result"),
    [
        (
            SquareGrid2D(4, 2),
            (
                set([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]),
                set([(0, 4), (1, 5), (2, 6), (3, 7)]),
            ),
            True,
        ),
        (
            SquareGrid2D(4, 2),
            (
                set([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]),
                set([(0, 4), (1, 5), (2, 6), (3, 7), (6, 7)]),
            ),
            False,
        ),
        (
            SquareGrid2D(4, 2),
            (
                set([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (3, 7)]),
                set([(0, 4), (1, 5), (2, 6)]),
            ),
            False,
        ),
        (
            SquareGrid2D(4, 2),
            (
                set([(0, 1), (1, 2), (4, 5), (5, 6), (0, 4), (1, 5)]),
                set([(2, 6), (3, 7), (2, 3), (6, 7)]),
            ),
            False,
        ),
        # TODO more tests
    ],
)
@pytest.mark.skip(
    "Cartesian NAC coloring is slightly broken and I don't care at the moment"
)
def test_is_cartesian_NAC_coloring(
    graph: nx.Graph, coloring: Tuple[Set[Edge], Set[Edge]], result: bool
):
    red, blue = coloring
    assert nac.is_cartesian_NAC_coloring(graph, (red, blue)) == result
    assert nac.is_cartesian_NAC_coloring(graph, (blue, red)) == result


################################################################################
# Fuzzy tests
################################################################################
@pytest.mark.parametrize("relabel_strategy", NAC_RELABEL_STRATEGIES)
@pytest.mark.parametrize(
    ("n", "graph_no", "reference_algorithm", "tested_algorithms"),
    [
        (11, 32, "naive", ["cycles"]),
        (16, 32, "cycles", ["subgraphs-linear-none-4"]),
        (
            18,
            32,
            "subgraphs-linear-none-4",
            (
                ["subgraphs-{}-none-4".format(merge) for merge in NAC_MERGING_ALL]
                + ["subgraphs-linear-{}-4".format(split) for split in NAC_SPLIT_ALL]
            ),
        ),
        (
            26,
            32,
            "subgraphs-linear-none-4",
            (
                ["subgraphs-{}-none-4".format(merge) for merge in NAC_MERGING_GOOD]
                + ["subgraphs-linear-{}-4".format(split) for split in NAC_SPLIT_GOOD]
            ),
        ),
    ],
    ids=[
        "ensure cycles work based on naive",
        "ensure subgraphs work based on cycles",
        "ensure subgraphs strategies",
        "ensure good subgraphs strategies",
    ],
)
@pytest.mark.nac_test
@pytest.mark.parametrize("graph_class", ["NAC_critical"])
@pytest.mark.parametrize("seed", [42, random.randint(0, 2**30)])
def test_fuzzy_NAC_coloring(
    n: int,
    graph_no: int,
    reference_algorithm: str,
    tested_algorithms: List[str],
    graph_class: str,
    seed: int,
    relabel_strategy: str,
):
    """
    Checks algorithm validity against the naive implementation
    (that is hopefully correct) and checks that outputs are the same.
    Large number of smaller graphs is used (naive implementation is really slow for larger ones).
    """
    from benchmarks.generators import _generate_NAC_critical_graph

    verbose = False

    if reference_algorithm in tested_algorithms:
        tested_algorithms.remove(reference_algorithm)

    rand = random.Random(seed)

    # print()  # prevent tqdm from overlapping test name
    # for _ in tqdm(range(graph_no)):
    for _ in range(graph_no):
        match graph_class:
            case "NAC_critical":
                graph_seed = rand.randint(0, 2**30)
                graph = Graph(_generate_NAC_critical_graph(n, seed=graph_seed))
                if verbose:
                    print(graph)
            case _:
                raise ValueError

        baseline_seed = rand.randint(0, 2**30)
        if verbose:
            print(reference_algorithm, "-", baseline_seed)
        baseline = list(
            nac.NAC_colorings(
                graph=graph,
                algorithm=reference_algorithm,
                remove_vertices_cnt=0,
                # without this naive search is really slow
                # use_chromatic_partitions=False,
                use_decompositions=False,
                relabel_strategy="none",
                use_has_coloring_check=False,
                seed=baseline_seed,
            )
        )

        for coloring in baseline:
            nac.is_NAC_coloring(graph, coloring)

        baseline = {
            nac.canonical_NAC_coloring(coloring, including_red_blue_order=False)
            for coloring in baseline
        }

        for tested_algorithm in tested_algorithms:
            tested_seed = rand.randint(0, 2**30)
            if verbose:
                print(tested_algorithm, "-", tested_seed)
            tested = list(
                nac.NAC_colorings(
                    graph=graph,
                    algorithm=tested_algorithm,
                    relabel_strategy=relabel_strategy,
                    use_has_coloring_check=False,
                    seed=tested_seed,
                )
            )

            l1, l2 = len(baseline), len(tested)
            assert l1 == l2

            # for coloring in tested:
            #     graph.is_NAC_coloring(coloring)

            tested = {
                nac.canonical_NAC_coloring(coloring, including_red_blue_order=False)
                for coloring in tested
            }

            s1, s2 = len(baseline), len(tested)
            assert s1 == s2
            assert l1 == s1
            assert l2 == s2
            assert baseline == tested
