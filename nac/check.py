"""
The module checks if the coloring given is a NAC coloring.
The main entry point is _is_NAC_coloring_impl
"""

from collections import defaultdict
from typing import *

import networkx as nx
import numpy as np

from nac.data_type import NACColoring, Edge

from nac.existence import check_NAC_constrains
from nac.data_type import NiceGraph

# how many times was the routine called
_NAC_CHECK_IS_NAC_COLORING = 0
_NAC_CHECK_CYCLE_MASK = 0
_NAC_MERGE = 0
_NAC_MERGE_NO_COMMON_VERTEX = 0


def _NAC_check_called_reset():
    global _NAC_CHECK_IS_NAC_COLORING
    global _NAC_CHECK_CYCLE_MASK
    global _NAC_MERGE
    global _NAC_MERGE_NO_COMMON_VERTEX
    _NAC_CHECK_IS_NAC_COLORING = 0
    _NAC_CHECK_CYCLE_MASK = 0
    _NAC_MERGE = 0
    _NAC_MERGE_NO_COMMON_VERTEX = 0


def NAC_check_called() -> Tuple[int, int, int, int]:
    return (
        _NAC_CHECK_IS_NAC_COLORING,
        _NAC_CHECK_CYCLE_MASK,
        _NAC_MERGE,
        _NAC_MERGE_NO_COMMON_VERTEX,
    )


def _check_for_almost_red_cycles(
    G: nx.Graph,
    red_edges: Iterable[Edge],
    blue_edges: Iterable[Edge],
) -> bool:
    """
    Checks if there is an almost cycle in the graph given with the given coloring.
    Does not check if the coloring is surjective.
    Returns true if the coloring has no such cycles..
    """
    G.clear_edges()
    G.add_edges_from(red_edges)

    component_mapping: Dict[int, int] = {}
    vertices: Iterable[int]
    for i, vertices in enumerate(nx.components.connected_components(G)):
        for v in vertices:
            component_mapping[v] = i

    for e1, e2 in blue_edges:
        if component_mapping[e1] == component_mapping[e2]:
            return False
    return True


def _is_NAC_coloring_impl(
    graph: nx.Graph,
    coloring: NACColoring,
) -> bool:
    """
    Check if the coloring given is a NAC coloring.
    The algorithm checks if all the edges are in the same component.

    This is an internal implementation, so some properties like injectivity
    are not checked for performance reasons - we only search for the cycles.

    Parameters:
    ----------
        coloring: the coloring to check if it is a NAC coloring.
        allow_non_surjective: if True, allows the coloring to be non-surjective.
            This can be useful for checking subgraphs - the can use only one color.
    ----------
    """
    global _NAC_CHECK_IS_NAC_COLORING
    _NAC_CHECK_IS_NAC_COLORING += 1

    red, blue = coloring

    # Workaround
    G = NiceGraph()
    G.add_nodes_from(graph.nodes)

    return _check_for_almost_red_cycles(G, red, blue) and _check_for_almost_red_cycles(
        G, blue, red
    )


def is_NAC_coloring(
    graph: nx.Graph,
    coloring: NACColoring | Dict[str, Collection[Edge]],
) -> bool:
    """
    Check if the coloring given is a NAC coloring.
    The algorithm checks if all the edges are in the same component.

    Parameters:
    ----------
        coloring: the coloring to check if it is a NAC coloring.
    ----------
    """
    red: Collection[Edge]
    blue: Collection[Edge]

    if type(coloring) == dict:
        red, blue = coloring["red"], coloring["blue"]
    else:
        red, blue = coloring
    assert type(red) == type(blue)

    if not check_NAC_constrains(graph):
        return False

    # Both colors have to be used
    if len(red) == 0 or len(blue) == 0:
        return False

    if len(red) + len(blue) != len(graph.edges):
        return False

    if type(red) == set and len(red.intersection(blue)) != 0:
        return False
    else:
        for e in red:
            if e in blue:
                return False

    return _is_NAC_coloring_impl(graph, (red, blue))


def _is_cartesian_NAC_coloring_impl(
    graph: nx.Graph,
    coloring: NACColoring,
) -> bool:
    """
    Check if the coloring given is a cartesian NAC coloring.

    Parameters:
    ----------
        coloring: the coloring to check if it is a cartesian NAC coloring.
    ----------

    Pseudocode:
        find red and blue components
        for each vertex get ids of it's components
        iterate over all the vertices with 2+ components
        mark neighboring components
        if two components are already neighboring, return false
        if no problem is found, return true
    """

    G = NiceGraph()
    G.add_nodes_from(graph.nodes)

    red, blue = coloring

    G.add_edges_from(red)

    if type(list(G.nodes)[0]) in [int, np.int8, np.int16, np.int32, np.int64]:
        max_value = max(nx.nodes(G)) + 1
        # prevents an attack where large node ids would be passed
        # and system would run out of memory
        if max_value < 4096:
            comp_ids = np.full(max_value, -1)
        else:
            comp_ids = defaultdict(lambda: -1)
    else:
        comp_ids = defaultdict(lambda: -1)

    id: int = -1
    for id, red_comp in enumerate(nx.components.connected_components(G)):
        for v in red_comp:
            comp_ids[v] = id
    id += 1

    G.clear_edges()
    G.add_edges_from(blue)

    neighbors: List[List[int]] = [[] for _ in range(id)]

    for blue_id, blue_comp in enumerate(nx.components.connected_components(G)):
        for v in blue_comp:
            red_id: int = comp_ids[v]
            if red_id == -1:
                continue

            if blue_id in neighbors[red_id]:
                return False
            neighbors[red_id].append(blue_id)
    return True


def is_cartesian_NAC_coloring(
    graph: nx.Graph,
    coloring: NACColoring | Dict[str, Collection[Edge]],
    allow_non_surjective: bool = False,
    runs_on_subgraph: bool = False,
) -> bool:
    """
    Check if the coloring given is a cartesian NAC coloring.

    Parameters:
    ----------
        coloring: the coloring to check if it is a cartesian NAC coloring.
        allow_non_surjective: if True, allows the coloring to be non-surjective.
            This can be useful for checking subgraphs - the can use only one color.
        runs_on_subgraph: if True, the check that all the graph edges are
            colored is disabled.
    ----------

    Pseudocode:
        find red and blue components
        for each vertex get ids of it's components
        iterate over all the vertices with 2+ components
        mark neighboring components
        if two components are already neighboring, return false
        if no problem is found, return true
    """
    red: Collection[Edge]
    blue: Collection[Edge]

    if type(coloring) == dict:
        red, blue = coloring["red"], coloring["blue"]
    else:
        red, blue = coloring
    assert type(red) == type(blue)

    # Both colors have to be used
    if len(red) == 0 or len(blue) == 0:
        return allow_non_surjective

    if not check_NAC_constrains(graph):
        return False

    if not runs_on_subgraph and len(red) + len(blue) != len(graph.edges):
        return False

    if type(red) == set and len(red.intersection(blue)) != 0:
        return False
    else:
        for e in red:
            if e in blue:
                return False

    return _is_cartesian_NAC_coloring_impl(graph, (red, blue))
