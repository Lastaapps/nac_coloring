"""
This modules is responsible for checking if an edge coloring is a NAC_coloring
"""

from typing import *

import networkx as nx

from nac.data_type import NACColoring, LoopError
from nac.nac_valid_classes import find_nac_mono_classes


def check_NAC_constrains(self: nx.Graph) -> bool:
    """
    Checks some basic constrains for NAC coloring to even make sense

    Throws:
        ValueError: if the graph is empty or has loops

    Returns:
        True if a NAC coloring may exist, False if none exists for sure
    """
    if nx.number_of_selfloops(self) > 0:
        raise LoopError()

    if self.nodes() == 0:
        raise ValueError("Undefined for an empty graph")

    if nx.is_directed(self):
        raise ValueError("Cannot process a directed graph")

    # NAC is a surjective edge coloring, you passed a graph with less then 2 edges
    if len(nx.edges(self)) < 2:
        return False

    return True


def _can_have_flexible_labeling(
    graph: nx.Graph,
) -> bool:
    """
    Paper: Graphs with flexible labelings - Theorem 3.1, 4.7.

    Uses equivalence that graph (more below) has NAC coloring iff.
    it has a flexible labeling. But for flexible labeling we know upper bound
    for number of edges in the graph.

    Parameters
    ----------
    graph:
        A connected graph with at leas one edge.
    ----------

    Return
        True if the graph can have NAC coloring,
        False if we are sure there is none.
    """
    assert graph.number_of_edges() >= 1
    assert nx.node_connectivity(graph) > 0
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    return m <= n * (n - 1) // 2 - (n - 2)


def _check_for_simple_stable_cut(
    graph: nx.Graph,
    certificate: bool,
) -> Optional[NACColoring | Any]:
    """
    Paper: Graphs with flexible labelings - Theorem 4.4, Corollary 4.5.

    If there is a single vertex outside of any triangle component,
    we can trivially find NAC coloring for the graph.
    Also handles nodes with degree <= 2.

    Parameters
    ----------
    graph:
        The graph to work with, basic NAC coloring constrains
        should be already checked.
    certificate:
        whether or not to return some NAC coloring
        obtainable by this method. See returns.
    ----------

    Returns
        If no NAC coloring can be found using this method, None is
        returned. If some NAC coloring can be found, return something
        (not None) if certificate is not needed (False)
        or the found coloring if it is requested (True).
    """
    _, component_to_edge = find_nac_mono_classes(graph)
    verticies_outside_triangle_components: Set[int] = set(
        # make sure we filter out isolated vertices
        u
        for u, d in graph.degree()
        if d > 0
    )
    for component_edges in component_to_edge:
        # component is not part of a triangle edge
        if len(component_edges) == 1:
            continue

        verticies_outside_triangle_components.difference_update(
            v for edge in component_edges for v in edge
        )

    if len(verticies_outside_triangle_components) == 0:
        return None

    if not certificate:
        return "NAC_coloring_exists"

    for v in verticies_outside_triangle_components:
        red = set((v, u) for u in graph.neighbors(v))
        if len(red) == graph.number_of_edges():
            # we found a wrong vertex
            # this may happen if the red part is the whole graph
            continue
        blue = set(graph.edges)

        # remove shared edges
        blue.difference_update(red)
        blue.difference_update((u, v) for v, u in red)

        assert len(red) > 0
        assert len(blue) > 0

        return (red, blue)
    assert False


def has_NAC_coloring_checks(self) -> Optional[bool]:
    """
    Implementation for has_NAC_coloring, but without fallback to
    single_NAC_coloring. May be used before an exhaustive search that
    wouldn't find anything anyway.
    """
    if _check_for_simple_stable_cut(self, False) is not None:
        return True

    if nx.algorithms.connectivity.node_connectivity(self) < 2:
        return True

    # Needs to be run after connectivity checks
    if not _can_have_flexible_labeling(self):
        return False

    return None
