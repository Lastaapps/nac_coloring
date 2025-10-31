"""
This module holds functions related to questions if a graph has a NAC coloring
with optional certificate.
"""

from __future__ import annotations

from typing import *

import networkx as nx

from nac.data_type import NACColoring

from nac.search import NAC_colorings
from nac.existence import (
    check_NAC_constrains,
    _check_for_simple_stable_cut,
    _can_have_flexible_labeling,
    has_NAC_coloring_checks,
)


def _single_general_NAC_coloring(self) -> Optional[NACColoring]:
    """
    Tries to find some trivial NAC coloring based on connectivity of
    the graph components. This coloring is trivially both NAC coloring
    and cartesian NAC coloring.
    """
    components = list(nx.algorithms.components.connected_components(self))
    if len(components) > 1:
        # filter all the single nodes
        components = list(filter(lambda nodes: len(nodes) > 1, components))
        component: Set[int] = components[0]

        # there are more disconnected components with at least one edge,
        # we can color both of them with different color and be done.
        if len(components) > 1:
            red, blue = set(), set()
            for u, v in nx.edges(self):
                (red if u in component else blue).add((u, v))
            return (red, blue)

        # if there is only one component with all the edges,
        # the NAC coloring exists <=> this component has NAC coloring
        return single_NAC_coloring_impl(nx.Graph(self.subgraph(component)))

    if nx.algorithms.connectivity.node_connectivity(self) < 2:
        generator = nx.algorithms.biconnected_components(self)
        component: Set[int] = next(generator)
        assert next(generator)  # make sure there are more components

        red, blue = set(), set()
        for v, u in self.edges:
            (red if v in component and u in component else blue).add((u, v))

        return (red, blue)

    return None


def has_NAC_coloring_impl(graph: nx.Graph) -> bool:
    """
    Same as single_NAC_coloring, but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
    """
    if not check_NAC_constrains(graph):
        return False

    res = has_NAC_coloring_checks(graph)
    if res is not None:
        return res

    return (
        single_NAC_coloring_impl(
            graph,
            # we already checked some things
            _is_first_check=False,
        )
        is not None
    )


def single_NAC_coloring_impl(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
    _is_first_check: bool = True,
) -> Optional[NACColoring]:
    """
    Finds only a single NAC coloring if it exists.
    Some other optimizations may be used
    to improve performance for some graph classes.

    Parameters
    ----------
    algorithm:
        The algorithm used in case we need to fall back
        to exhaustive search.
    _is_first_check:
        Internal parameter, do not change!
        Skips some useless checks as those things were already checked
        before in has_NAC_coloring.
    ----------
    """
    if _is_first_check:
        if not check_NAC_constrains(graph):
            return None

        res = _check_for_simple_stable_cut(graph, True)
        if res is not None:
            return res

        res = _single_general_NAC_coloring(graph)
        if res is not None:
            return res

        # Need to be run after connectivity checks
        if not _can_have_flexible_labeling(graph):
            return None

    return next(
        iter(
            NAC_colorings(
                graph=graph,
                algorithm=algorithm,
                # we already checked for bridges
                use_decompositions=False,
                use_has_coloring_check=False,
            )
        ),
        None,
    )


################################################################################
# Cartesian NAC
################################################################################


def has_cartesian_NAC_coloring_impl(self) -> bool:
    """
    Same as single_castesian_NAC_coloring,
    but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
    """
    if not self.check_NAC_constrains():
        return False

    if nx.algorithms.connectivity.node_connectivity(self) < 2:
        return True
    return self.single_cartesian_NAC_coloring() is not None


def single_cartesian_NAC_coloring_impl(
    graph: nx.Graph,
    algorithm: str | None,
) -> Optional[NACColoring]:
    """
    Finds only a single NAC coloring if it exists.
    Some other optimizations may be used
    to improve performance for some graph classes.
    """
    if not check_NAC_constrains(graph):
        return None

    common = _single_general_NAC_coloring(graph)
    if common is not None:
        return common

    return next(
        NAC_colorings(
            graph=graph,
            algorithm=algorithm,
            use_decompositions=False,
        ),
        None,
    )
