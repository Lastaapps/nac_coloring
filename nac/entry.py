"""
This module holds the nac module main API, that should stay stable over time.
"""

from typing import *

import networkx as nx

from nac.data_type import NACColoring
from nac.search import NAC_colorings_impl
from nac.single import has_NAC_coloring_impl, single_NAC_coloring_impl
from nac.single import (
    has_cartesian_NAC_coloring_impl,
    single_cartesian_NAC_coloring_impl,
)
from nac.nac_valid_classes import NACValidClassType


def NAC_colorings(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
    relabel_strategy: str = "none",
    nac_valid_class_type: NACValidClassType = NACValidClassType.EXTENDED,
    use_decompositions: bool = True,
    use_has_coloring_check: bool = True,
    seed: int | None = None,
) -> Iterable[NACColoring]:
    return NAC_colorings_impl(
        self=graph,
        algorithm=algorithm,
        relabel_strategy=relabel_strategy,
        monochromatic_class_type=nac_valid_class_type,
        use_decompositions=use_decompositions,
        is_cartesian=False,
        remove_vertices_cnt=0,
        use_has_coloring_check=use_has_coloring_check,
        seed=seed,
    )


def cartesian_NAC_colorings(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
    relabel_strategy: str = "none",
    monochromatic_class_type: NACValidClassType = NACValidClassType.EXTENDED,
    use_decompositions: bool = True,
    use_has_coloring_check: bool = True,
    seed: int | None = None,
) -> Iterable[NACColoring]:
    return NAC_colorings_impl(
        self=graph,
        algorithm=algorithm,
        relabel_strategy=relabel_strategy,
        monochromatic_class_type=monochromatic_class_type,
        use_decompositions=use_decompositions,
        is_cartesian=True,
        remove_vertices_cnt=0,
        use_has_coloring_check=use_has_coloring_check,
        seed=seed,
    )


def has_NAC_coloring(graph: nx.Graph) -> bool:
    """
    Same as single_NAC_coloring, but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
    """
    return has_NAC_coloring_impl(graph)


def single_NAC_coloring(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
) -> Optional[NACColoring]:
    return single_NAC_coloring_impl(graph, algorithm)


def has_cartesian_NAC_coloring(self) -> bool:
    """
    Same as single_castesian_NAC_coloring,
    but the certificate may not be created,
    so some additional tricks are used the performance may be improved.
    """
    return has_cartesian_NAC_coloring_impl(self)


def single_cartesian_NAC_coloring(
    graph: nx.Graph,
    algorithm: str | None = "subgraphs",
) -> Optional[NACColoring]:
    """
    Finds only a single NAC coloring if it exists.
    Some other optimizations may be used
    to improve performance for some graph classes.
    """
    return single_cartesian_NAC_coloring_impl(graph, algorithm)
