"""
This modules is the main entry point for the NAC-colorings search.
The main entry point is the function NAC_colorings_impl.
Then the graph given is relabeled according to
the given relabel strategy.

Then algorithm argument is parsed and either
Naive algorithm (_NAC_colorings_naive),
Naive alg. optimized using cycles mask matching (_NAC_colorings_cycles) and
Subgraph decomposition algorithm (_NAC_colorings_subgraphs).
It contains a huge switch statement choosing the splitting strategy
and another one for merging strategies.
The neighbors strategy is implemented in _subgraphs_strategy_neighbors

The main pair of the search uses only a bitmask representing
the coloring and converts it back to NAC coloring once a check is need.
"""

from functools import reduce
import random
from typing import *

import networkx as nx
import numpy as np

from nac.algorithms import (
    NAC_colorings_cycles,
    NAC_colorings_naive,
    NAC_colorings_subgraphs,
    NAC_colorings_with_non_surjective,
    NAC_colorings_without_vertex,
)
from nac.util.repetable_iterator import RepeatableIterator

from nac.data_type import NACColoring, Edge, NiceGraph
from nac.nac_valid_classes import (
    NACValidClassType,
    find_nac_mono_classes,
    create_component_graph_from_components,
)
from nac.existence import has_NAC_coloring_checks, check_NAC_constrains
from nac.check import (
    _is_cartesian_NAC_coloring_impl,
    _is_NAC_coloring_impl,
    _NAC_check_called_reset,
)


def _NAC_colorings_cross_product(
    first: Iterable[NACColoring], second: Iterable[NACColoring]
) -> Iterable[NACColoring]:
    """
    This makes a cartesian cross product of two NAC coloring iterators.
    Unlike the python crossproduct, this adds the colorings inside the tuples.
    """
    cache = RepeatableIterator(first)
    for s in second:
        for f in cache:
            yield s[0] + f[0], s[1] + f[1]


def _NAC_colorings_for_single_edges(
    edges: Sequence[Edge], include_non_surjective: bool
) -> Iterable[NACColoring]:
    """
    Creates all the NAC colorings for the edges given.
    The colorings are not check for validity.
    Make sure the edges given cannot form a red/blue almost cycle.
    """

    def binaryToGray(num: int) -> int:
        return num ^ (num >> 1)

    red: Set[Edge] = set()
    blue: Set[Edge] = set(edges)

    if include_non_surjective:
        # create immutable versions
        r, b = list(red), list(blue)
        yield r, b
        yield b, r

    prev_mask = 0
    for mask in range(1, 2 ** len(edges) // 2):
        mask = binaryToGray(mask)
        diff = prev_mask ^ mask
        prev_mask = mask
        is_new = (mask & diff) > 0
        edge = edges[diff.bit_length()]
        if is_new:
            red.add(edge)
            blue.remove(edge)
        else:
            blue.add(edge)
            red.remove(edge)

        # create immutable versions
        r, b = list(red), list(blue)
        yield r, b
        yield b, r


################################################################################
def _NAC_colorings_from_bridges(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
    copy: bool = True,
) -> Iterable[NACColoring]:
    """
    Optimization for NAC coloring search that first finds bridges and
    biconnected components. After that it find coloring for each component
    separately and then combines all the possible found NAC colorings.

    Parameters
    ----------
    processor:
        A function that takes a graph (subgraph of the graph given)
        and finds all the NAC colorings for it.
    copy:
        If the graph should be copied before making any destructive changes.
    ----------

    Returns:
        All the NAC colorings of the graph.
    """

    bridges = list(nx.bridges(graph))
    if len(bridges) == 0:
        return processor(graph)

    if copy:
        graph = nx.Graph(graph)
    graph.remove_edges_from(bridges)
    components = [
        nx.induced_subgraph(graph, comp)
        for comp in nx.components.connected_components(graph)
    ]
    components = filter(lambda x: x.number_of_nodes() > 1, components)

    colorings = [
        NAC_colorings_with_non_surjective(comp, processor(comp)) for comp in components
    ]
    colorings.append(
        _NAC_colorings_for_single_edges(bridges, include_non_surjective=True),
    )

    iterator = reduce(_NAC_colorings_cross_product, colorings)

    # Skip initial invalid coloring that are not surjective
    iterator = filter(lambda x: len(x[0]) * len(x[1]) != 0, iterator)

    return iterator


def _NAC_colorings_from_articulation_points(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
) -> Iterable[NACColoring]:
    colorings: List[Iterable[NACColoring]] = []
    for component in nx.components.biconnected_components(graph):
        subgraph = nx.induced_subgraph(graph, component)
        iterable = processor(subgraph)
        iterable = NAC_colorings_with_non_surjective(subgraph, iterable)
        colorings.append(iterable)

    iterator = reduce(_NAC_colorings_cross_product, colorings)

    # Skip initial invalid coloring that are not surjective
    iterator = filter(lambda x: len(x[0]) * len(x[1]) != 0, iterator)

    return iterator


def _NAC_colorings_with_trivial_stable_cuts(
    graph: nx.Graph,
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    copy: bool = True,
) -> Iterable[NACColoring]:
    _, component_to_edge = find_nac_mono_classes(
        graph,
        is_cartesian_NAC_coloring=False,
    )

    verticies_in_triangle_components: Set[int] = set()
    for component_edges in component_to_edge:
        # component is not part of a triangle edge
        if len(component_edges) == 1:
            continue

        verticies_in_triangle_components.update(
            v for edge in component_edges for v in edge
        )

    # All nodes are members of a triangle component
    if len(verticies_in_triangle_components) == graph.number_of_nodes():
        return processor(graph)

    if copy:
        graph = nx.Graph(graph)

    removed_edges: List[Edge] = []
    # vertices outside of triangle components
    for v in verticies_in_triangle_components - set(graph.nodes):
        for u in graph.neighbors(v):
            removed_edges.append((u, v))
        graph.remove_node(v)

    coloring = processor(graph)
    coloring = NAC_colorings_with_non_surjective(graph, coloring)

    def handle_vertex(
        graph: nx.Graph,
        v: int,
        get_coloring: Callable[[nx.Graph], Iterable[NACColoring]],
    ) -> Iterable[NACColoring]:
        graph = nx.Graph(graph)
        edges: List[Edge] = []
        for u in graph.neighbors(v):
            edges.append((u, v))

        for coloring in get_coloring(graph):
            # create red/blue components
            pass

        pass

    pass


################################################################################
def _renamed_coloring(
    ordered_vertices: Sequence[int],
    colorings: Iterable[NACColoring],
) -> Iterable[NACColoring]:
    """
    Expects graph vertices to be named from 0 to N-1.
    """
    for coloring in colorings:
        yield tuple(
            [(ordered_vertices[u], ordered_vertices[v]) for u, v in group]
            for group in coloring
        )


def _relabel_graph_for_NAC_coloring(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
    seed: int,
    strategy: str = "random",
    copy: bool = True,
    restart_after: int | None = None,
) -> Iterable[NACColoring]:
    vertices = list(graph.nodes)

    if strategy == "none":
        if set(vertices) == set(range(graph.number_of_nodes())):
            return processor(graph)

        # make all the nodes names in range 0..<n
        mapping = {v: k for k, v in enumerate(vertices)}
        graph = nx.relabel_nodes(graph, mapping, copy=copy)
        return _renamed_coloring(vertices, processor(graph))

    used_vertices: Set[int] = set()
    ordered_vertices: List[int] = []

    if restart_after is None:
        restart_after = graph.number_of_nodes()

    random.Random(seed).shuffle(vertices)

    if strategy == "random":
        mapping = {v: k for k, v in enumerate(vertices)}
        graph = nx.relabel_nodes(graph, mapping, copy=copy)
        return _renamed_coloring(vertices, processor(graph))

    for start in vertices:
        if start in used_vertices:
            continue

        found_vertices = 0
        match strategy:
            case "bfs":
                iterator = nx.bfs_edges(graph, start)
            case "beam_degree":
                iterator = nx.bfs_beam_edges(
                    graph,
                    start,
                    lambda v: nx.degree(graph, v),
                    width=max(5, int(np.sqrt(graph.number_of_nodes()))),
                )
            case _:
                raise ValueError(
                    f"Unknown strategy for relabeling: {strategy}, posible values are none, random, bfs and beam_degree"
                )

        used_vertices.add(start)
        ordered_vertices.append(start)

        for tail, head in iterator:
            if head in used_vertices:
                continue

            used_vertices.add(head)
            ordered_vertices.append(head)

            found_vertices += 1
            if found_vertices >= restart_after:
                break

        assert start in used_vertices

    mapping = {v: k for k, v in enumerate(ordered_vertices)}
    graph = nx.relabel_nodes(graph, mapping, copy=copy)
    return _renamed_coloring(ordered_vertices, processor(graph))


################################################################################
def NAC_colorings_impl(
    self: nx.Graph,
    algorithm: str,
    relabel_strategy: str,
    use_decompositions: bool,
    is_cartesian: bool,
    monochromatic_class_type: NACValidClassType,
    remove_vertices_cnt: int,
    use_has_coloring_check: bool,
    seed: int | None,
) -> Iterable[NACColoring]:
    _NAC_check_called_reset()

    if not check_NAC_constrains(self):
        return []

    # Checks if it even makes sense to do the search
    if use_has_coloring_check and has_NAC_coloring_checks(self) == False:
        return []

    rand = random.Random(seed)

    def run(graph: nx.Graph) -> Iterable[NACColoring]:
        graph = NiceGraph(graph)

        # in case graph has no edges because of some previous optimizations,
        # there are no NAC colorings
        if graph.number_of_edges() == 0:
            return []

        edge_to_component, component_to_edge = find_nac_mono_classes(
            graph,
            monochromatic_class_type,
            is_cartesian_NAC_coloring=is_cartesian,
        )

        comp_graph = create_component_graph_from_components(graph, edge_to_component)

        algorithm_parts = list(algorithm.split("-"))
        match algorithm_parts[0]:
            case "naive":
                is_NAC_coloring = (
                    _is_cartesian_NAC_coloring_impl
                    if is_cartesian
                    else _is_NAC_coloring_impl
                )
                return NAC_colorings_naive(
                    graph,
                    list(comp_graph.nodes),
                    component_to_edge,
                    is_NAC_coloring,
                )
            case "cycles":
                is_NAC_coloring = (
                    _is_cartesian_NAC_coloring_impl
                    if is_cartesian
                    else _is_NAC_coloring_impl
                )

                if len(algorithm_parts) == 1:
                    return NAC_colorings_cycles(
                        graph,
                        list(comp_graph.nodes),
                        component_to_edge,
                        is_NAC_coloring,
                        is_cartesian,
                    )
                return NAC_colorings_cycles(
                    graph,
                    list(comp_graph.nodes),
                    component_to_edge,
                    is_NAC_coloring,
                    from_angle_preserving_components=is_cartesian,
                    use_all_cycles=bool(algorithm_parts[1]),
                )
            case "subgraphs":
                is_NAC_coloring = (
                    _is_cartesian_NAC_coloring_impl
                    if is_cartesian
                    else _is_NAC_coloring_impl
                )

                if len(algorithm_parts) == 1:
                    return NAC_colorings_subgraphs(
                        graph,
                        comp_graph,
                        component_to_edge,
                        is_NAC_coloring,
                        from_angle_preserving_components=is_cartesian,
                        seed=rand.randint(0, 2**16 - 1),
                    )
                return NAC_colorings_subgraphs(
                    graph,
                    comp_graph,
                    component_to_edge,
                    is_NAC_coloring,
                    from_angle_preserving_components=is_cartesian,
                    seed=rand.randint(0, 2**16 - 1),
                    merge_strategy=algorithm_parts[1],
                    order_strategy=algorithm_parts[2],
                    preferred_chunk_size=(
                        int(algorithm_parts[3])
                        if algorithm_parts[3] != "auto"
                        else None
                    ),
                    use_smart_split=(
                        algorithm_parts[4] == "smart"
                        if len(algorithm_parts) == 5
                        else False
                    ),
                )
            case _:
                raise ValueError(f"Unknown algorighm type: {algorithm}")

    def apply_processor(
        processor: Callable[[nx.Graph], Iterable[NACColoring]],
        func: Callable[
            [Callable[[nx.Graph], Iterable[NACColoring]], nx.Graph],
            Iterable[NACColoring],
        ],
    ) -> Callable[[nx.Graph], Iterable[NACColoring]]:
        return lambda g: func(processor, g)

    graph: nx.Graph = self
    processor: Callable[[nx.Graph], Iterable[NACColoring]] = run

    assert not (is_cartesian and remove_vertices_cnt > 0)
    for _ in range(remove_vertices_cnt):
        processor = apply_processor(
            processor, lambda p, g: NAC_colorings_without_vertex(p, g, None)
        )

    if use_decompositions:
        # processor = apply_processor(
        #     processor, lambda p, g: _NAC_colorings_from_bridges(p, g)
        # )
        processor = apply_processor(
            processor,
            lambda p, g: _NAC_colorings_from_articulation_points(p, g),
        )

    processor = apply_processor(
        processor,
        lambda p, g: _relabel_graph_for_NAC_coloring(
            p, g, strategy=relabel_strategy, seed=rand.randint(0, 2**16 - 1)
        ),
    )

    return processor(graph)


def NAC_colorings(
    graph: nx.Graph,
    algorithm: str = "subgraphs",
    relabel_strategy: str = "none",
    monochromatic_class_type: NACValidClassType = NACValidClassType.EXTENDED,
    use_decompositions: bool = True,
    remove_vertices_cnt: int = 0,
    use_has_coloring_check: bool = True,
    seed: int | None = None,
) -> Iterable[NACColoring]:
    return NAC_colorings_impl(
        self=graph,
        algorithm=algorithm,
        relabel_strategy=relabel_strategy,
        monochromatic_class_type=monochromatic_class_type,
        use_decompositions=use_decompositions,
        is_cartesian=False,
        remove_vertices_cnt=remove_vertices_cnt,
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
