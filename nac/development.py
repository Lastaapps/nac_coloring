"""
This module holds some utility functions for development
"""

from __future__ import annotations

from collections import defaultdict
from typing import *

import networkx as nx
import time

from nac.data_type import Edge

statistics_storage: Dict[int, List[Any]] = defaultdict(list)

# optional printing
NAC_PRINT_SWITCH = False
# NAC_PRINT_SWITCH = True


def NAC_statistics_colorings_merge[
    T
](
    func: Callable[..., Tuple[Iterable[T], int]],
) -> Callable[
    ..., Tuple[Iterable[T], int]
]:
    return func

    def stats(*args, **kwargs) -> Tuple[List[T], int]:
        level = max(args[0][1].bit_count(), args[1][1].bit_count()) * 2
        start = time.time()
        orig, mask = func(*args, **kwargs)
        res: List[T] = list(orig)
        end = time.time()
        diff = end - start
        statistics_storage[level].append((diff, len(res)))
        return res, mask

    return stats


def NAC_statistics_generator[
    T
](func: Callable[..., Iterable[T]],) -> Callable[..., Iterable[T]]:
    return func

    def stats(*args, **kwargs) -> Iterable[T]:
        level = 0
        start = time.time()
        res = list(func(*args, **kwargs))
        end = time.time()
        diff = end - start
        statistics_storage[level].append((diff, len(res)))
        return res

    return stats


def print_stats():
    for key in sorted(statistics_storage.keys(), reverse=True):
        l = statistics_storage[key]
        s = sum(map(lambda x: x[0], l))
        print(f"Level: {key:2d} - {s:.8f} -> {l}")
    statistics_storage.clear()


graphviz_colors = [
    "red",
    "green",
    "cyan",
    "purple",
    "yellow",
    "orange",
    "pink",
    "teal",
    "navy",
    "maroon",
    "gray",
    "silver",
    "gold",
]


def graphviz_components(
    name: str,
    component_to_edges: Collection[Collection[Edge]],
) -> str:
    my_graph = nx.Graph()
    my_graph.name = name

    for i, component in enumerate(component_to_edges):
        for edge in component:
            my_graph.add_edges_from(
                [
                    (
                        *edge,
                        {
                            "color": graphviz_colors[i % len(graphviz_colors)],
                            "style": "filled",
                            "label": i,
                        },
                    )
                ]
            )

    return nx.nx_agraph.to_agraph(my_graph)


def graphviz_graph(
    name: str,
    component_to_edges: List[List[Edge]],
    chunk_sizes: List[int],
    vertices: List[int],
) -> str:
    my_graph = nx.Graph()
    my_graph.name = name
    offset = 0

    for i, chunk_size in enumerate(chunk_sizes):
        local_vertices = vertices[offset : offset + chunk_size]
        offset += chunk_size
        for v in local_vertices:
            my_graph.add_edges_from(
                [
                    (
                        *e,
                        {
                            "color": graphviz_colors[i % len(graphviz_colors)],
                            "style": "filled",
                        },
                    )
                    for e in component_to_edges[v]
                ]
            )

    return nx.nx_agraph.to_agraph(my_graph)


def graphviz_component_graph(
    comp_graph: nx.Graph,
    name: str,
    component_to_edges: List[List[Edge]],
    chunk_sizes: List[int],
    vertices: List[int],
) -> str:
    my_comp_graph = nx.Graph()
    my_comp_graph.name = name
    offset = 0

    for i, chunk_size in enumerate(chunk_sizes):
        local_vertices = vertices[offset : offset + chunk_size]
        offset += chunk_size
        for v in local_vertices:
            my_comp_graph.add_nodes_from(
                [
                    (
                        v,
                        {
                            "color": graphviz_colors[i % len(graphviz_colors)],
                            "style": "filled",
                        },
                    )
                ]
            )
    my_comp_graph.add_edges_from(comp_graph.edges)
    my_comp_graph = nx.relabel_nodes(
        my_comp_graph,
        {v: f"{component_to_edges[v]} ({v})" for v in my_comp_graph.nodes()},
    )

    return nx.nx_agraph.to_agraph(my_comp_graph)
