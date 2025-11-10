from typing import Callable, Iterable, Optional

import networkx as nx
from pyrigi.data_type import Vertex

from stablecut.types import SeparatingCut


def _to_vertices[T](vertices: Iterable[T] | SeparatingCut[T]) -> set[T]:
    """
    Converts multiple input formats into the graph separator.
    """
    if isinstance(vertices, set):
        return vertices
    if isinstance(vertices, SeparatingCut):
        return vertices.cut
    return set(vertices)


def stable_set_violation[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
) -> Optional[tuple[T, T]]:
    """
    Checks if the given set of vertices is stable in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """

    vertices = _to_vertices(vertices)
    for v in vertices:
        for n in graph.neighbors(v):
            if n in vertices:
                return v, n
    return None


def is_stable_set[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
) -> bool:
    """
    Checks if the given set of vertices is stable in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """
    return stable_set_violation(graph, vertices) is None


def _revertable_set_removal[T: Vertex, R](
    graph: nx.Graph,
    vertices: set[T],
    opt: Callable[[nx.Graph], R],
) -> R:
    """
    Remove given vertices from the graph, perform operation,
    return vertices along with edges back.

    Parameters
    ----------
    graph:
        The graph from which vertices will be removed
    vertices:
        Vertex set to remove
    opt:
        Operation to perform on a graph with vertices removed

    Note
    ----
        Edge and vertex data are not preserved, make a copy yourself.
    """
    copy = nx.is_frozen(graph)

    if copy:
        graph = nx.Graph(graph)
        neighbors = []
    else:
        neighbors = [(u, v) for u in vertices for v in graph.neighbors(u)]

    graph.remove_nodes_from(vertices)

    res = opt(graph)

    if not copy:
        graph.add_edges_from(neighbors)

    return res


def is_separating_set[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
) -> bool:
    """
    Checks if the given set of vertices is a separator in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """

    vertices = _to_vertices(vertices)
    return _revertable_set_removal(graph, vertices, lambda g: not nx.is_connected(g))


def is_separating_set_dividing[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
    u: T,
    v: T,
) -> bool:
    """
    Checks if the given cut separates vertices u and v.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    u:
        The first vertex
    v:
        The second vertex

    Raises
    ------
    If either of the vertices is contained in the set, exception is thrown
    """
    vertices = _to_vertices(vertices)

    if u in vertices:
        raise ValueError(f"u={u} is in the cut set")
    if v in vertices:
        raise ValueError(f"v={v} is in the cut set")

    def check_graph(g: nx.Graph) -> bool:
        components = nx.connected_components(g)
        for c in components:
            if u in c and v in c:
                return False
        return True

    return _revertable_set_removal(graph, vertices, check_graph)


def is_stable_cutset[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """
    return is_stable_set(graph, vertices) and is_separating_set(graph, vertices)


def is_stable_cutset_dividing[T: Vertex](
    graph: nx.Graph,
    vertices: Iterable[T] | SeparatingCut[T],
    u: T,
    v: T,
) -> bool:
    """
    Checks if the given set of vertices is a stable cut in the given graph.

    Parameters
    ----------
    graph:
        The graph to check
    vertices:
        The vertices to check
    """
    return is_stable_set(graph, vertices) and is_separating_set_dividing(
        graph, vertices, u, v
    )
