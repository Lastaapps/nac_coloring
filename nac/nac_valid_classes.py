"""
This module is responsible for finding monochromatic classes or
triangle components.
"""

from collections import defaultdict
from typing import *
from enum import Enum
import networkx as nx

from nac.util.union_find import UnionFind
from nac.data_type import Edge


class NACValidClassType(Enum):
    """
    Represents the way monochromatic classes are found.
    """

    """Each edge is its own monochromatic class."""
    EDGES = "EDGES"
    """Each triangle-connected component it its own monochromatic class."""
    TRIANGLES = "TRIANGLES"
    """Creates monochromatic classes according to the paper."""
    EXTENDED = "MONOCHROMATIC"


def _trivial_monochromatic_classes(
    graph: nx.Graph,
) -> Tuple[Dict[Edge, int], List[List[Edge]]]:
    edge_to_component: Dict[Edge, int] = {}
    component_to_edges: List[List[Edge]] = []
    for i, e in enumerate(graph.edges):
        edge_to_component[e] = i
        component_to_edges.append([e])
    return edge_to_component, component_to_edges


def find_monochromatic_classes(
    graph: nx.Graph,
    class_type: NACValidClassType = NACValidClassType.EXTENDED,
    is_cartesian_NAC_coloring: bool = False,
) -> Tuple[Dict[Edge, int], List[List[Edge]]]:
    """
    Finds all the components of triangle equivalence.

    Returns mapping from edges to their component id (int)
    and the other way around from component to set of edges
    Order of vertices in the edge pairs is arbitrary.

    If the cartesian mode is switched on,
    also all the cycles of four are found and merged.
    Make sure you handle this correctly in your subsequent pipeline.

    Components are indexed from 0.
    """
    if class_type == NACValidClassType.EDGES:
        return _trivial_monochromatic_classes(graph)

    components = UnionFind[Edge]()

    # Finds triangles
    for edge in graph.edges:
        v, u = edge

        # We cannot sort vertices and we cannot expect
        # any regularity or order of the vertices
        components.join((u, v), (v, u))

        v_neighbours = set(graph.neighbors(v))
        u_neighbours = set(graph.neighbors(u))
        intersection = v_neighbours.intersection(u_neighbours)
        for w in intersection:
            components.join((u, v), (w, v))
            components.join((u, v), (w, u))

    # Checks for edges & triangles over component
    # This MUST be run before search for squares of other search
    # that may produce disconnected components, as cycles may not exist!
    # There routines are highly inefficient, but the time is still
    # negligible compared to the main algorithm running time.
    if class_type == NACValidClassType.EXTENDED:
        # we try again until we find no other component to merge
        # new opinions may appear later
        # could be most probably implemented smarter
        done = False
        while not done:
            done = True

            vertex_to_components: List[Set[Edge]] = [
                set() for _ in range(max(graph.nodes) + 1)
            ]

            # prepare updated vertex to component mapping
            for e in graph.edges:
                comp_id = components.find(e)
                vertex_to_components[e[0]].add(comp_id)
                vertex_to_components[e[1]].add(comp_id)

            # v is the top of the triangle over component
            for v in graph.nodes:
                # maps component to set of vertices containing it
                comp_to_vertices: Dict[Edge, Set[int]] = defaultdict(set)
                for u in graph.neighbors(v):
                    # find all the components v neighbors with
                    for comp in vertex_to_components[u]:
                        comp_to_vertices[comp].add(u)

                # if we found more edges to the same component,
                # we also found a triangle and we merge it's arms
                for comp, vertices in comp_to_vertices.items():
                    if comp == len(vertices) <= 1:
                        continue

                    vertices = iter(vertices)
                    w = next(vertices)
                    for u in vertices:
                        # if something changed, we may have another
                        # change for improvement the next round
                        done &= not components.join((v, w), (v, u))

    # Finds squares
    for edge in graph.edges if is_cartesian_NAC_coloring else []:
        v, u = edge

        v_neighbours = set(graph.neighbors(v))
        v_neighbours.remove(u)

        for w in graph.neighbors(u):
            if w == v:
                continue
            for n in graph.neighbors(w):
                if n not in v_neighbours:
                    continue

                components.join((u, v), (w, n))
                components.join((u, w), (v, n))

    edge_to_component: Dict[Edge, int] = {}
    component_to_edge: List[List[Edge]] = []

    for edge in graph.edges:
        root = components.find(edge)

        if root not in edge_to_component:
            edge_to_component[root] = id = len(component_to_edge)
            component_to_edge.append([])
        else:
            id = edge_to_component[root]

        edge_to_component[edge] = id
        component_to_edge[id].append(edge)

    return edge_to_component, component_to_edge


def create_component_graph_from_components(
    graph: nx.Graph,
    edges_to_components: Dict[Edge, int],
) -> nx.Graph:
    """
    Deprecated as the whole component graph idea

    Creates a component graph from the components given.
    Each edge must belong to a component.
    Ids of these components are then used
    as vertices of the new graph.
    Component id's start from 0.
    """

    # graph used to find NAC coloring easily
    comp_graph = nx.Graph()

    def get_edge_component(e: Edge) -> int:
        u, v = e
        res = edges_to_components.get((u, v))
        if res is None:
            res = edges_to_components[(v, u)]
        return res

    for v in graph.nodes:
        edges = list(graph.edges(v))
        for i in range(0, len(edges)):
            c1 = get_edge_component(edges[i])
            # this must be explicitly added in case
            # the edge has no neighboring edges
            # in that case this is a single node
            comp_graph.add_node(c1)

            for j in range(i + 1, len(edges)):
                c2 = get_edge_component(edges[j])
                if c1 == c2:
                    continue
                elif c1 < c2:
                    comp_graph.add_edge(c1, c2)
                else:
                    comp_graph.add_edge(c2, c1)

    return comp_graph
