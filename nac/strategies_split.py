from collections import deque
import random
from typing import *

import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection
import numpy as np

from nac import Edge, NiceGraph

from nac.cycle_detection import find_cycles


def degree_ordered_nodes(graph: nx.Graph) -> List[int]:
    return list(
        map(
            lambda x: x[0],
            sorted(
                graph.degree(),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
    )


def _cycles_per_vertex(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[List[Tuple[int, ...]]]:
    """
    For each vertex we find all the cycles that contain it
    Finds all the shortest cycles in the graph
    """
    cycles = find_cycles(
        graph,
        set(comp_graph.nodes),
        component_to_edges,
        all=True,
    )

    vertex_cycles = [[] for _ in range(max(comp_graph.nodes) + 1)]
    for cycle in cycles:
        for v in cycle:
            vertex_cycles[v].append(cycle)
    return vertex_cycles


################################################################################
def subgraphs_strategy_degree_cycles(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[int]:
    degree_ordered_comp_ids = degree_ordered_nodes(comp_graph)
    vertex_cycles = _cycles_per_vertex(graph, comp_graph, component_to_edges)

    ordered_comp_ids: List[int] = []
    used_comp_ids: Set[int] = set()
    for v in degree_ordered_comp_ids:
        # Handle components with no cycles
        if v not in used_comp_ids:
            ordered_comp_ids.append(v)
            used_comp_ids.add(v)

        for cycle in vertex_cycles[v]:
            for u in cycle:
                if u in used_comp_ids:
                    continue
                ordered_comp_ids.append(u)
                used_comp_ids.add(u)
    return ordered_comp_ids


def subgraphs_strategy_cycles(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[int]:
    degree_ordered_comp_ids = degree_ordered_nodes(comp_graph)
    vertex_cycles = _cycles_per_vertex(graph, comp_graph, component_to_edges)

    ordered_comp_ids: List[int] = []
    used_comp_ids: Set[int] = set()
    all_comp_ids = deque(degree_ordered_comp_ids)

    while all_comp_ids:
        v = all_comp_ids.popleft()

        # the vertex may have been used before from a cycle
        if v in used_comp_ids:
            continue

        queue: Deque[int] = deque([v])

        while queue:
            u = queue.popleft()

            if u in used_comp_ids:
                continue

            ordered_comp_ids.append(u)
            used_comp_ids.add(u)

            for cycle in vertex_cycles[u]:
                for u in cycle:
                    if u in used_comp_ids:
                        continue
                    queue.append(u)
    return ordered_comp_ids


def subgraphs_strategy_cycles_match_chunks(
    chunk_sizes: Sequence[int],
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
) -> List[int]:
    chunk_size = chunk_sizes[0]
    degree_ordered_comp_ids = degree_ordered_nodes(comp_graph)
    vertex_cycles = _cycles_per_vertex(graph, comp_graph, component_to_edges)

    ordered_comp_ids: List[int] = []
    used_comp_ids: Set[int] = set()
    all_comp_ids = deque(degree_ordered_comp_ids)

    while all_comp_ids:
        v = all_comp_ids.popleft()

        # the vertex may have been used before from a cycle
        if v in used_comp_ids:
            continue

        used_in_epoch = 0
        queue: Deque[int] = deque([v])

        while queue and used_in_epoch < chunk_size:
            u = queue.popleft()

            if u in used_comp_ids:
                continue

            ordered_comp_ids.append(u)
            used_comp_ids.add(u)
            used_in_epoch += 1

            for cycle in vertex_cycles[u]:
                for u in cycle:
                    if u in used_comp_ids:
                        continue
                    queue.append(u)

        while used_in_epoch < chunk_size and len(used_comp_ids) != len(
            degree_ordered_comp_ids
        ):
            v = all_comp_ids.pop()
            if v in used_comp_ids:
                continue
            ordered_comp_ids.append(v)
            used_comp_ids.add(v)
            used_in_epoch += 1

    for v in degree_ordered_comp_ids:
        if v in used_comp_ids:
            continue
        ordered_comp_ids.append(v)

    return ordered_comp_ids


def subgraphs_strategy_bfs(
    comp_graph: nx.Graph,
    chunk_sizes: Sequence[int],
) -> List[int]:
    graph = nx.Graph(comp_graph)
    used_comp_ids: Set[int] = set()
    ordered_comp_ids_groups: List[List[int]] = [[] for _ in chunk_sizes]

    for v in degree_ordered_nodes(graph):
        if v in used_comp_ids:
            continue

        index_min = min(
            range(len(ordered_comp_ids_groups)),
            key=lambda x: len(ordered_comp_ids_groups[x]) / chunk_sizes[x],
        )
        target = ordered_comp_ids_groups[index_min]

        added_comp_ids: List[int] = [v]
        used_comp_ids.add(v)
        target.append(v)

        for _, u in nx.bfs_edges(graph, v):
            if u in used_comp_ids:
                continue

            added_comp_ids.append(u)
            target.append(u)
            used_comp_ids.add(u)

            if len(target) == chunk_sizes[index_min]:
                break

        graph.remove_nodes_from(added_comp_ids)

    return [v for group in ordered_comp_ids_groups for v in group]


def subgraphs_strategy_neighbors(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    chunk_sizes: Sequence[int],
    iterative: bool,
    start_with_cycle: bool,
    use_degree: bool,
    seed: int | None,
) -> List[int]:
    """
    Params
    ------
        graph:
            subgraph of the original graph that the algorithm operates on
        comp_graph:
            holds monochromatic components
        components_to_edges:
            mapping giving for a component id
            (a vertex in comp_graph) list of edges of the component
        chunk_sizes:
            target sizes of resulting subgraphs
        iterative:
            tries to add component to a subgraph,
            but starts with the next one instead of adding cycles.
            A cycles is added after a round is finished in the next round.
            Rounds run until all the subgraphs are full.
        start_with_cycle:
            a monochromatic cycle is added to each subgraph at start
            then normal process continues
        use_degree:

        seed:
            seeds internal pseudo random generator

        for each component
    """
    rand = random.Random(seed)

    # 'Component' stands for monochromatic classes as for legacy naming

    # comp_graph is just a peace of legacy code that we did not optimize away yet
    # here it serves only as a set of monochromatic components to consider
    comp_graph = nx.Graph(comp_graph)
    ordered_comp_ids_groups: List[List[int]] = [[] for _ in chunk_sizes]

    # if False, chunk does need to assign random component
    is_random_component_required: List[bool] = [True for _ in chunk_sizes]

    edge_to_components: Dict[Tuple[int, int], int] = {
        e: comp_id for comp_id, comp in enumerate(component_to_edges) for e in comp
    }

    # start algo and fill a chunk
    while comp_graph.number_of_nodes() > 0:
        component_ids = list(comp_graph.nodes)
        rand_comp = component_ids[rand.randint(0, len(component_ids) - 1)]

        # could by avoided by having a proper subgraph
        local_ordered_comp_ids: Set[int] = {
            v
            for comp_id, comp in enumerate(component_to_edges)
            for e in comp
            for v in e
            if comp_id in component_ids
        }

        # represents index of the chosen subgraph
        chunk_index = min(
            range(len(ordered_comp_ids_groups)),
            key=lambda x: len(ordered_comp_ids_groups[x]) / chunk_sizes[x],
        )
        # list to add monochromatic class to
        target = ordered_comp_ids_groups[chunk_index]

        # components already added to the subgraph
        added_components: Set[int]

        # if subgraph is still empty, we add a random monochromatic class to it
        if is_random_component_required[chunk_index]:
            if start_with_cycle:
                # we find all the cycles of reasonable length in the graph
                cycles = find_cycles(
                    graph,
                    set(comp_graph.nodes),
                    component_to_edges,
                    all=True,
                )

                # we find all the cycles with the shortest length
                shortest: List[Tuple[int, ...]] = []
                shortest_len = 2**42
                for cycle in cycles:
                    if len(cycle) == shortest_len:
                        shortest.append(cycle)
                    elif len(cycle) < shortest_len:
                        shortest = [cycle]
                        shortest_len = len(cycle)

                # no cycles found of the cycles are to long
                if len(cycles) == 0 or shortest_len > chunk_sizes[chunk_index] - len(
                    target
                ):
                    added_components: Set[int] = set([rand_comp])
                    target.append(rand_comp)
                else:
                    cycle = shortest[rand.randint(0, len(shortest) - 1)]
                    added_components = set([c for c in cycle])
                    target.extend(cycle)
            else:
                added_components: Set[int] = set([rand_comp])
                target.append(rand_comp)
            is_random_component_required[chunk_index] = False
        else:
            added_components = set()

        # vertices of the already chosen components
        used_vertices = {
            v for comp in target for e in component_to_edges[comp] for v in e
        }

        # vertices queue of vertices to search through
        opened: Set[int] = set()

        # add all the neighbors of the vertices of the already used components
        for v in used_vertices:
            for u in graph.neighbors(v):
                if u in used_vertices:
                    continue
                if u not in local_ordered_comp_ids:
                    continue
                opened.add(u)

        # goes trough the opened vertices and searches for cycles
        # until we run out or vertices, fill a subgraph or reach iteration limit
        iteration_no = 0
        while (
            opened
            and len(target) < chunk_sizes[chunk_index]
            and (iteration_no <= 2 or not iterative)
        ):
            comp_added = False

            # compute score or each vertex, using or not using vertex degrees
            if not use_degree:
                values = [
                    (u, len(used_vertices.intersection(graph.neighbors(u))))
                    for u in opened
                ]
            else:
                values = [
                    (
                        u,
                        (
                            len(used_vertices.intersection(graph.neighbors(u))),
                            # degree
                            -len(
                                local_ordered_comp_ids.intersection(graph.neighbors(u))
                            ),
                        ),
                    )
                    for u in opened
                ]

            # chooses a vertex with the highest score
            best_vertex = max(values, key=lambda x: x[1])[0]

            # we take the common neighborhood of already used vertices and the chosen vertex
            for neighbor in used_vertices.intersection(graph.neighbors(best_vertex)):
                # vertex is not part of the current subgraph
                if neighbor not in local_ordered_comp_ids:
                    continue

                # component of the edge incident to the best vertex
                # and the chosen vertex
                comp_id: int = edge_to_components.get(
                    (best_vertex, neighbor),
                    edge_to_components.get((neighbor, best_vertex), None),
                )
                # the edge is part of another component
                if comp_id not in component_ids:
                    continue
                # the component of the edge was already added
                if comp_id in added_components:
                    continue

                # we can add the component of the edge
                added_components.add(comp_id)
                target.append(comp_id)
                comp_added = True

                # Checks if the cycles can continue
                # subgraph is full
                if len(target) >= chunk_sizes[chunk_index]:
                    break

                # add new vertices to the used vertices so they can be used
                # in a next iteration
                new_vertices: Set[int] = {
                    v for e in component_to_edges[comp_id] for v in e
                }
                used_vertices |= new_vertices
                opened -= new_vertices

                # open neighbors of the newly added vertices
                for v in new_vertices:
                    for u in graph.neighbors(v):
                        if u in used_vertices:
                            continue
                        if u not in local_ordered_comp_ids:
                            continue
                        opened.add(u)

            if comp_added:
                iteration_no += 1
            else:
                opened.remove(best_vertex)

        # Nothing happened, we need to find some component randomly
        if iteration_no == 0:
            is_random_component_required[chunk_index] = True

        comp_graph.remove_nodes_from(added_components)
    return [v for group in ordered_comp_ids_groups for v in group]


def subgraphs_strategy_beam_neighbors_deprecated(
    comp_graph: nx.Graph,
    chunk_sizes: Sequence[int],
    start_with_triangles: bool,
    start_from_min: bool,
) -> List[int]:
    comp_graph = nx.Graph(comp_graph)
    ordered_comp_ids_groups: List[List[int]] = [[] for _ in chunk_sizes]
    beam_size: int = min(chunk_sizes[0], 10)

    while comp_graph.number_of_nodes() > 0:
        if start_from_min:
            start = min(comp_graph.degree(), key=lambda x: x[1])[0]
        else:
            start = max(comp_graph.degree(), key=lambda x: x[1])[0]

        if start not in comp_graph.nodes:
            continue

        queue: List[int] = [start]

        index_min = min(
            range(len(ordered_comp_ids_groups)),
            key=lambda x: len(ordered_comp_ids_groups[x]) / chunk_sizes[x],
        )
        target = ordered_comp_ids_groups[index_min]

        bfs_visited: Set[int] = set([start])
        added_comp_ids: Set[int] = set()

        # it's quite beneficial to start with a triangle
        # in fact we just apply the same strategy as later
        # just for the first vertex added as it has no context yet
        if start_with_triangles:
            start_neighbors = set(comp_graph.neighbors(start))
            for neighbor in start_neighbors:
                if (
                    len(start_neighbors.intersection(comp_graph.neighbors(neighbor)))
                    > 0
                ):
                    queue.append(neighbor)
                    bfs_visited.add(neighbor)
                    break

        while queue and len(target) < chunk_sizes[index_min]:
            values = [
                len(added_comp_ids.intersection(comp_graph.neighbors(u))) for u in queue
            ]

            sorted_by_metric = sorted(
                [i for i in range(len(values))],
                key=lambda i: values[i],
                reverse=True,
            )
            v = queue[sorted_by_metric[0]]
            queue = [queue[i] for i in sorted_by_metric[1 : beam_size + 1]]

            added_comp_ids.add(v)
            target.append(v)

            for u in comp_graph.neighbors(v):
                if u in bfs_visited:
                    continue
                bfs_visited.add(u)
                queue.append(u)

        comp_graph.remove_nodes_from(added_comp_ids)
    return [v for group in ordered_comp_ids_groups for v in group]


def subgraphs_strategy_components_deprecated(
    comp_graph: nx.Graph,
    chunk_sizes: Sequence[int],
    start_from_biggest_component: bool,
) -> List[int]:
    chunk_no = len(chunk_sizes)
    # NetworkX crashes otherwise
    if comp_graph.number_of_nodes() < 2:
        return list(comp_graph.nodes())

    k_components = nx.connectivity.k_components(comp_graph)

    if len(k_components) == 0:
        return list(comp_graph.nodes())

    keys = sorted(k_components.keys(), reverse=True)

    if not start_from_biggest_component:
        for i, key in enumerate(keys):
            if len(k_components[key]) <= chunk_no:
                keys = keys[i:]
                break

    ordered_comp_ids: List[int] = []
    used_comp_ids: Set[int] = set()

    for key in keys:
        for component in k_components[key]:
            for v in component:
                if v in used_comp_ids:
                    continue
                ordered_comp_ids.append(v)
                used_comp_ids.add(v)

    # make sure all the nodes were added
    for v in comp_graph.nodes():
        if v in used_comp_ids:
            continue
        ordered_comp_ids.append(v)

    return ordered_comp_ids


def subgraphs_strategy_kernighan_lin(
    comp_graph: nx.Graph,
    preferred_chunk_size: int,
    seed: int,
    weight_key: str | None = None,
) -> List[List[int]]:
    rand = random.Random(seed)

    def do_split(comp_subgraph: nx.Graph) -> List[List[int]]:
        if comp_subgraph.number_of_nodes() <= max(2, preferred_chunk_size * 3 // 2):
            return [list(comp_subgraph.nodes)]

        a, b = kernighan_lin_bisection(
            comp_subgraph,
            weight=weight_key,
            seed=rand.randint(0, 2**30),
        )

        # subgraphs may be empty
        if len(a) == 0 or len(b) == 0:
            return [list(a | b)]

        a = nx.induced_subgraph(comp_subgraph, a)
        b = nx.induced_subgraph(comp_subgraph, b)

        return do_split(a) + do_split(b)

    return do_split(comp_graph)


def subgraphs_strategy_cuts(
    comp_graph: nx.Graph,
    preferred_chunk_size: int,
    seed: int,
) -> List[List[int]]:
    rand = random.Random(seed)

    def do_split(comp_subgraph: nx.Graph) -> List[List[int]]:
        # required as induced subgraphs are frozen
        comp_subgraph = NiceGraph(comp_subgraph)

        if comp_subgraph.number_of_nodes() <= max(2, preferred_chunk_size * 3 // 2):
            return [list(comp_subgraph.nodes)]

        # choose all the vertices
        vert_deg = list(comp_subgraph.degree)
        first_deg = max(d for _, d in vert_deg)
        second_deg = max((d for _, d in vert_deg if d != first_deg), default=first_deg)
        perspective = [v for v, d in vert_deg if d == first_deg or d == second_deg]

        # n/2 iterations are fun as result of s->t and t->s may differ
        the_best_cut: Tuple[Set[int], Set[int]] = (set(comp_subgraph.nodes), set())
        for s in range(len(perspective)):
            for t in range(len(perspective)):
                if s == t:
                    continue
                cut_edges = nx.algorithms.connectivity.minimum_st_edge_cut(
                    comp_subgraph,
                    perspective[s],
                    perspective[t],
                )
                comp_subgraph.remove_edges_from(cut_edges)

                # there may be multiple if the graph is disconnected
                components = nx.connected_components(comp_subgraph)
                component = next(components)

                score = np.abs(2 * len(component) - comp_subgraph.number_of_nodes())
                curr_score = np.abs(len(the_best_cut[0]) - len(the_best_cut[1]))

                # choose the partitioning with the most balanced halves
                # and the randomness is also to super correct here...
                if score < curr_score or (
                    score == curr_score
                    and rand.randint(0, int(np.sqrt(len(perspective)))) == 0
                ):
                    the_best_cut = (
                        component,
                        set(v for comp in components for v in comp),
                    )
                comp_subgraph.add_edges_from(cut_edges)

        a, b = the_best_cut
        # subgraphs may be empty
        if len(a) == 0 or len(b) == 0:
            return [list(a | b)]

        a = nx.induced_subgraph(comp_subgraph, a)
        b = nx.induced_subgraph(comp_subgraph, b)

        return do_split(a) + do_split(b)

    return do_split(comp_graph)
