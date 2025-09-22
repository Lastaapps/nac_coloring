import itertools
import math
import random
from collections import defaultdict
from queue import PriorityQueue
from typing import *

import networkx as nx

import nac.check
from nac.core import (
    coloring_from_mask,
    create_bitmask_for_component_graph_cycle,
    mask_matches_templates,
    mask_to_vertices,
)
from nac.cycle_detection import find_cycles
from nac.data_type import Edge, NACColoring
from nac.development import (
    NAC_PRINT_SWITCH,
    NAC_statistics_colorings_merge,
    NAC_statistics_generator,
    graphviz_component_graph,
    graphviz_graph,
)
from nac.strategies_merging import (
    dynamic,
    linear,
    log,
    min_max,
    promising_cycles,
    recursion,
    score,
    shared_vertices,
    sorted_bits,
    sorted_size,
)
from nac.strategies_split import (
    degree_ordered_nodes,
    subgraphs_strategy_beam_neighbors_deprecated,
    subgraphs_strategy_bfs,
    subgraphs_strategy_components_deprecated,
    subgraphs_strategy_cuts,
    subgraphs_strategy_cycles,
    subgraphs_strategy_cycles_match_chunks,
    subgraphs_strategy_degree_cycles,
    subgraphs_strategy_kernighan_lin,
    subgraphs_strategy_neighbors,
)
from nac.util.repetable_iterator import RepeatableIterator
from nac.util.union_find import UnionFind


def NAC_colorings_naive(
    graph: nx.Graph,
    component_ids: List[int],
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
) -> Iterable[NACColoring]:
    """
    Naive implementation of the basic search algorithm
    """

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(component_ids) // 2):
        coloring = coloring_from_mask(
            component_ids,
            component_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


def NAC_colorings_cycles(
    graph: nx.Graph,
    components_ids: List[int],
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    from_angle_preserving_components: bool,
    use_all_cycles: bool = False,
) -> Iterable[NACColoring]:
    """
    Implementation of the naive algorithm improved by using cycles.
    """
    # so we start with 0
    components_ids.sort()

    # find some small cycles for state filtering
    cycles = find_cycles(
        graph,
        set(components_ids),
        component_to_edges,
        all=use_all_cycles,
    )
    # the idea is that smaller cycles reduce the state space more
    cycles = sorted(cycles, key=lambda c: len(c))

    templates = [
        create_bitmask_for_component_graph_cycle(
            graph, component_to_edges.__getitem__, c
        )
        for c in cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    # if len(cycles) != 0:
    #     templates_and_validities = [create_bitmask(c) for c in cycles]
    #     templates, validities = zip(*templates_and_validities)
    #     templates = np.stack(templates)
    #     validities = np.stack(validities)
    # else:
    #     templates = np.empty((0, len(components_ids)), dtype=np.bool)
    #     validities = np.empty((0, len(components_ids)), dtype=np.bool)

    # this is used for mask inversion, because how ~ works on python
    # numbers, if we used some kind of bit arrays,
    # this would not be needed.
    subgraph_mask = 0  # 2 ** len(components_ids) - 1
    for v in components_ids:
        subgraph_mask |= 1 << v
    # subgraph_mask = np.ones(len(components_ids), dtype=np.bool)
    # demasking = 2**np.arange(len(components_ids))

    # iterate all the coloring variants
    # division by 2 is used as the problem is symmetrical
    for mask in range(1, 2 ** len(components_ids) // 2):
        # This is part of a slower implementation using numpy
        # TODO remove before the final merge
        # my_mask = (demasking & mask).astype(np.bool)
        # def check_cycles(my_mask: np.ndarray) -> bool:
        #     # we mask the cycles
        #     masked = templates & my_mask
        #     usable_rows = np.sum(masked, axis=-1) == 1
        #     valid = masked & validities
        #     return bool(np.any(valid[usable_rows]))
        # if check_cycles(my_mask) or check_cycles(my_mask ^ subgraph_mask):
        #     continue

        if mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = coloring_from_mask(
            components_ids,
            component_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])


################################################################################
def NAC_colorings_without_vertex(
    processor: Callable[[nx.Graph], Iterable[NACColoring]],
    graph: nx.Graph,
    vertex: int | None,
) -> Iterable[NACColoring]:
    if graph.number_of_nodes() <= 1:
        return

    if vertex is None:
        vertex = min(graph.degree, key=lambda vd: vd[1])[0]

    subgraph = nx.Graph(graph)
    subgraph.remove_node(vertex)

    endpoints = list(graph.neighbors(vertex))
    # print()
    # print(Graph(graph))
    # print(Graph(subgraph))

    # TODO cache when the last coloring is opposite of the current one
    iterator = iter(
        NAC_colorings_with_non_surjective(
            subgraph,
            processor(subgraph),
        )
    )

    # In case there are no edges in the subgraph,
    # both the all red and blue components are empty
    if subgraph.number_of_edges() == 0:
        r, b = next(iterator)
        assert len(r) == 0
        assert len(b) == 0

    # in many of my algorithms I return coloring and it's symmetric
    # variant after each other. In that case local results are also symmetric
    previous_coloring: NACColoring | None = None
    previous_results: List[NACColoring] = []

    for coloring in iterator:
        if previous_coloring is not None:
            if (
                coloring[0] == previous_coloring[1]
                and coloring[1] == previous_coloring[0]
            ):
                for previous in previous_results:
                    yield previous[1], previous[0]
                previous_coloring = None
                continue
        previous_coloring = coloring
        previous_results = []

        # Naming explanation
        # vertex -> forbidden
        # endpoint -> for each edge (vertex, u), it's u
        # edge -> one of the edges incident to vertex
        # component -> connected component in the coloring given
        # group -> set of components
        #          if there are more edges/endpoints, that share the same color

        # print(f"{coloring=}")

        # create red & blue components
        def find_components_and_neighbors(
            red_edges: Collection[Edge],
            blue_edges: Collection[Edge],
        ) -> Tuple[Dict[int, int], List[Set[int]]]:
            sub = nx.Graph()
            sub.add_nodes_from(endpoints)
            sub.add_edges_from(red_edges)
            vertex_to_comp: Dict[int, int] = defaultdict(lambda: -1)
            id = -1
            for id, comp in enumerate(nx.connected_components(sub)):
                for v in comp:
                    vertex_to_comp[v] = id
            neighboring_components: List[Set[int]] = [set() for _ in range(id + 1)]
            for u, v in blue_edges:
                c1, c2 = vertex_to_comp[u], vertex_to_comp[v]
                if c1 != c2 and c1 != -1 and c2 != -1:
                    neighboring_components[c1].add(c2)
                    neighboring_components[c2].add(c1)
            return vertex_to_comp, neighboring_components

        red_endpoint_to_comp, red_neighboring_components = (
            find_components_and_neighbors(coloring[0], coloring[1])
        )
        blue_endpoint_to_comp_id, blue_neighboring_components = (
            find_components_and_neighbors(coloring[1], coloring[0])
        )
        endpoint_to_comp_id = (red_endpoint_to_comp, blue_endpoint_to_comp_id)
        neighboring_components = (
            red_neighboring_components,
            blue_neighboring_components,
        )
        # print(f"{endpoint_to_comp_id=}")
        # print(f"{neighboring_components=}")

        # find endpoints that are connected to the same component
        # we do this for both the red and blue, as this constrain has
        # to be fulfilled for both of them
        comp_to_endpoint: Tuple[Dict[int, int], Dict[int, int]] = ({}, {})
        same_groups = UnionFind()
        for u in endpoints:
            comp_id = endpoint_to_comp_id[0][u]
            if comp_id < 0:
                pass
            elif comp_id in comp_to_endpoint[0]:
                # if there is already a vertex for this component, we merge
                same_groups.join(comp_to_endpoint[0][comp_id], u)
            else:
                # otherwise we create a first record for the edge
                comp_to_endpoint[0][comp_id] = u

            # same as above, but for blue
            comp_id = endpoint_to_comp_id[1][u]
            if comp_id < 0:
                pass
            elif comp_id in comp_to_endpoint[1]:
                same_groups.join(comp_to_endpoint[1][comp_id], u)
            else:
                comp_to_endpoint[1][comp_id] = u

        # print(f"{same_groups=}")
        # print(f"{comp_to_endpoint=}")

        # Now we need to create an equivalent of component_to_edges
        # or to be precise vertex to edges
        # endpoint_to_same: List[List[int]] = [[] for _ in endpoints]
        endpoint_to_same: Dict[int, List[int]] = defaultdict(list)
        unique = set()
        for u in endpoints:
            id = same_groups.find(u)
            endpoint_to_same[id].append(u)
            unique.add(id)
        # print(f"{endpoint_to_same=}")

        # Now we need to remove empty components,
        # but keep relation to neighbors (update vertex_to_components)
        groups: List[List[int]] = []
        group_to_components: Tuple[List[List[int]], List[List[int]]] = (
            [],
            [],
        )  # holds component ids
        group_to_components_masks: Tuple[List[int], List[int]] = (
            [],
            [],
        )  # holds component ids

        for endpoints_in_same_group in endpoint_to_same.values():
            if len(endpoints_in_same_group) == 0:
                continue

            groups.append(endpoints_in_same_group)
            group_to_components[0].append(
                list(
                    filter(
                        lambda x: x >= 0,
                        (endpoint_to_comp_id[0][u] for u in endpoints_in_same_group),
                    )
                )
            )
            group_to_components[1].append(
                list(
                    filter(
                        lambda x: x >= 0,
                        (endpoint_to_comp_id[1][u] for u in endpoints_in_same_group),
                    )
                )
            )
            mask0, mask1 = 0, 0
            for u in endpoints_in_same_group:
                if u >= 0:
                    mask0 |= 1 << endpoint_to_comp_id[0][u]
                    mask1 |= 1 << endpoint_to_comp_id[1][u]
            group_to_components_masks[0].append(mask0)
            group_to_components_masks[1].append(mask1)

        # print(f"{groups=}")
        # print(f"{group_to_components=}")

        # Map everything to groups so we don't need to use indirection later
        groups_neighbors: Tuple[List[int], List[int]] = ([], [])
        for red_components, blue_components in zip(*group_to_components):
            # groups_neighbors[0].append(
            #     {n for comp_id in red_components for n in neighboring_components[0][comp_id]}
            # )
            # groups_neighbors[1].append(
            #     {n for comp_id in blue_components for n in neighboring_components[1][comp_id]}
            # )

            mask = 0
            for comp_id in red_components:
                for n in neighboring_components[0][comp_id]:
                    mask |= 1 << n
            groups_neighbors[0].append(mask)

            mask = 0
            for comp_id in blue_components:
                for n in neighboring_components[1][comp_id]:
                    mask |= 1 << n
            groups_neighbors[1].append(mask)

        # print(f"{groups_neighbors=}")

        # Check for contradictions
        # if there are in one component edges that share the same component
        # using one color, but use neighboring components using the other
        # color, we can exit now as all the coloring cannot pass
        # contradiction_found = False
        # TODO not working
        # for red_components, blue_components, red_neighbors, blue_neighbors in zip(
        #     *group_to_components, *groups_neighbors
        # ):
        #     if len(red_neighbors.intersection(red_components)) > 0:
        #         contradiction_found = True
        #     if len(blue_neighbors.intersection(blue_components)) > 0:
        #         contradiction_found = True
        # if contradiction_found:
        #     print("Contradiction found")
        #     break
        # TODO check if not all the components are neighboring

        groups_edges: List[List[Edge]] = [
            [(vertex, endpoint) for endpoint in group] for group in groups
        ]

        # Now we iterate all the possible colorings of groups
        for mask in range(2 ** len(groups)):
            # we create red and blue components lists
            # current_comps = ([], [])
            # for i in range(len(groups)):
            #     if mask & (1 << i):
            #         current_comps[0].extend(group_to_components[0][i])
            #     else:
            #         current_comps[1].extend(group_to_components[1][i])
            current_comps0, current_comps1 = (0, 0)
            for i in range(len(groups)):
                if mask & (1 << i):
                    current_comps0 |= group_to_components_masks[0][i]
                else:
                    current_comps1 |= group_to_components_masks[1][i]

            # no neighboring connected by the same color
            r = iter(range(len(groups)))
            i = next(r, None)
            valid = True
            while valid and i is not None:
                if mask & (1 << i):
                    if groups_neighbors[0][i] & current_comps0:
                        valid = False
                else:
                    if groups_neighbors[1][i] & current_comps1:
                        valid = False
                i = next(r, None)

            if not valid:
                continue

            current_edges: Tuple[List[Edge], List[Edge]] = ([], [])
            for i in range(len(groups)):
                k = int((mask & (1 << i)) == 0)
                current_edges[k].extend(groups_edges[i])

            to_emit = (
                coloring[0] + current_edges[0],
                coloring[1] + current_edges[1],
            )
            if len(to_emit[0]) * len(to_emit[1]) > 0:
                # print(f"{graph.number_of_nodes()} -> {to_emit=}")
                # assert nx.Graph(graph).is_NAC_coloring(to_emit)

                yield to_emit
                previous_results.append(to_emit)
        # print()


################################################################################
def NAC_colorings_with_non_surjective(
    comp: nx.Graph, colorings: Iterable[NACColoring]
) -> Iterable[NACColoring]:
    """
    This takes an iterator of NAC colorings and yields from it.
    Before it it sends non-surjective "NAC colorings"
    - all red and all blue coloring.
    """
    r, b = [], list(comp.edges())
    yield r, b
    yield b, r
    yield from colorings


################################################################################
def _smart_split(
    local_graph: nx.Graph,
    local_chunk_sizes: List[int],
    search_func: Callable[[nx.Graph, Sequence[int]], List[int]],
) -> List[int]:
    """
    # Smart split

    Ensures components that are later joined are next to each other
    increasing the chance of NAC colorings being dismissed sooner
    """
    if len(local_chunk_sizes) <= 2:
        return search_func(local_graph, local_chunk_sizes)

    length = len(local_chunk_sizes)
    sizes = (
        sum(local_chunk_sizes[: length // 2]),
        sum(local_chunk_sizes[length // 2 :]),
    )
    ordered_comp_ids = search_func(local_graph, sizes)
    groups = (ordered_comp_ids[: sizes[0]], ordered_comp_ids[sizes[0] :])
    graphs = tuple(nx.induced_subgraph(local_graph, g) for g in groups)
    assert len(graphs) == 2
    return _smart_split(
        graphs[0], local_chunk_sizes[: length // 2], search_func
    ) + _smart_split(graphs[1], local_chunk_sizes[length // 2 :], search_func)


def _subgraphs_join_epochs(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    from_angle_preserving_components: bool,
    ordered_comp_ids: List[int],
    epoch_1: Iterable[int],
    subgraph_mask_1: int,
    epoch_2: RepeatableIterator[int],
    subgraph_mask_2: int,
) -> Iterable[int]:
    """
    Joins almost NAC colorings of two disjoined subgraphs of a t-graph.
    This function works by doing a cross product of the almost NAC colorings
    on subgraphs, joining the subgraps into a single sub graph
    and checking if the joined colorings are still almost NAC.

    Almost NAC coloring is NAC coloring that is not necessarily surjective.

    Parameters
    ----------
    epoch_1
        iterator of almost NAC colorings of the first subgraph
        except for the symmetric ones.
    epoch_2
        iterator of almost NAC colorings of the second subgraph
        except for the symmetric ones.

    Returns
    -------
    Found almost NAC colorings and mask of the new subgraph.
    For each almost NAC coloring for each symmetric tuple, only one is yielded.
    """

    if subgraph_mask_1 & subgraph_mask_2:
        raise ValueError("Cannot join two subgraphs with common nodes")

    subgraph_mask = subgraph_mask_1 | subgraph_mask_2

    local_ordered_comp_ids: List[int] = []

    # local vertex -> global index
    mapping: Dict[int, int] = {}

    for i, v in enumerate(ordered_comp_ids):
        if (1 << i) & subgraph_mask:
            mapping[v] = i
            local_ordered_comp_ids.append(v)

    local_comp_graph = nx.Graph(nx.induced_subgraph(comp_graph, local_ordered_comp_ids))
    local_cycles = find_cycles(
        graph,
        set(local_comp_graph.nodes),
        component_to_edges,
    )

    mapped_components_to_edges = lambda ind: component_to_edges[ordered_comp_ids[ind]]
    # cycles with indices of the comp ids in the global order
    local_cycles = [tuple(mapping[c] for c in cycle) for cycle in local_cycles]
    templates = [
        create_bitmask_for_component_graph_cycle(
            graph, mapped_components_to_edges, cycle
        )
        for cycle in local_cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    counter = 0
    if NAC_PRINT_SWITCH:
        print(
            f"Join started ({2 ** subgraph_mask_1.bit_count()}+{2 ** subgraph_mask_2.bit_count()}->{2 ** subgraph_mask.bit_count()})"
        )

    # in case lazy_product is removed, return nested fors as they are faster
    # and also return repeatable iterator requirement
    mask_iterator = (
        (mask1, mask2 ^ mod)
        for mask1 in epoch_1
        for mask2 in epoch_2
        for mod in (0, subgraph_mask_2)
    )

    # this prolongs the overall computation time,
    # but in case we need just a "small" number of colorings,
    # this can provide them faster
    # disabled as result highly depended on the graph given
    # TODO benchmark on larger dataset
    # mask_iterator = lazy_product(epoch1, epoch2)

    for mask1, mask2 in mask_iterator:
        mask = mask1 | mask2

        if mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = coloring_from_mask(
            ordered_comp_ids,
            component_to_edges,
            mask,
            subgraph_mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        counter += 1
        yield mask

    if NAC_PRINT_SWITCH:
        print(f"Join yielded: {counter}")


@NAC_statistics_generator
def _subgraph_colorings_generator(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    ordered_comp_ids: List[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    match 0:
        case 0:
            return _subgraph_colorings_cycles_generator(
                graph,
                comp_graph,
                component_to_edges,
                is_NAC_coloring_routine,
                ordered_comp_ids,
                chunk_size,
                offset,
            )
        case 1:
            # TODO implement _NAC_CHECK_CYCLE_MASK counting
            assert False
            return _subgraph_colorings_removal_generator(
                graph,
                comp_graph,
                component_to_edges,
                is_NAC_coloring_routine,
                ordered_comp_ids,
                chunk_size,
                offset,
            )


@NAC_statistics_generator
def _subgraph_colorings_cycles_generator(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    ordered_comp_ids: List[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    """
    iterate all the coloring variants
    division by 2 is used as the problem is symmetrical
    """
    # The last chunk can be smaller
    local_ordered_comp_ids: List[int] = ordered_comp_ids[offset : offset + chunk_size]
    # print(f"Local comp_ids: {local_ordered_comp_ids}")

    local_comp_graph = nx.Graph(nx.induced_subgraph(comp_graph, local_ordered_comp_ids))
    local_cycles = find_cycles(
        graph,
        set(local_comp_graph.nodes),
        component_to_edges,
    )

    # local -> first chunk_size vertices
    mapping = {x: i for i, x in enumerate(local_ordered_comp_ids)}

    mapped_components_to_edges = lambda ind: component_to_edges[
        local_ordered_comp_ids[ind]
    ]
    local_cycles = (tuple(mapping[c] for c in cycle) for cycle in local_cycles)
    templates = [
        create_bitmask_for_component_graph_cycle(
            graph, mapped_components_to_edges, cycle
        )
        for cycle in local_cycles
    ]
    templates = [t for t in templates if t[1] > 0]

    counter = 0
    subgraph_mask = 2 ** len(local_ordered_comp_ids) - 1
    for mask in range(0, 2**chunk_size // 2):
        if mask_matches_templates(templates, mask, subgraph_mask):
            continue

        coloring = coloring_from_mask(
            local_ordered_comp_ids,
            component_to_edges,
            mask,
        )

        if not is_NAC_coloring_routine(graph, coloring):
            continue

        counter += 1
        yield mask << offset

    if NAC_PRINT_SWITCH:
        print(f"Base yielded: {counter}")


@NAC_statistics_generator
def _subgraph_colorings_removal_generator(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    partitions: List[int],
    chunk_size: int,
    offset: int,
) -> Iterable[int]:
    """
    iterate all the coloring variants
    division by 2 is used as the problem is symmetrical
    """

    # TODO assert for cartesian NAC coloring

    local_partitions: List[int] = partitions[offset : offset + chunk_size]
    local_edges = [
        edge for comp_id in local_partitions for edge in component_to_edges[comp_id]
    ]
    graph = nx.edge_subgraph(graph, local_edges)

    def processor(graph: nx.Graph) -> Iterable[NACColoring]:
        # print(f"graph={Graph(graph)}")
        # return list(_NAC_colorings_without_vertex(processor, graph, None))
        return NAC_colorings_without_vertex(processor, graph, None)

    nodes_iter = iter(local_partitions)
    # This makes sure no symmetric colorings are produced
    disabled_comp = next(nodes_iter)
    edge_map = defaultdict(int)
    disabled_edge = component_to_edges[disabled_comp][0]
    edge_map[(disabled_edge[0], disabled_edge[1])] = len(component_to_edges) + 1
    edge_map[(disabled_edge[1], disabled_edge[0])] = len(component_to_edges) + 1

    for comp_id in nodes_iter:
        edge = component_to_edges[comp_id][0]
        edge_map[(edge[0], edge[1])] = local_partitions.index(comp_id) + 1
        edge_map[(edge[1], edge[0])] = local_partitions.index(comp_id) + 1

    # print()
    # print(f"{component_to_edges=}")
    # print(f"{partitions=} {offset=} {chunk_size=}")
    # print(f"{edge_map=}")
    # produced = set()
    # print("Accepted", bin(0))
    # all_ones = 2**len(local_partitions) - 1

    counter = 0
    yield 0
    for red, blue in processor(graph):
        # TODO swith red and blue if blue is smaller
        # print(f"{red=} {blue=}")
        mask = 0
        invalid_coloring_found = False
        for edge in red:
            partition_ind = edge_map[edge]
            if partition_ind == 0:
                continue
            mask |= 1 << partition_ind
            partition_ind -= 1
            partition_ind = (partition_ind < len(local_partitions)) * partition_ind

            # make sure all the edges of each partition share the same component
            partition_id = local_partitions[partition_ind]
            for partiotion_edge in component_to_edges[partition_id]:
                if (
                    partiotion_edge not in red
                    and (partiotion_edge[1], partiotion_edge[0]) not in red
                ):
                    invalid_coloring_found = True
                    # print(f"Invalid red  coloring {partition_ind=}({edge_map[edge]-1}) -> {partition_id=}")
                    break
            if invalid_coloring_found:
                break

        for edge in blue:
            partition_ind = edge_map[edge]
            if partition_ind == 0:
                continue
            partition_ind -= 1
            partition_ind = (partition_ind < len(local_partitions)) * partition_ind

            # make sure all the edges of each partition share the same component
            partition_id = local_partitions[partition_ind]
            for partiotion_edge in component_to_edges[partition_id]:
                if (
                    partiotion_edge not in blue
                    and (partiotion_edge[1], partiotion_edge[0]) not in blue
                ):
                    invalid_coloring_found = True
                    # print(f"Invalid blue coloring {partition_ind=}({edge_map[edge]-1}) -> {partition_id=}")
                    break
            if invalid_coloring_found:
                break
        if invalid_coloring_found:
            continue

        mask >>= 1
        if mask & (1 << len(component_to_edges)):
            continue
        counter += 1
        mask <<= offset

        # print("Accepted", bin(mask))
        # coloring = coloring_from_mask(
        #     local_partitions,
        #     component_to_edges,
        #     mask >> offset,
        # )
        # print(f"{coloring=}")
        # if mask in produced:
        #     assert False
        # produced.add(mask)

        yield mask

    if NAC_PRINT_SWITCH:
        print(f"Base yielded: {counter}")


def _apply_split_strategy_to_order_vertices(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    chunk_sizes: List[int],
    preferred_chunk_size: int,
    order_strategy: str,
    use_smart_split: bool,
    seed: int,
) -> Tuple[List[int], List[int]]:
    def process(
        search_func: Callable[[nx.Graph, Sequence[int]], List[int]],
    ):
        if use_smart_split:
            return _smart_split(
                comp_graph,
                chunk_sizes,
                search_func,
            )
        else:
            return search_func(comp_graph, chunk_sizes)

    match order_strategy:
        case "none":
            ordered_comp_ids = list(comp_graph.nodes())

        case "random":
            ordered_comp_ids = list(comp_graph.nodes())
            random.Random(seed).shuffle(ordered_comp_ids)

        case "degree":
            ordered_comp_ids = process(lambda g, _: degree_ordered_nodes(g))

        case "degree_cycles":
            ordered_comp_ids = process(
                lambda g, _: subgraphs_strategy_degree_cycles(
                    graph,
                    g,
                    component_to_edges,
                )
            )
        case "cycles":
            ordered_comp_ids = process(
                lambda g, _: subgraphs_strategy_cycles(
                    graph,
                    g,
                    component_to_edges,
                )
            )
        case "cycles_match_chunks":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_cycles_match_chunks(
                    l,
                    graph,
                    g,
                    component_to_edges,
                )
            )
        case "bfs":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_bfs(
                    comp_graph=g,
                    chunk_sizes=l,
                )
            )
        case "neighbors":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_neighbors(
                    graph=graph,
                    comp_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=False,
                    iterative=False,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "neighbors_cycle" | "neighbors_cycles":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_neighbors(
                    graph=graph,
                    comp_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=True,
                    iterative=False,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "neighbors_degree":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_neighbors(
                    graph=graph,
                    comp_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=False,
                    iterative=False,
                    use_degree=True,
                    seed=seed,
                )
            )
        case "neighbors_degree_cycle":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_neighbors(
                    graph=graph,
                    comp_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=True,
                    iterative=False,
                    use_degree=True,
                    seed=seed,
                )
            )
        case "neighbors_iterative":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_neighbors(
                    graph=graph,
                    comp_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=False,
                    iterative=True,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "neighbors_iterative_cycle":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_neighbors(
                    graph=graph,
                    comp_graph=g,
                    component_to_edges=component_to_edges,
                    chunk_sizes=l,
                    start_with_cycle=True,
                    iterative=True,
                    use_degree=False,
                    seed=seed,
                )
            )
        case "beam_neighbors":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_beam_neighbors_deprecated(
                    comp_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=False,
                    start_from_min=True,
                )
            )
        case "beam_neighbors_max":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_beam_neighbors_deprecated(
                    comp_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=False,
                    start_from_min=False,
                )
            )
        case "beam_neighbors_triangles":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_beam_neighbors_deprecated(
                    comp_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=True,
                    start_from_min=True,
                )
            )
        case "beam_neighbors_max_triangles":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_beam_neighbors_deprecated(
                    comp_graph=g,
                    chunk_sizes=l,
                    start_with_triangles=True,
                    start_from_min=False,
                )
            )
        case "components_biggest":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_components_deprecated(
                    g, l, start_from_biggest_component=True
                )
            )
        case "components_spredded":
            ordered_comp_ids = process(
                lambda g, l: subgraphs_strategy_components_deprecated(
                    g, l, start_from_biggest_component=False
                )
            )
        case "kernighan_lin":
            subgraph_classes = subgraphs_strategy_kernighan_lin(
                comp_graph=comp_graph,
                preferred_chunk_size=preferred_chunk_size,
                seed=seed,
            )
            # TODO refactor later
            chunk_sizes = [len(subgraph) for subgraph in subgraph_classes]
            ordered_comp_ids = [v for subgraph in subgraph_classes for v in subgraph]
        case "cuts":
            subgraph_classes = subgraphs_strategy_cuts(
                comp_graph=comp_graph,
                preferred_chunk_size=preferred_chunk_size,
                seed=seed,
            )
            # TODO refactor later
            chunk_sizes = [len(subgraph) for subgraph in subgraph_classes]
            ordered_comp_ids = [v for subgraph in subgraph_classes for v in subgraph]
        case _:
            raise ValueError(
                f"Unknown strategy: {order_strategy}, supported: none, degree, degree_cycles, cycles, cycles_match_chunks, bfs, beam_neighbors, components_biggest, components_spredded"
            )
    return ordered_comp_ids, chunk_sizes


################################################################################
@NAC_statistics_colorings_merge
def _colorings_merge(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    from_angle_preserving_components: bool,
    ordered_comp_ids: List[int],
    colorings_1: Tuple[Iterable[int], int],
    colorings_2: Tuple[Iterable[int], int],
) -> Tuple[Iterable[int], int]:
    nac.check._NAC_MERGE += 1

    (epoch1, subgraph_mask_1) = colorings_1
    (epoch2, subgraph_mask_2) = colorings_2
    epoch1 = RepeatableIterator(epoch1)
    epoch2 = RepeatableIterator(epoch2)

    vertices_1 = mask_to_vertices(ordered_comp_ids, component_to_edges, colorings_1[1])
    vertices_2 = mask_to_vertices(ordered_comp_ids, component_to_edges, colorings_2[1])

    if len(vertices_1.intersection(vertices_2)) <= 1:
        nac.check._NAC_MERGE_NO_COMMON_VERTEX += 1

        def generator() -> Iterator[int]:
            for c1 in epoch1:
                for c2 in epoch2:
                    yield c1 | c2
                    yield c1 | c2 ^ subgraph_mask_2  # switched colors

        return (
            generator(),
            subgraph_mask_1 | subgraph_mask_2,
        )

    # if at least two vertices are shared, we need to do the full check
    return (
        _subgraphs_join_epochs(
            graph,
            comp_graph,
            component_to_edges,
            is_NAC_coloring_routine,
            from_angle_preserving_components,
            ordered_comp_ids,
            epoch1,
            subgraph_mask_1,
            epoch2,
            subgraph_mask_2,
        ),
        subgraph_mask_1 | subgraph_mask_2,
    )


def _apply_merge_strategy(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    from_angle_preserving_components: bool,
    ordered_comp_ids: List[int],
    merge_strategy: str,
    all_epochs: List[Tuple[Iterable[int], int]],
) -> List[Tuple[Iterable[int], int]]:
    def colorings_merge_wrapper(
        colorings_1: Tuple[Iterable[int], int],
        colorings_2: Tuple[Iterable[int], int],
    ) -> Tuple[Iterable[int], int]:
        return _colorings_merge(
            graph=graph,
            comp_graph=comp_graph,
            component_to_edges=component_to_edges,
            is_NAC_coloring_routine=is_NAC_coloring_routine,
            from_angle_preserving_components=from_angle_preserving_components,
            ordered_comp_ids=ordered_comp_ids,
            colorings_1=colorings_1,
            colorings_2=colorings_2,
        )

    match merge_strategy:
        case "linear":
            return linear(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case "sorted_bits":
            return sorted_bits(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case "sorted_size":
            return sorted_size(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case "log" | "log_reverse":
            return log(
                is_reversed=merge_strategy == "log_reverse",
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )

        case "min_max":
            return min_max(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case "score":
            return score(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )

        case "dynamic":
            return dynamic(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )

        case "recursion":
            return recursion(
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )

        case "shared_vertices":
            return shared_vertices(
                component_to_edges=component_to_edges,
                ordered_comp_ids=ordered_comp_ids,
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )

        case "promising_cycles":
            return promising_cycles(
                component_to_edges=component_to_edges,
                ordered_comp_ids=ordered_comp_ids,
                colorings_merge_wrapper=colorings_merge_wrapper,
                all_epochs=all_epochs,
            )
        case _:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")


################################################################################
def NAC_colorings_subgraphs(
    graph: nx.Graph,
    comp_graph: nx.Graph,
    component_to_edges: List[List[Edge]],
    is_NAC_coloring_routine: Callable[[nx.Graph, NACColoring], bool],
    seed: int,
    from_angle_preserving_components: bool,
    use_smart_split: bool = True,
    merge_strategy: (
        str | Literal["linear", "log", "log_reverse", "min_max", "weight"]
    ) = "linear",
    preferred_chunk_size: int | None = None,
    order_strategy: str = "neighbors_degree",
) -> Iterable[NACColoring]:
    """
    This version of the algorithm splits the graphs into subgraphs,
    find NAC colorings for each of them. The subgraphs are then merged
    and new colorings are reevaluated till we reach the original graph again.
    The algorithm tries to find optimal subgraphs and merge strategy.
    """
    rand = random.Random(seed)

    # These values are taken from benchmarks as (almost) optimal
    if preferred_chunk_size is None:
        preferred_chunk_size = 5 if comp_graph.number_of_nodes() < 12 else 6

    preferred_chunk_size = min(preferred_chunk_size, comp_graph.number_of_nodes())
    assert preferred_chunk_size >= 1

    # Represents size (no. of vertices (components) of the t-graph) of a basic subgraph
    components_no = comp_graph.number_of_nodes()

    def create_chunk_sizes() -> List[int]:
        """
        Makes sure all the chunks are the same size of 1 bigger

        Could be probably significantly simpler,
        like np.fill and add some bitmask, but this was my first idea,
        get over it.
        """
        # chunk_size = max(
        #     int(np.sqrt(components_no)), min(preferred_chunk_size, components_no)
        # )
        # chunk_no = (components_no + chunk_size - 1) // chunk_size
        chunk_no = components_no // preferred_chunk_size
        chunk_sizes = []
        remaining_len = components_no
        for _ in range(chunk_no):
            # ceiling floats, scary
            chunk_sizes.append(
                min(
                    math.ceil(remaining_len / (chunk_no - len(chunk_sizes))),
                    remaining_len,
                )
            )
            remaining_len -= chunk_sizes[-1]
        return chunk_sizes

    chunk_sizes = create_chunk_sizes()

    ordered_comp_ids, chunk_sizes = _apply_split_strategy_to_order_vertices(
        graph=graph,
        comp_graph=comp_graph,
        component_to_edges=component_to_edges,
        chunk_sizes=chunk_sizes,
        preferred_chunk_size=preferred_chunk_size,
        order_strategy=order_strategy,
        use_smart_split=use_smart_split,
        seed=rand.randint(0, 2**30),
    )

    assert components_no == len(ordered_comp_ids)

    if NAC_PRINT_SWITCH:
        print("-" * 80)
        print(
            graphviz_graph(
                order_strategy, component_to_edges, chunk_sizes, ordered_comp_ids
            )
        )
        print("-" * 80)
        print(
            graphviz_component_graph(
                comp_graph,
                order_strategy,
                component_to_edges,
                chunk_sizes,
                ordered_comp_ids,
            )
        )

        print("-" * 80)
        print(f"Vertices no:  {nx.number_of_nodes(graph)}")
        print(f"Edges no:     {nx.number_of_edges(graph)}")
        print(f"T-graph size: {nx.number_of_nodes(comp_graph)}")
        print(f"Comp. to ed.: {component_to_edges}")
        print(f"Chunk no.:    {len(chunk_sizes)}")
        print(f"Chunk sizes:  {chunk_sizes}")
        print("-" * 80)

    # Holds all the NAC colorings for a subgraph represented by the second bitmask
    all_epochs: List[Tuple[Iterable[int], int]] = []
    # No. of components already processed in previous chunks
    offset = 0
    for chunk_size in chunk_sizes:
        subgraph_mask = 2**chunk_size - 1
        all_epochs.append(
            (
                _subgraph_colorings_generator(
                    graph,
                    comp_graph,
                    component_to_edges,
                    is_NAC_coloring_routine,
                    ordered_comp_ids,
                    chunk_size,
                    offset,
                ),
                subgraph_mask << offset,
            )
        )
        offset += chunk_size

    all_epochs = _apply_merge_strategy(
        graph=graph,
        comp_graph=comp_graph,
        component_to_edges=component_to_edges,
        is_NAC_coloring_routine=is_NAC_coloring_routine,
        from_angle_preserving_components=from_angle_preserving_components,
        ordered_comp_ids=ordered_comp_ids,
        merge_strategy=merge_strategy,
        all_epochs=all_epochs,
    )

    assert len(all_epochs) == 1
    expected_subgraph_mask = 2**components_no - 1
    assert expected_subgraph_mask == all_epochs[0][1]

    for mask in all_epochs[0][0]:
        # print(f"Got mask={bin(mask)}")
        if mask == 0 or mask.bit_count() == len(ordered_comp_ids):
            continue

        coloring = coloring_from_mask(
            ordered_comp_ids,
            component_to_edges,
            mask,
        )

        yield (coloring[0], coloring[1])
        yield (coloring[1], coloring[0])
