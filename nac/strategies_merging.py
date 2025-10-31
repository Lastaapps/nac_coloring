from queue import PriorityQueue
import itertools
from typing import *

import networkx as nx

from nac.core import mask_to_graph, mask_to_vertices
from nac.data_type import Edge

Epoch: TypeAlias = Tuple[Iterable[int], int]


def linear(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    res: Tuple[Iterable[int], int] = all_epochs[0]
    for g in all_epochs[1:]:
        res = colorings_merge_wrapper(res, g)
    return [res]


def sorted_bits(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    """
    Similar to linear, but sorts the subgraphs by size first
    """
    all_epochs = sorted(all_epochs, key=lambda x: x[1].bit_count(), reverse=True)
    while len(all_epochs) > 1:
        iterable, mask = colorings_merge_wrapper(
            all_epochs[-1],
            all_epochs[-2],
        )
        all_epochs.pop()
        all_epochs[-1] = (iterable, mask)
    return all_epochs


def sorted_size(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    """
    Sorts the subgraphs by number of their NAC colorings
    and merges them linearly from the smallest to the biggest
    """
    # Joins the subgraphs like a tree
    all_epochs: List[Tuple[List[int], int]] = [(list(i), m) for i, m in all_epochs]

    all_epochs = sorted(all_epochs, key=lambda x: len(x[0]), reverse=True)

    while len(all_epochs) > 1:
        iterable, mask = colorings_merge_wrapper(
            all_epochs[-1],
            all_epochs[-2],
        )
        all_epochs.pop()
        all_epochs[-1] = (list(iterable), mask)
    return all_epochs


def log(
    is_reversed: bool,
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    # Joins the subgraphs like a tree
    while len(all_epochs) > 1:
        next_all_epochs: List[Tuple[Iterable[int], int]] = []

        # always join 2 subgraphs
        for batch in itertools.batched(all_epochs, 2):
            if len(batch) == 1:
                next_all_epochs.append(batch[0])
                continue

            next_all_epochs.append(colorings_merge_wrapper(*batch))

        if is_reversed:
            next_all_epochs = list(reversed(next_all_epochs))

        all_epochs = next_all_epochs
    return all_epochs


def min_max(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    """
    Joins smallest and largest subgraphs first
    """
    # Joins the subgraphs like a tree
    all_epochs: List[Tuple[List[int], int]] = [(list(i), m) for i, m in all_epochs]
    while len(all_epochs) > 1:
        next_all_epochs: List[Tuple[List[int], int]] = []

        all_epochs = sorted(all_epochs, key=lambda x: x[1].bit_count())

        for batch_id in range(len(all_epochs) // 2):
            iterable, mask = colorings_merge_wrapper(
                all_epochs[batch_id],
                all_epochs[-(batch_id + 1)],
            )
            next_all_epochs.append((list(iterable), mask))

        if len(all_epochs) % 2 == 1:
            next_all_epochs.append(all_epochs[len(all_epochs) // 2])

        all_epochs = next_all_epochs
    return all_epochs


def score(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    """
    This approach forbids the online version of the algorithm

    Iterations are run until the original graph is restored.
    In each iteration a score is computed and the pair with
    the best score is chosen.
    Score tries to mimic the work required to join the subgraphs.
    Which is the product of # of colorings on both the subgraphs
    times the size of the resulting subgraph.
    """
    all_epochs: List[Tuple[List[int], int]] = [(list(i), m) for i, m in all_epochs]
    while len(all_epochs) > 1:
        best_pair = (0, 1)
        lowers_score = 9_223_372_036_854_775_807
        for i in range(len(all_epochs)):
            e1 = all_epochs[i]
            for j in range(i + 1, len(all_epochs)):
                e2 = all_epochs[j]
                # size of colorings product * size of graph
                score = len(e1[0]) * len(e2[0]) * (e1[1] | e2[1]).bit_count()
                if score < lowers_score:
                    lowers_score = score
                    best_pair = (i, j)

        e1 = all_epochs[best_pair[0]]
        e2 = all_epochs[best_pair[1]]
        iterable, mask = colorings_merge_wrapper(
            all_epochs[best_pair[0]],
            all_epochs[best_pair[1]],
        )
        # this is safe (and slow) as the second coordinate is greater
        all_epochs.pop(best_pair[1])
        all_epochs[best_pair[0]] = (list(iterable), mask)
    return all_epochs


def dynamic(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    # Joins the subgraphs like a tree
    all_epochs: List[Tuple[List[int], int]] = [(list(i), m) for i, m in all_epochs]

    all_ones = 2 ** len(all_epochs) - 1
    cache: List[None | Tuple[int, int, int, Tuple[int, int]]] = [
        None for _ in range(2 ** len(all_epochs))
    ]
    cache[all_ones] = (0, 0, 0, (0, 1))

    def dynamic_process_search(
        mask: int,
    ) -> Tuple[int, int, int, Tuple[int, int]]:
        if cache[mask] is not None:
            return cache[mask]

        if mask.bit_count() + 1 == len(all_epochs):
            index = (mask ^ all_ones).bit_length() - 1
            a, b = all_epochs[index]
            return 0, len(a), b.bit_count(), (index, index)

        least_work = 9_223_372_036_854_775_807**2
        result: Tuple[int, int, int, Tuple[int, int]]

        for i in range(len(all_epochs)):
            if mask & (1 << i):
                continue

            e1 = all_epochs[i]
            for j in range(i + 1, len(all_epochs)):
                if mask & (1 << j):
                    continue

                e2 = all_epochs[j]
                # size of colorings product * size of graph
                my_size = (e1[1] | e2[1]).bit_count()
                my_work = len(e1[0]) * len(e2[0]) * my_size
                my_outputs = len(e1[0]) * len(e2[0])  # TODO heuristics
                mask ^= (1 << i) | (1 << j)
                work, outputs, size, _ = dynamic_process_search(mask)
                final_size = my_size + size
                final_output = my_outputs * outputs
                final_work = work + my_work + final_output * final_size
                mask ^= (1 << i) | (1 << j)
                if final_work < least_work:
                    least_work = final_work
                    result = (final_work, final_output, final_size, (i, j))
        cache[mask] = result
        return result

    algo_mask = 0
    while algo_mask != all_ones:
        _, _, _, best_pair = dynamic_process_search(algo_mask)

        if best_pair[0] == best_pair[1]:
            all_epochs = [all_epochs[best_pair[0]]]
            break

        algo_mask |= 1 << best_pair[1]

        e1 = all_epochs[best_pair[0]]
        e2 = all_epochs[best_pair[1]]
        iterable, mask = colorings_merge_wrapper(
            all_epochs[best_pair[0]],
            all_epochs[best_pair[1]],
        )
        # this is safe (and slow) as the second coordinate is greater
        all_epochs[best_pair[0]] = (list(iterable), mask)
    return all_epochs


def recursion(
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    # Joins the subgraphs like a tree
    all_epochs: List[Tuple[List[int], int]] = [(list(i), m) for i, m in all_epochs]

    all_ones = 2 ** len(all_epochs) - 1
    algo_mask = 0

    cache: List[None | Tuple[int, int, int, Tuple[int, int]]] = [
        None for _ in range(2 ** len(all_epochs))
    ]
    cache[all_ones] = (0, 0, 0, (0, 1))

    while algo_mask != all_ones:

        def dynamic_process_search(
            mask: int,
        ) -> Tuple[int, int, int, Tuple[int, int]]:
            if cache[mask] is not None:
                return cache[mask]

            if mask.bit_count() + 1 == len(all_epochs):
                index = (mask ^ all_ones).bit_length() - 1
                a, b = all_epochs[index]
                return 0, len(a), b.bit_count(), (index, index)

            least_work = 9_223_372_036_854_775_807**2
            result: Tuple[int, int, int, Tuple[int, int]]

            for i in range(len(all_epochs)):
                if mask & (1 << i):
                    continue

                e1 = all_epochs[i]
                for j in range(i + 1, len(all_epochs)):
                    if mask & (1 << j):
                        continue

                    e2 = all_epochs[j]
                    # size of colorings product * size of graph
                    my_size = (e1[1] | e2[1]).bit_count()
                    my_work = len(e1[0]) * len(e2[0]) * my_size
                    my_outputs = len(e1[0]) * len(e2[0])  # TODO heuristics
                    mask ^= (1 << i) | (1 << j)
                    work, outputs, size, _ = dynamic_process_search(mask)
                    final_size = my_size + size
                    final_output = my_outputs * outputs
                    final_work = work + my_work + final_output * final_size
                    mask ^= (1 << i) | (1 << j)
                    if final_work < least_work:
                        least_work = final_work
                        result = (
                            final_work,
                            final_output,
                            final_size,
                            (i, j),
                        )
            cache[mask] = result
            return result

        # actually, the whole cache should be invalidated
        def invalidate_cache(
            cache: List[None | Tuple[int, int, int, Tuple[int, int]]],
            bit: int,
        ) -> List[None | Tuple[int, int, int, Tuple[int, int]]]:
            cache = [None for _ in range(2 ** len(all_epochs))]
            cache[all_ones] = (0, 0, 0, (0, 1))
            return cache

        _, _, _, best_pair = dynamic_process_search(algo_mask)

        if best_pair[0] == best_pair[1]:
            all_epochs = [all_epochs[best_pair[0]]]
            break

        algo_mask |= 1 << best_pair[1]
        cache = invalidate_cache(cache, best_pair[0])

        e1 = all_epochs[best_pair[0]]
        e2 = all_epochs[best_pair[1]]
        iterable, mask = colorings_merge_wrapper(
            all_epochs[best_pair[0]],
            all_epochs[best_pair[1]],
        )
        # this is safe (and slow) as the second coordinate is greater
        all_epochs[best_pair[0]] = (list(iterable), mask)
    return all_epochs


def shared_vertices(
    component_to_edges: List[List[Edge]],
    ordered_comp_ids: List[int],
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    while len(all_epochs) > 1:
        best = (0, 0, 1)
        subgraph_vertices: List[Set[int]] = [
            mask_to_vertices(ordered_comp_ids, component_to_edges, mask)
            for _, mask in all_epochs
        ]
        for i in range(0, len(subgraph_vertices)):
            for j in range(i + 1, len(subgraph_vertices)):
                vert1 = subgraph_vertices[i]
                vert2 = subgraph_vertices[j]
                vertex_no = len(vert1.intersection(vert2))
                if vertex_no > best[0]:
                    best = (vertex_no, i, j)
        res = colorings_merge_wrapper(all_epochs[best[1]], all_epochs[best[2]])

        all_epochs[best[1]] = res
        all_epochs.pop(best[2])
    return all_epochs


def promising_cycles(
    component_to_edges: List[List[Edge]],
    ordered_comp_ids: List[int],
    colorings_merge_wrapper: Callable[[Epoch, Epoch], Epoch],
    all_epochs: List[Epoch],
) -> List[Epoch]:
    """
    This strategy looks for edges that when added
    to a subgraph can cause a cycle.
    We try to maximize the number of cycles.
    If the number of cycles match, we try the following strategies:
    - if components share a vertex, they get higher priority
    - graph pairs with lower number of monochromatic components get higher priority
    - the number of common vertices
    - unspecified
    """

    def mask_to_edges_and_components(
        mask: int,
    ) -> Tuple[List[Edge], List[Set[int]]]:
        """
        Creates a graph for the monochromatic classes given by the mask
        """
        graph = mask_to_graph(ordered_comp_ids, component_to_edges, mask)
        return list(graph.edges), list(nx.connected_components(graph))

    def find_potential_cycles(edges: List[Edge], components: List[Set[int]]) -> int:
        """
        Counts the number of edges that may create an almost cycle
        in the other subgraph.
        """
        cycles: int = 0
        for u, v in edges:
            for comp in components:
                u_in_comp = u in comp
                v_in_comp = v in comp
                cycles += u_in_comp and v_in_comp

        return cycles

    def find_shared_vertices(
        components1: List[Set[int]], components2: List[Set[int]]
    ) -> int:
        """
        Counts the number of edges that may create an almost cycle
        in the other subgraph.
        """
        if sum(len(c) for c in components1) < sum(len(c) for c in components2):
            components1, components2 = components2, components1

        count: int = 0
        for s1 in components1:
            for u in s1:
                for s2 in components2:
                    if u in s2:
                        count += 1
                        break

        return count

    # maps from subgraph (mask) to it's decomposition
    mapped_graphs: Dict[int, Tuple[List[Edge], List[Set[int]]]] = {}

    # set merged graphs
    class QueueItem(NamedTuple):
        score: Tuple[int, int, int, int]
        mask_1: int
        mask_2: int

    queue: PriorityQueue[QueueItem] = PriorityQueue()

    def on_new_graph(
        mask: int,
        graph_props: Tuple[List[Edge], List[Set[int]]],
        checks_limit: int | None = None,
    ):
        """
        Schedules new events in the priority queue for a new graph
        """
        if checks_limit is None:
            checks_limit = len(all_epochs)
        for existing in all_epochs[:checks_limit]:
            other_mask = existing[1]
            other_props = mapped_graphs[other_mask]

            cycles1 = find_potential_cycles(graph_props[0], other_props[1])
            cycles2 = find_potential_cycles(other_props[0], graph_props[1])

            shared_vertices = find_shared_vertices(graph_props[1], other_props[1])

            # +1 minimize, -1 maximize
            score = (
                -1 * (cycles1 + cycles2),
                -1 * (shared_vertices > 0),
                +1 * (mask.bit_count() + other_mask.bit_count()),
                -1 * shared_vertices,
            )
            queue.put(QueueItem(score, existing[1], mask))
            assert other_mask != mask

    # add base subgraphs to the related data structures
    for i, mask in enumerate(all_epochs):
        _, mask = mask

        # create a graph representation
        graph_props = mask_to_edges_and_components(mask)
        mapped_graphs[mask] = graph_props

        on_new_graph(mask, graph_props, checks_limit=i)

    while not queue.empty():
        _, mask1, mask2 = queue.get()
        # skip subgraphs that are already handled
        if mask1 not in mapped_graphs or mask2 not in mapped_graphs:
            continue

        # find indices in all_epochs and apply
        ind1: int = next(filter(lambda x: x[1][1] == mask1, enumerate(all_epochs)))[0]
        ind2: int = next(filter(lambda x: x[1][1] == mask2, enumerate(all_epochs)))[0]
        res = colorings_merge_wrapper(all_epochs[ind1], all_epochs[ind2])

        # register the new graph
        mask = res[1]
        graph_props = mask_to_edges_and_components(mask)
        mapped_graphs[mask] = graph_props
        on_new_graph(mask, graph_props)

        # remove old graphs
        mapped_graphs.pop(all_epochs[ind1][1])
        mapped_graphs.pop(all_epochs[ind2][1])
        all_epochs[ind1] = res
        all_epochs.pop(ind2)
    return all_epochs
