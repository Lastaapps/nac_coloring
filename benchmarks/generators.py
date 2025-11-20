"""
This modules is used to generate and store random graphs of the given class.
There graphs can later be loaded using the dataset module
"""

import math
import os
import random
from pathlib import Path
from typing import *

import networkx as nx
from tqdm import tqdm

import nac
from benchmarks import datasets
from nac import NiceGraph as Graph


class RangeWithCount(NamedTuple):
    """
    Represents range of values with assigned count
    The high bound is excluded
    """

    low: int
    high: int
    cnt: int


def _write_graphs_to_file(
    path: str | Path,
    graphs: Sequence[nx.Graph],
):
    with open(path, "wb") as f:
        for graph in graphs:
            f.write(nx.readwrite.graph6.to_graph6_bytes(graph, header=False))


def _generate_random_graphs_impl(
    ranges_with_count: Sequence[RangeWithCount],
    generate_graph: Callable[[int, int | None], nx.Graph],
    dir: str,
    filename_template: str,
    seed: int | None,
) -> List[Tuple[int, List[nx.Graph]]]:
    """
    Base function for generating the graphs
    """
    os.makedirs(dir, exist_ok=True)
    rand = random.Random(seed)

    configs = [(n, c) for l, h, c in ranges_with_count for n in range(l, h)]
    results: List[Tuple[int, List[nx.Graph]]] = []
    for n, count in tqdm(configs):
        path = Path(os.path.join(dir, f"{filename_template.format(n)}.g6"))

        graphs: List[nx.Graph]
        if path.is_file():
            graphs = list(datasets.load_graph6_graphs_from_file(str(path)))
        else:
            graphs = []

        # skip already generated graphs
        for _ in range(len(graphs)):
            rand.randint(0, 2**30)

        with open(path, "ab") as f:
            for _ in range(count - len(graphs)):
                graph = generate_graph(n, rand.randint(0, 2**30))
                graphs.append(graph)

                f.write(nx.readwrite.graph6.to_graph6_bytes(graph, header=False))
                f.flush()

                import time

                start = time.time()
                while time.time() < start + 1:
                    pass

        results.append((n, graphs))
    return results


# takes ~1h 30m on my laptop
def generate_random_minimally_rigid_graphs(
    dir: str = datasets.DIR_LAMAN_RANDOM,
    filename_template: str = "minimally_rigid_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:
    ranges = (
        RangeWithCount(10, 20, 128),
        RangeWithCount(20, 30, 64),
        RangeWithCount(30, 40, 32),
        RangeWithCount(40, 50, 16),
        RangeWithCount(50, 60, 8),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_minimally_rigid(n, seed),
        dir,
        filename_template,
        seed,
    )


def generate_random_nac_critical_graphs(
    dir: str = os.path.join(datasets.DIR_RANDOM, "nac_critical"),
    filename_template: str = "nac_critical_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:
    ranges = (
        RangeWithCount(10, 20, 500),
        RangeWithCount(20, 30, 500),
        RangeWithCount(30, 40, 500),
        RangeWithCount(40, 50, 500),
        RangeWithCount(50, 60, 500),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_NAC_critical_graph(n, seed),
        dir,
        filename_template,
        seed,
    )


def generate_random_globally_rigid_nac_critical_graphs(
    dir: str = os.path.join(datasets.DIR_RANDOM, "globally_rigid_nac_critical"),
    filename_template: str = "globally_rigid_nac_critical_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:
    ranges = (
        RangeWithCount(10, 20, 100),
        RangeWithCount(20, 30, 100),
        RangeWithCount(30, 40, 100),
        RangeWithCount(40, 50, 100),
        RangeWithCount(50, 60, 100),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_NAC_critical_globally_rigid_graph(n, seed),
        dir,
        filename_template,
        seed,
    )


def generate_threshold_globally_rigid_graphs(
    dir: str = os.path.join(datasets.DIR_RANDOM, "globally_rigid_threshold"),
    filename_template: str = "globally_rigid_threshold_{0}",
    seed: int | None = 42,
) -> List[Tuple[int, List[nx.Graph]]]:
    ranges = (
        RangeWithCount(10, 20, 100),
        RangeWithCount(20, 30, 100),
        RangeWithCount(30, 40, 100),
        RangeWithCount(40, 50, 100),
        RangeWithCount(50, 60, 100),
        RangeWithCount(60, 70, 100),
        RangeWithCount(70, 80, 100),
        RangeWithCount(80, 90, 100),
        RangeWithCount(90, 100, 100),
        RangeWithCount(100, 110, 100),
        RangeWithCount(110, 120, 100),
        RangeWithCount(120, 130, 100),
        RangeWithCount(130, 140, 100),
    )
    return _generate_random_graphs_impl(
        ranges,
        lambda n, seed: _generate_threshold_globally_rigid_graph(n, seed),
        dir,
        filename_template,
        seed,
    )


################################################################################
# Generate a single graph of the given class
################################################################################


def _generate_minimally_rigid(
    n: int,
    seed: int | None,
    min_degree: int | None = None,
) -> nx.Graph:
    import pyrigi.graph

    rand = random.Random(seed)

    while True:
        graph = pyrigi.Graph(nx.gnm_random_graph(n, 2 * n - 3, rand.randint(0, 2**30)))
        if min_degree is not None:
            if next((1 for d in nx.degree(graph) if d < min_degree), None) is not None:
                continue
        if not nx.is_connected(graph):
            continue
        if not graph.is_min_rigid():
            continue
        if len(nac.find_nac_mono_classes(graph)[1]) == 1:
            continue

        return graph


def _generate_NAC_critical_graph(
    n: int,
    seed: int | None,
    log_base: float = math.e,
) -> nx.Graph:
    """
    Generates sparse graphs that should have likely few NAC colorings
    or no NAC colorings what so ever.

    Uses slightly lower bound that the original paper
    Sharp thresholds for NAC-colourings and stable cuts in random graphs, 2025,
    Katie Clinch, John Haslegrave, Tony Huynh, Anthony Nixon
    """
    rand = random.Random(seed)

    p = (2 * math.log(n, log_base) / (n * n)) ** (1 / 3)

    while True:
        graph = Graph(nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30)))
        if not nx.is_connected(graph):
            continue

        return graph


def _generate_NAC_critical_globally_rigid_graph(
    n: int,
    seed: int | None,
    log_base: float = math.e,
) -> nx.Graph:
    """
    Generates graphs that should have likely few NAC colorings
    or no NAC colorings what so ever and are globally rigid.

    Uses bound for NAC-critical graphs provided by paper
    Sharp thresholds for NAC-colourings and stable cuts in random graphs, 2025,
    Katie Clinch, John Haslegrave, Tony Huynh, Anthony Nixon
    """
    import pyrigi

    rand = random.Random(seed)

    p = (2 * math.log(n, log_base) / (n * n)) ** (1 / 3)
    while True:
        graph = pyrigi.Graph(
            nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30))
        )
        if not nx.is_connected(graph):
            continue
        if len(nac.find_nac_mono_classes(graph)[1]) == 1:
            continue
        if not graph.is_globally_rigid():
            continue

        return graph


def _generate_threshold_rigid_graph(
    n: int,
    seed: int | None,
    log_base: float = math.e,
) -> nx.Graph:
    """
    Generates globally rigid graphs.

    Uses bound from Theorem 4.4 from
    The 2-Dimensional Rigidity of Certain Families of Graphs, 2006
    Bill Jackson, Brigitte Servatius, and Herman Servatius
    """
    import pyrigi

    rand = random.Random(seed)

    p = (math.log(n) + 2 * math.log(math.log(n))) / n
    while True:
        graph = pyrigi.Graph(
            nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30))
        )
        if not nx.is_connected(graph):
            continue
        if len(nac.find_nac_mono_classes(graph)[1]) == 1:
            continue
        if not graph.is_rigid():
            continue

        return graph


def _generate_threshold_globally_rigid_graph(
    n: int,
    seed: int | None,
) -> nx.Graph:
    """
    Generates globally rigid graphs.

    Uses bound from Theorem 4.4 from
    The 2-Dimensional Rigidity of Certain Families of Graphs, 2006
    Bill Jackson, Brigitte Servatius, and Herman Servatius
    """
    import pyrigi

    rand = random.Random(seed)

    p = (math.log(n) + 3 * math.log(math.log(n))) / n
    while True:
        graph = pyrigi.Graph(
            nx.fast_gnp_random_graph(n, p, seed=rand.randint(0, 2**30))
        )
        if not nx.is_connected(graph):
            continue
        if len(nac.find_nac_mono_classes(graph)[1]) == 1:
            continue
        if not graph.is_globally_rigid():
            continue

        return graph
