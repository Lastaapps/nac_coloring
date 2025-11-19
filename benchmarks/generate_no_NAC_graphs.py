#!/usr/bin/env python
from typing import *
from dataclasses import dataclass
from collections import defaultdict, deque
import random
import importlib
from random import Random
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline as backend_inline
from matplotlib.backends import backend_agg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import networkx as nx
import os
import time
import datetime
import signal
import itertools
import base64

from tqdm import tqdm

import nac as nac
import nac.util
from nac import NACValidClassType

import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description="Process data with minimum and maximum values."
    )
    parser.add_argument(
        "--min", type=int, required=True, help="Minimum value for processing."
    )
    parser.add_argument(
        "--max", type=int, required=True, help="Maximum value for processing."
    )
    return parser


def create_strategy(
    param: Tuple[str, str, str, int], use_smart_split: bool
) -> Tuple[str, str]:
    relabel, split, merge, subgraph = param
    algo_name = "subgraphs-{}-{}-{}{}".format(
        merge, split, subgraph, "-smart" if use_smart_split else ""
    )
    return (relabel, algo_name)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out!")


def with_timeout(func, timeout=60):
    """Runs a function with a timeout."""

    def wrapper(*args, **kwargs):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        try:
            return func(*args, **kwargs)
        except TimeoutError as _:
            return None
        finally:
            signal.alarm(0)  # Disable the alarm

    return wrapper


DIR = "./graphs_store/no_NAC_coloring"
os.makedirs(DIR, exist_ok=True)


def store_graph(graph: nx.Graph, class_type: NACValidClassType) -> None:
    vertex_no = graph.number_of_nodes()
    file_name = f"no_NAC_coloring-{vertex_no}-{class_type.name}.g6"
    bytes = nx.graph6.to_graph6_bytes(graph, header=False).strip()
    with open(os.path.join(DIR, file_name), "ba") as f:
        f.write(bytes)
        f.write(b"\n")


@with_timeout
def has_NAC_coloring(graph: nx.Graph) -> bool:
    coloring = next(
        iter(
            nac.NAC_colorings(
                graph,
                algorithm=create_strategy(
                    ("", "neighbors_degree", "shared_vertices", 6), False
                )[1],
            )
        ),
        None,
    )
    return coloring is not None


def search_large_graph_no_NAC_coloring(
    nl: int,
    nh: int,
) -> None:
    rand = random.Random()

    counter_attempts = tqdm()
    counter_success = tqdm()
    counter_timeouts = tqdm()

    while True:
        counter_attempts.update()
        n = rand.randint(nl, nh)
        m = 2 * n - 2 + rand.randint(0, 8 * n)
        graph = nx.gnm_random_graph(n, m, seed=rand.randint(0, 2**30))
        if not nx.is_connected(graph):
            continue
        classes_no = nac.find_nac_mono_classes(
            graph,
            # class_type=MonochromaticClassType.MONOCHROMATIC,
            class_type=NACValidClassType.TRIANGLES,
        )[1]
        # if not len(classes_no) > 10:
        if not len(classes_no) > 2 * np.sqrt(n):  # triangle-connected
            # if not len(classes_no) > np.sqrt(n):  # monochromatic
            continue

        res = has_NAC_coloring(graph)
        if res is None:
            counter_timeouts.update()
        if res != False:
            continue

        # print(f"{classes_no}: {nx.graph6.to_graph6_bytes(graph, header=False).strip()}")
        counter_success.update()
        store_graph(graph, NACValidClassType.TRIANGLES)


parser = create_parser()
args = parser.parse_args()

search_large_graph_no_NAC_coloring(args.min, args.max)
