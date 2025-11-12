from typing import *
from dataclasses import dataclass
import random
import itertools

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
import base64

from tqdm import tqdm

import nac as nac
import nac.util

###############################################################################
# https://stackoverflow.com/a/75898999
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def copy_doc(wrapper: Callable[P, T]):
    """An implementation of functools.wraps."""

    def decorator(func: Callable) -> Callable[P, T]:
        func.__doc__ = wrapper.__doc__
        return func

    return decorator


@copy_doc(plt.figure)
def figure(num: Any = 1, *args, **kwargs) -> Figure:
    """Creates a figure that is independent on the global plt state"""
    fig = Figure(*args, **kwargs)

    def show():
        manager = backend_agg.new_figure_manager_given_figure(num, fig)
        display(
            manager.canvas.figure,
            metadata=backend_inline._fetch_figure_metadata(manager.canvas.figure),
        )
        manager.destroy()

    fig.show = show
    return fig


###############################################################################
class LazyList[T](List[T]):
    """
    Delays list creation until first request
    """

    def __init__(self, generator: Callable[[], Iterable[T]]):
        self._generator = generator
        self._list: List[T] | None = None

    def _get_list(self) -> List[T]:
        if self._list is None:
            self._list = list(self._generator())
        return self._list

    def __iter__(self) -> Iterator[T]:
        return self._get_list().__iter__()

    def __len__(self) -> int:
        return self._get_list().__len__()

    def __getitem__(self, i: SupportsIndex) -> T:
        return self._get_list().__getitem__(i)


###############################################################################
from typing import List

class Columns:

    TIMESTAMP = "timestamp"
    GRAPH = "graph"
    DATASET = "dataset"
    VERTEX_NUM = "vertex_num"
    EDGE_NUM = "edge_num"
    TRIANGLE_COMPONENTS_NUM = "triangle_components_num"
    EXTENDED_CLASSES_NUM = "extended_classes_num"
    RELABEL = "relabel"
    SPLIT = "split"
    MERGING = "merging"
    SUBGRAPH_SIZE = "subgraph_size"
    USE_SMART_SPLIT = "use_smart_split"
    USED_EXTENDED_CLASSES = "used_extended_classes"
    ANY_FINISHED = "nac_any_finished"
    FIRST_COLORING_NUM = "nac_first_coloring_num"
    FIRST_MEAN_TIME = "nac_first_mean_time"
    FIRST_ROUNDS = "nac_first_rounds"
    FIRST_CHECK_IS_NAC = "nac_first_check_is_NAC"
    FIRST_CHECKS = "nac_first_check_cycle_mask"
    FIRST_MERGE = "nac_first_merge"
    FIRST_MERGE_NO_COMMON_VERTEX = "nac_first_merge_no_common_vertex"
    ALL_COLORING_NUM = "nac_all_coloring_num"
    ALL_MEAN_TIME = "nac_all_mean_time"
    ALL_ROUNDS = "nac_all_rounds"
    ALL_CHECK_IS_NAC = "nac_all_check_is_NAC"
    ALL_CHECKS = "nac_all_check_cycle_mask"
    ALL_MERGE = "nac_all_merge"
    ALL_MERGE_NO_COMMON_VERTEX = "nac_all_merge_no_common_vertex"

    first: list[str] = [FIRST_MEAN_TIME, FIRST_CHECKS]
    all: list[str] = [ALL_MEAN_TIME, ALL_CHECKS]

    ALL: list[str] = [
        TIMESTAMP,
        GRAPH,
        DATASET,
        VERTEX_NUM,
        EDGE_NUM,
        TRIANGLE_COMPONENTS_NUM,
        EXTENDED_CLASSES_NUM,
        RELABEL,
        SPLIT,
        MERGING,
        SUBGRAPH_SIZE,
        USE_SMART_SPLIT,
        USED_EXTENDED_CLASSES,
        ANY_FINISHED,
        FIRST_COLORING_NUM,
        FIRST_MEAN_TIME,
        FIRST_ROUNDS,
        FIRST_CHECK_IS_NAC,
        FIRST_CHECKS,
        FIRST_MERGE,
        FIRST_MERGE_NO_COMMON_VERTEX,
        ALL_COLORING_NUM,
        ALL_MEAN_TIME,
        ALL_ROUNDS,
        ALL_CHECK_IS_NAC,
        ALL_CHECKS,
        ALL_MERGE,
        ALL_MERGE_NO_COMMON_VERTEX,
    ]
    identifying = [
        GRAPH,
        DATASET,
        SPLIT,
        RELABEL,
        MERGING,
        SUBGRAPH_SIZE,
        USE_SMART_SPLIT,
        USED_EXTENDED_CLASSES,
    ]

COLUMNS: list[str] = Columns.ALL


@dataclass
class MeasurementResult:
    timestamp: datetime.datetime
    graph: str
    dataset: str
    vertex_num: int
    edge_num: int
    triangle_components_num: int
    triangle_extended_classes_num: int
    relabel: str
    split: str
    merging: str
    subgraph_size: int
    use_smart_split: bool
    used_triangle_extended_classes: bool
    nac_any_finished: bool
    nac_first_coloring_num: Optional[int]
    nac_first_mean_time: Optional[int]
    nac_first_rounds: Optional[int]
    nac_first_check_is_NAC: Optional[int]
    nac_first_check_cycle_mask: Optional[int]
    nac_first_merge: Optional[int]
    nac_first_merge_no_common_vertex: Optional[int]
    nac_all_coloring_num: Optional[int]
    nac_all_mean_time: Optional[int]
    nac_all_rounds: Optional[int]
    nac_all_check_is_NAC: Optional[int]
    nac_all_check_cycle_mask: Optional[int]
    nac_all_merge: Optional[int]
    nac_all_merge_no_common_vertex: Optional[int]

    def to_list(self) -> List:
        return [
            self.timestamp,
            self.graph,
            self.dataset,
            self.vertex_num,
            self.edge_num,
            self.triangle_components_num,
            self.triangle_extended_classes_num,
            self.relabel,
            self.split,
            self.merging,
            self.subgraph_size,
            self.use_smart_split,
            self.used_triangle_extended_classes,
            self.nac_any_finished,
            self.nac_first_coloring_num,
            self.nac_first_mean_time,
            self.nac_first_rounds,
            self.nac_first_check_is_NAC,
            self.nac_first_check_cycle_mask,
            self.nac_first_merge,
            self.nac_first_merge_no_common_vertex,
            self.nac_all_coloring_num,
            self.nac_all_mean_time,
            self.nac_all_rounds,
            self.nac_all_check_is_NAC,
            self.nac_all_check_cycle_mask,
            self.nac_all_merge,
            self.nac_all_merge_no_common_vertex,
        ]


###############################################################################
def to_benchmark_results(data: List[MeasurementResult] = []) -> pd.DataFrame:
    """
    Converts the list of measurements into a dataframe
    """
    return pd.DataFrame(
        [x.to_list() for x in data],
        columns=COLUMNS,
    )


def graph_to_id(graph: nx.Graph) -> str:
    """
    Encodes a graph as a base64 string
    """
    return base64.standard_b64encode(
        nx.graph6.to_graph6_bytes(graph, header=False).strip()
    ).decode()


def graph_from_id(id: str) -> nx.Graph:
    """
    Encodes a graph from a base64 string
    """
    return nac.NiceGraph(nx.graph6.from_graph6_bytes(base64.standard_b64decode(id)))


###############################################################################

OUTPUT_DIR: str = None
OUTPUT_BENCH_FILE_START = "bench_res"
OUTPUT_VERBOSE: bool = False


def get_output_dir() -> str:
    return OUTPUT_DIR


def find_latest_record_file(
    prefix: str,
    dir: str,
) -> str | None:
    """
    Finds the latest record file name according to the sorted order
    """

    def filter_cond(name: str) -> bool:
        return name.startswith(prefix) and name.endswith(".csv")

    data = sorted(filter(filter_cond, os.listdir(dir)), reverse=True)

    if len(data) == 0:
        return None
    file_name = data[0]
    return file_name


def load_records(
    file_name: str | None = None,
    dir: str | None = None,
    allow_output: bool | None = None,
) -> pd.DataFrame:
    """
    Loads the results from the last run or the run specified by `file_name` in the `dir` given.
    """
    dir = dir or OUTPUT_DIR
    if file_name == None:
        file_name = find_latest_record_file(OUTPUT_BENCH_FILE_START, dir)
        if file_name is None:
            if allow_output or OUTPUT_VERBOSE:
                print(f"No file with results found in {dir}!")
            return to_benchmark_results()
        print(f"Found file: {file_name}")

    path = os.path.join(dir, file_name)
    df = pd.read_csv(path)
    df = df[COLUMNS]
    assert len(df) > 0
    if df.isna().sum().sum() > 0:
        print(f"File {file_name} has missing values!")
    return df


def store_results(
    df: pd.DataFrame,
    file_name: str | None = None,
    dir: str | None = None,
) -> str:
    """
    Stores results in the given file
    """
    dir = dir or OUTPUT_DIR
    if file_name is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{OUTPUT_BENCH_FILE_START}_{current_time}.csv"
    path = os.path.join(dir, file_name)

    # Filter out outliers (over 60s) that run when I put my laptor into sleep mode
    df = df.query(f"{Columns.ALL_MEAN_TIME} < 60_000 and {Columns.FIRST_MEAN_TIME} < 60_000")

    df.to_csv(path, header=True, index=False)
    return file_name


def update_stored_data(
    dfs: List[pd.DataFrame] = [],
    head_loaded: bool = True,
) -> pd.DataFrame:
    """
    Adds the given dataframes to the stored data clearing any duplicates
    """
    df = load_records()
    if head_loaded:
        display(df)
    if len(dfs) != 0:
        new_df = pd.concat(dfs)

        if new_df.isna().sum().sum() > 0:
            print("Empty values found!")
            display(new_df[new_df.isna().any(axis=1)])
            new_df = new_df[~(new_df.isna().any(axis=1))]

        df = pd.concat((df, new_df))
    df = df.drop_duplicates( subset=Columns.identifying, keep="last")
    store_results(df)
    return df


###############################################################################
class BenchmarkTimeoutException(Exception):
    """
    Exception raised when the benchmark timed out
    """

    def __init__(self, msg: str = "The benchmark timed out", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


def with_timeout[**T, R, D](
    function: Callable[T, R], time_limit: int | None, default: D
) -> Callable[T, R | D]:
    """
    Stops the function execution after a specified timeout in seconds is reached.
    """
    if time_limit is None:
        return function

    def impl(*args: P.args, **kwargs: P.kwargs):
        try:
            # signals are not exact, but generally work
            def timeout_handler(signum, frame):
                raise BenchmarkTimeoutException()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(time_limit)

            return function(*args, **kwargs)
        except BenchmarkTimeoutException:
            return default
        finally:
            signal.alarm(0)

    return impl


###############################################################################
@dataclass
class MeasuredRecord:
    """
    Pepresents measurement result of a single type
    """

    time_sum: int = 0
    coloring_no: int = 0
    rounds: int = 0
    checks_is_NAC: int = 0
    checks_cycle_mask: int = 0
    merge: int = 0
    merge_no_common_vetex: int = 0

    @property
    def mean_time(self) -> int:
        if self.rounds == 0:
            return 0
        return int(self.time_sum / self.rounds * 1000)


@dataclass
class MeasuredData:
    """
    Groups measurement results for both the types of tests
    """

    first: Optional[MeasuredRecord]
    all: Optional[MeasuredRecord]


###############################################################################


def nac_benchmark_core(
    graph: nx.Graph,
    rounds: int,
    first_only: bool,
    strategy: Tuple[str, str],
    use_triangle_extended_classes: bool,
    time_limit: int,
    seed: int | None = 42,
) -> MeasuredData:
    """
    Runs benchmarks for NAC coloring search
    Returns results grouped by relabel, split, merge and subgraph size strategies
    """

    if use_triangle_extended_classes:
        nac_valid_class_type = nac.NACValidClassType.EXTENDED
    else:
        nac_valid_class_type = nac.NACValidClassType.TRIANGLES

    result = MeasuredData(None, None)
    rand = random.Random(seed)

    def find_colorings():
        start_time = time.time()

        itr = iter(
            nac.NAC_colorings(
                graph=graph,
                algorithm=strategy[1],
                relabel_strategy=strategy[0],
                nac_valid_class_type=nac_valid_class_type,
                seed=rand.randint(0, 2**30),
            )
        )

        first_col = next(itr, None)
        first_time = time.time()

        if result.first is None:
            result.first = MeasuredRecord()
        result.first = MeasuredRecord(
            time_sum=result.first.time_sum + first_time - start_time,
            coloring_no=0 if first_col is None else 1,
            rounds=result.first.rounds + 1,
            checks_is_NAC=nac.NAC_check_called()[0],
            checks_cycle_mask=nac.NAC_check_called()[1],
            merge=nac.NAC_check_called()[2],
            merge_no_common_vetex=nac.NAC_check_called()[3],
        )

        if first_only:
            return

        j = 0
        for j, coloring in enumerate(itr):
            pass
        end_time = time.time()

        if result.all is None:
            result.all = MeasuredRecord()
        result.all = MeasuredRecord(
            time_sum=result.all.time_sum + end_time - start_time,
            coloring_no=j + 1 + 1,
            rounds=result.all.rounds + 1,
            checks_is_NAC=nac.NAC_check_called()[0],
            checks_cycle_mask=nac.NAC_check_called()[1],
            merge=nac.NAC_check_called()[2],
            merge_no_common_vetex=nac.NAC_check_called()[3],
        )

    def run() -> None:
        [find_colorings() for _ in range(rounds)]

    with_timeout(
        run,
        time_limit=time_limit * rounds,
        default=None,
    )()

    return result


###############################################################################
def create_measurement_result(
    graph: nx.Graph,
    dataset_name: str,
    triangle_components: int,
    extended_classes: int,
    nac_first: Optional[MeasuredRecord],
    nac_all: Optional[MeasuredRecord],
    relabel_strategy: str,
    split_strategy: str,
    merge_strategy: str,
    subgraph_size: int,
    use_smart_split: bool,
    used_triangle_extended_classes: bool,
    timestamp: datetime.datetime = datetime.datetime.now(datetime.UTC),
) -> MeasurementResult:
    """
    Constructor for a MeasurementResult
    """
    vertex_no = nx.number_of_nodes(graph)
    edge_no = nx.number_of_edges(graph)

    nac_any_finished = (nac_first or nac_all) is not None
    nac_first = nac_first or MeasuredRecord()
    nac_all = nac_all or MeasuredRecord()

    return MeasurementResult(
        timestamp=timestamp,
        graph=graph_to_id(graph),
        dataset=dataset_name,
        vertex_num=vertex_no,
        edge_num=edge_no,
        triangle_components_num=triangle_components,
        triangle_extended_classes_num=extended_classes,
        relabel=relabel_strategy,
        split=split_strategy,
        merging=merge_strategy,
        subgraph_size=subgraph_size,
        use_smart_split=use_smart_split,
        used_triangle_extended_classes=used_triangle_extended_classes,
        nac_any_finished=nac_any_finished,
        nac_first_coloring_num=nac_first.coloring_no,
        nac_first_mean_time=nac_first.mean_time,
        nac_first_rounds=nac_first.rounds,
        nac_first_check_is_NAC=nac_first.checks_is_NAC,
        nac_first_check_cycle_mask=nac_first.checks_cycle_mask,
        nac_first_merge=nac_first.merge,
        nac_first_merge_no_common_vertex=nac_first.merge_no_common_vetex,
        nac_all_coloring_num=nac_all.coloring_no,
        nac_all_mean_time=nac_all.mean_time,
        nac_all_rounds=nac_all.rounds,
        nac_all_check_is_NAC=nac_all.checks_is_NAC,
        nac_all_check_cycle_mask=nac_all.checks_cycle_mask,
        nac_all_merge=nac_all.merge,
        nac_all_merge_no_common_vertex=nac_all.merge_no_common_vetex,
    )


###############################################################################
# ANALYTICS
###############################################################################
def drop_outliers(
    df: pd.DataFrame,
    var: str = "nac_first_mean_time",
    bottom_perc: float = 0.01,
    top_perc: float = 0.01,
) -> pd.DataFrame:
    """
    Drop top and bottom percent of outliers
    """
    bottom = df[var].quantile(bottom_perc)
    top = df[var].quantile(1 - top_perc)
    return df[(df[var] >= bottom) & (df[var] <= top)]


def filter_graphs_that_finished_for_all_strategies(df: pd.DataFrame) -> pd.Series:
    """
    Keep only graphs where all the tested strategies finished
    """
    all_strategies_groups = df[[Columns.ANY_FINISHED]].groupby(Columns.GRAPH)
    all_strategies_finished = all_strategies_groups.all()
    return all_strategies_finished[all_strategies_finished == True].index.unique()


def finished_graphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returs runs where there are only graphs wholse all runs finished
    """
    return df.loc[filter_graphs_that_finished_for_all_strategies(df)]


def finished_graphs_no_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returs runs where there are only graphs wholse all runs finished excluding the naive-cycles algorithm
    """
    return df.loc[
        filter_graphs_that_finished_for_all_strategies(
            df.query(f"{Columns.SPLIT} != 'naive-cycles'")
        )
    ]


# Tries to preserve graphs that partially failed
def replace_failed_results(
    df: pd.DataFrame, replace_with: int | None = None
) -> pd.DataFrame:
    """
    Replaces failed runs with dummy data and marks such data for later processing
    """
    # Chooses the expected time that was actually used
    if replace_with is None:
        max_runtime = df.query(f"{Columns.ANY_FINISHED} == True")[
            [Columns.FIRST_MEAN_TIME, Columns.ALL_MEAN_TIME]
        ].max(axis=None)
        if max_runtime > 25_000:
            replace_with = 5_000
        else:
            replace_with = ((max_runtime + 4_999) // 5_000) * 5_000

    df = df.copy()
    df_failed = df[Columns.ANY_FINISHED] == False
    df[NAC_DUMMY_MEAN_TIME_USED] = False
    df.loc[df_failed, NAC_DUMMY_MEAN_TIME_USED] = True
    df.loc[df_failed, Columns.FIRST_MEAN_TIME] = replace_with
    df.loc[df_failed, Columns.ALL_MEAN_TIME] = replace_with
    df.loc[df_failed, Columns.FIRST_CHECKS] = (
        0  # these results will be automatically filtered out
    )
    df.loc[df_failed, Columns.ALL_CHECKS] = 0
    df.loc[df_failed, Columns.ANY_FINISHED] = True
    return df


################################################################################
# LaTeX export tooling
#
# Resources:
# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
################################################################################

DEFAULT_FIG_WIDTH = 398.33858
LATEX_ENABLED = False
FIGURES_DIR = "figures"
GOLDEN_RATIO = (5**0.5 - 1) / 2

def fig_size(
    width: float = DEFAULT_FIG_WIDTH,
    fraction: float = 2,
    height_ratio: float = GOLDEN_RATIO,
    subplots: Tuple[int, int] = (1, 1),
):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_ratio * (subplots[0] / subplots[1])

    if subplots == (1, 1):
        fig_width_in *= 3 / 4
        fig_height_in *= 2 / 5

    return (fig_width_in, fig_height_in)


def export_figure(
    fig: Figure,
    dataset: str,
    dir: str = FIGURES_DIR,
) -> None:
    if not LATEX_ENABLED:
        return
    mode = (
        "first"
        if "first" in fig.value_column
        else "all"
        if "all" in fig.value_column
        else fig.value_column
    )
    groupped_by = {
        Columns.VERTEX_NUM: "vertex",
        Columns.EXTENDED_CLASSES_NUM: "monochromatic",
        Columns.TRIANGLE_COMPONENTS_NUM: "triangle",
    }[fig.x_column]
    metric = "runtime" if "time" in fig.value_column else "checks"
    kind = fig.based_on
    aggregations = fig.aggregations.replace(" ", "-")
    export_figure_impl(fig, dataset, mode, groupped_by, metric, kind, aggregations, dir)


def export_figure_impl(
    fig: Figure,
    dataset: str,
    mode: Literal["first", "all"],
    groupped_by: Literal["vertex", "monochromatic", "triangle"],
    metric: Literal["runtime", "checks", "reduction"],
    kind: Literal["relabel", "split", "merging", "use_smart_split", "subgraph_size"],
    aggregations: Literal["mean", "median", "3rd quartile"],
    dir: str = FIGURES_DIR,
) -> None:
    if not LATEX_ENABLED:
        return
    os.makedirs(dir, exist_ok=True)
    file_name = f"graph_export_{dataset}_{mode}_{groupped_by}_{metric}_{kind}_{aggregations}.pgf"
    # print(f"Exporting: {file_name}")
    fig.savefig(os.path.join(dir, file_name), format="pgf", bbox_inches="tight")

def export_extended_classes_vs_triangle_components(
        fig: Figure,
        dataset: str,
        dir: str = FIGURES_DIR,
    ):
    if not LATEX_ENABLED:
        return
    os.makedirs(dir, exist_ok=True)
    file_name = f"monochromatic_vs_triangle_{dataset}.pgf"
    fig.savefig(os.path.join(dir, file_name), format="pgf", bbox_inches="tight")


def export_standard_figure_list(
    dataset: str,
    figs: Sequence[Figure],
    dir: str = FIGURES_DIR,
) -> None:
    if not LATEX_ENABLED:
        return
    for fig in tqdm(figs):
        export_figure(fig, dataset, dir)


def enable_latex_output():
    from matplotlib.backends import backend_pgf  # for LaTeX output

    global LATEX_ENABLED
    LATEX_ENABLED = True

    # import seaborn as sns
    # sns.set_style("whitegrid")
    # sns.set_theme("paper")
    # matplotlib.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.family": "serif",
            # Use LaTeX default serif font.
            "font.serif": [],
            "text.usetex": True,
            # "font.size": 6,
            # "savefig.dpi": 600,
            "legend.fontsize": 10, # 12 is to large for some graph
            "figure.titlesize": 18,
            # "axes.labelsize": 6,
            # "xtick.labelsize": 6,
            # "ytick.labelsize": 6
        }
    )

def trcon(short: bool = False) -> str:
    if LATEX_ENABLED:
        base = r"$\triangle$"
    else:
        base = "△"

    if not short:
        return base + "-connected"
    else:
        return base + "-conn."

def trext(short: bool = False) -> str:
    if LATEX_ENABLED:
        base = r"$\triangle$"
    else:
        base = "△"

    if not short:
        return base + "-extended"
    else:
        return base + "-ext."

###############################################################################
def plot_extended_vs_original_triangle_components(df: pd.DataFrame, dataset: str | None = None, n: int = 10):
    """
    Compares effectivness of △-extended classes and △-connected components
    """
    if dataset is not None:
        df = df.query(f"{Columns.DATASET} == '{dataset}'")
    df = df
    df = df.reset_index()[[Columns.GRAPH, Columns.EXTENDED_CLASSES_NUM, Columns.TRIANGLE_COMPONENTS_NUM]].drop_duplicates().set_index(Columns.GRAPH)
    graph_no = df.index.nunique()
    graph_no_bellow_monoch_n = df.query(f"{Columns.EXTENDED_CLASSES_NUM} < {n}").index.nunique()
    graph_no_bellow_triang_n = df.query(f"{Columns.TRIANGLE_COMPONENTS_NUM} < {n}").index.nunique()
    print(f"Graphs with less then {n} △-extended classes: {graph_no_bellow_monoch_n}/{graph_no} ({graph_no_bellow_monoch_n/graph_no*100}%)")
    print(f"Graphs with less then {n} △-connected components: {graph_no_bellow_triang_n}/{graph_no} ({graph_no_bellow_triang_n/graph_no*100}%)")

    dist = 1+df[Columns.TRIANGLE_COMPONENTS_NUM].max()-df[Columns.TRIANGLE_COMPONENTS_NUM].min()
    dist = (dist+3)//4
    alpha = 0.6
    fig = figure(
        figsize=fig_size(DEFAULT_FIG_WIDTH//2, height_ratio=1),
    )
    ax = fig.subplots(1,1)
    ax.hist(df[Columns.EXTENDED_CLASSES_NUM], bins=dist, alpha=alpha, stacked=True, label=f"{trext()} classes")
    ax.hist(df[Columns.TRIANGLE_COMPONENTS_NUM], bins=dist, alpha=alpha, stacked=True, label=f"{trcon()} components")
    ax.set_xlabel(f"Number of {trext(short=True)} classes or {trcon(short=True)} components")
    ax.set_ylabel("Number of graphs")
    ax.legend()
    if not LATEX_ENABLED and dataset:
        ax.set_title(dataset)
    return fig

###############################################################################
NAC_DUMMY_MEAN_TIME_USED = "nac_dummy_mean_time_used"


def _legend_order_key(label: str) -> int:
    label = label.lower()
    if "naive cycles" in label:
        return 0
    if "cycles\\_match" in label:
        return 1
    if "neighbors\\_degree" in label:
        return 3
    if "neighbors" in label:
        return 2
    if "none" in label:
        return 4
    if "cuts" in label:
        return 5
    if "kernighan" in label:
        return 6
    return 99


def _group_and_plot(
    df: pd.DataFrame,
    log_scale: bool,
    axs: List[plt.Axes],
    aggregations: List[Literal["mean", "median", "3rd quartile"]],
    x_column: Literal["vertex_no", "monochromatic_classes_no"],
    based_on: Literal["relabel", "split", "merging"],
    value_columns: List[Literal["nac_first_mean_time", "nac_all_mean_time"]],
):
    # In case we are using only dummy values for a x-axes tick, we do not plot it
    is_using_dummy = NAC_DUMMY_MEAN_TIME_USED in df.columns

    if is_using_dummy:
        df = df.loc[:, [x_column, based_on, *value_columns, NAC_DUMMY_MEAN_TIME_USED]]
    else:
        df = df.loc[:, [x_column, based_on, *value_columns]]
    groupped = df.groupby([x_column, based_on])

    for ax, aggregation in zip(axs, aggregations):
        match aggregation:
            case "mean":
                action = lambda x: x.mean()
            case "median":
                action = lambda x: x.median()
            case "3rd quartile":
                action = lambda x: x.quantile(0.75)
        aggregated = groupped.agg(
            {col: action for col in value_columns}
            | ({NAC_DUMMY_MEAN_TIME_USED: "all"} if is_using_dummy else {})
        )

        aggregated = aggregated.reorder_levels([based_on, x_column], axis=0)

        for name in aggregated.index.get_level_values(based_on).unique():
            data = aggregated.loc[name]
            if is_using_dummy:
                data = data.query(f"{NAC_DUMMY_MEAN_TIME_USED} == False")
            if len(data) == 0:
                continue
            for value_column in value_columns:
                title = (
                    ",".join([name, value_column]) if len(value_columns) > 1 else name
                )
                ax.plot(data.index, data[value_column], label=title)

        rename_based_on = {
            Columns.VERTEX_NUM: "Vertices",
            Columns.TRIANGLE_COMPONENTS_NUM: f"{trcon()} components",
            Columns.EXTENDED_CLASSES_NUM: f"{trext()} classes",
        }

        # ax.set_title(f"{rename_based_on[x_column]} {based_on} ({aggregation})")
        # ax.set_title(f"{rename_based_on[x_column]} ({aggregation})")
        if len(aggregations) > 1:
            ax.set_title(f"{aggregation.capitalize()}")
        if log_scale:
            ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(rename_based_on[x_column])

        if "time" in value_columns[0]:
            ax.set_ylabel(r"Time [ms]")
        elif "check" in value_columns[0]:
            ax.set_ylabel(r"Checks [call]")
        else:
            ax.set_ylabel("FIX ME!")

        # Sort legend labels according to our rules
        handles, labels = zip(
            *sorted(zip(*ax.get_legend_handles_labels()), key=lambda x: x[1])
        )
        order = sorted([(_legend_order_key(lab), i) for i, lab in enumerate(labels)])
        order = [i for _, i in order]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

        if len(aggregations) == 1 or LATEX_ENABLED:
            ax.legend(
                handles,
                labels,
                bbox_to_anchor=(1.0, 1.0),
                loc="upper left",
            )
        else:
            ax.legend(
                handles,
                labels,
            )

def plot_frame(
    title: str,
    df: pd.DataFrame,
    ops_value_columns_sets: List[str | List[str]]=[
        Columns.FIRST_MEAN_TIME,
        Columns.FIRST_CHECKS,
        Columns.ALL_MEAN_TIME,
        Columns.ALL_CHECKS,
    ],
    ops_x_column=[
        Columns.EXTENDED_CLASSES_NUM,
        Columns.VERTEX_NUM,
    ],
    ops_based_on=[
        # Columns.RELABEL,
        # Columns.SPLIT,
        # Columns.MERGING,
        "split_merging",
        # Columns.USE_SMART_SPLIT,
        # Columns.SUBGRAPH_SIZE,
    ],
    ops_aggregation=[
        "mean",
        "median",
        "3rd quartile",
    ],
    log_scale: bool = True,
    filter_out_exhaustive_mergin_strategies_for_first: bool = True,
) -> List[Figure]:
    """
    Plot a dataframe

    Parameters
    ---------
    title: str
        The title of the plot
    df: pd.DataFrame
        The dataframe to plot
    ops_value_columns_sets:
        Lists of columns to plot at once
    ops_x_column:
        Name of the X-axis columns
    ops_based_on:
        Group based on these columns - separate lines are produced
    ops_aggregation:
        Aggregation functions to apply on columns from ops_value_columns_sets
    """
    print(f"Plotting {df.shape[0]} records and {df.index.nunique()} graphs...")
    figs = []

    ops_value_columns_sets = [[item,] if isinstance(item, str) else list(item) for item in ops_value_columns_sets]

    title_rename = {
        Columns.FIRST_MEAN_TIME: "First NAC-coloring, Runtime",
        Columns.FIRST_CHECKS: "First NAC-coloring, Checks number",
        Columns.ALL_MEAN_TIME: "All NAC-colorings, Runtime",
        Columns.ALL_CHECKS: "All NAC-colorings, Checks number",
    }

    for value_columns in ops_value_columns_sets:
        local_df = df[(df[value_columns] != 0).all(axis=1)]

        if filter_out_exhaustive_mergin_strategies_for_first and "first" in value_columns[0]:
            local_df = local_df.query(f"{Columns.MERGING} != 'sorted_size' and {Columns.MERGING} != 'score'")

        if local_df.shape[0] == 0:
            continue

        for x_column in ops_x_column:
            for based_on in ops_based_on:
                if not LATEX_ENABLED:
                    nrows = 1
                    ncols = len(ops_aggregation)

                    fig = figure(nrows * ncols, (20, 4 * nrows), layout="constrained")
                    title_detail = " | ".join(
                        title_rename[value_column] for value_column in value_columns
                    )
                    fig.suptitle(f"{title} ({title_detail})")  # , fontsize=20

                    figs.append(fig)
                    row = 0
                    axs = [
                        fig.add_subplot(nrows, ncols, i + ncols * row + 1)
                        for i in range(len(ops_aggregation))
                    ]

                    _group_and_plot(
                        local_df,
                        log_scale,
                        axs,
                        ops_aggregation,
                        x_column,
                        based_on,
                        value_columns,
                    )
                    row += 1
                else:
                    for aggregation in ops_aggregation:
                        nrows = 1
                        ncols = 1

                        fig = figure(
                            nrows * ncols,
                            figsize=fig_size(
                                # width=DEFAULT_FIG_WIDTH if len(ops_aggregation) > 1 else DEFAULT_FIG_WIDTH * 3/4,
                                subplots=(nrows, ncols),
                            ),
                            layout="constrained",
                        )

                        # Used later to safe the figue to the correct file
                        fig.value_column = value_columns[0]
                        fig.x_column = x_column
                        fig.based_on = based_on
                        fig.aggregations = aggregation  # "+".join(ops_aggregation)

                        figs.append(fig)
                        row = 0
                        axs = [fig.add_subplot(nrows, ncols, ncols * row + 1)]

                        _group_and_plot(
                            local_df,
                            log_scale,
                            axs,
                            [aggregation],
                            x_column,
                            based_on,
                            value_columns,
                        )
                        row += 1
    return figs


###############################################################################
def _plot_is_NAC_coloring_calls_groups(
    title: str,
    df: pd.DataFrame,
    ax: plt.Axes,
    x_column: Literal["vertex_no", "monochromatic_classes_no"],
    value_columns: List[Literal["nac_first_mean_time", "nac_all_mean_time"]],
    aggregation: Literal["mean", "median", "3rd quartile"],
    legend_rename_dict: Dict[str, str] = {},
):
    df = df.loc[:, [x_column, *value_columns]]
    groupped = df.groupby([x_column])
    match aggregation:
        case "mean":
            aggregated = groupped.mean()
        case "median":
            aggregated = groupped.median()
        case "3rd quartile":
            aggregated = groupped.quantile(0.75)

    rename_based_on = {
        Columns.VERTEX_NUM: "Vertices",
        Columns.TRIANGLE_COMPONENTS_NUM: f"{trcon()} components",
        Columns.EXTENDED_CLASSES_NUM: f"{trext()} classes",
    }

    # display(aggregated)
    aggregated.plot(ax=ax)
    if not LATEX_ENABLED:
        ax.set_title(f"{title} - {aggregation.capitalize()}")
    ax.set_ylabel("Checks [call]")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(rename_based_on[x_column])
    handles, labels = ax.get_legend_handles_labels()
    if LATEX_ENABLED:
        ax.legend(
            handles,
            [legend_rename_dict[l] for l in labels],
            bbox_to_anchor=(1.0, 1.0),
            loc="upper left",
        )
    else:
        ax.legend(
            handles,
            [legend_rename_dict[l] for l in labels],
            # loc = 'upper left',
        )


def plot_is_NAC_coloring_calls(
    df: pd.DataFrame,
) -> List[Figure]:
    figs = []

    df = df.query(f"{Columns.ALL_COLORING_NUM} != 0").copy()
    print(f"Plotting {df.shape[0]} records...")

    related_columns = [
        Columns.VERTEX_NUM,
        Columns.EDGE_NUM,
        Columns.TRIANGLE_COMPONENTS_NUM,
        Columns.EXTENDED_CLASSES_NUM,
        Columns.ALL_COLORING_NUM,
        Columns.ALL_CHECK_IS_NAC,
        Columns.ALL_CHECKS,
    ]
    df = df.loc[:, related_columns]
    # this does not help our algorithm to stand out, but the graphs can be drawn more easily

    df["exp_edge_num"] = 2 ** (df[Columns.EDGE_NUM] - 1)
    df["exp_triangle_component_num"] = 2 ** (df[Columns.TRIANGLE_COMPONENTS_NUM] - 1)
    df["exp_monochromatic_class_num"] = 2 ** (df[Columns.EXTENDED_CLASSES_NUM] - 1)

    df["scaled_edge_num"] = df[Columns.EDGE_NUM] / df[Columns.ALL_COLORING_NUM]
    df["scaled_triangle_component_num"] = (
        df[Columns.TRIANGLE_COMPONENTS_NUM] / df[Columns.ALL_COLORING_NUM]
    )
    df["scaled_monochromatic_class_num"] = (
        df[Columns.EXTENDED_CLASSES_NUM] / df[Columns.ALL_COLORING_NUM]
    )
    df["scaled_nac_all_check_cycle_mask"] = (
        df[Columns.ALL_CHECKS] / df[Columns.ALL_COLORING_NUM]
    )

    df["inv_edge_num"] = df[Columns.ALL_COLORING_NUM] / df[Columns.EDGE_NUM]
    df["inv_triangle_component_num"] = (
        df[Columns.ALL_COLORING_NUM] / df[Columns.TRIANGLE_COMPONENTS_NUM]
    )
    df["inv_monochromatic_class_num"] = (
        df[Columns.ALL_COLORING_NUM] / df[Columns.EXTENDED_CLASSES_NUM]
    )
    df["inv_nac_all_check_cycle_mask"] = (
        df[Columns.ALL_COLORING_NUM] / df[Columns.ALL_CHECKS]
    )
    df["inv_nac_all_check_is_NAC"] = (
        df[Columns.ALL_COLORING_NUM] / df[Columns.ALL_CHECK_IS_NAC]
    )

    df["new_edge_num"] = df[Columns.EDGE_NUM] / df["exp_monochromatic_class_num"]
    df["new_triangle_component_num"] = (
        df[Columns.TRIANGLE_COMPONENTS_NUM] / df["exp_monochromatic_class_num"]
    )
    df["new_monochromatic_class_num"] = (
        df[Columns.EXTENDED_CLASSES_NUM] / df["exp_monochromatic_class_num"]
    )
    df["new_nac_all_check_cycle_mask"] = (
        df[Columns.ALL_CHECKS] / df["exp_monochromatic_class_num"]
    )

    rename_dict = {
        "exp_edge_num": "Naive - Edges",
        "exp_triangle_component_num": f"Naive - {trcon()} components",
        "exp_monochromatic_class_num": f"Naive - {trext()} classes",
        "nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "nac_all_check_is_NAC": "Subgraphs - IsNACColoring",
        "scaled_edge_num": "Naive - Edges",
        "scaled_triangle_component_num": f"Naive - {trcon()} components",
        "scaled_monochromatic_class_num": f"Naive - {trext()} classes",
        "scaled_nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "inv_edge_num": "Naive - Edges",
        "inv_triangle_component_num": f"Naive - {trcon()} components",
        "inv_monochromatic_class_num": f"Naive - {trext()} classes",
        "inv_nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "inv_nac_all_check_is_NAC": "Subgraphs - IsNACColoring",
        "new_edge_num": "Naive - Edges",
        "new_triangle_component_num": f"Naive - {trcon()} components",
        "new_monochromatic_class_num": f"Naive - {trext()} classes",
        "new_nac_all_check_cycle_mask": "Subgraphs - CycleMask",
        "new_nac_all_check_is_NAC": "Subgraphs - IsNACColoring",
    }

    ops_x_column = [
        Columns.EXTENDED_CLASSES_NUM,
        Columns.VERTEX_NUM,
    ]
    ops_value_groups = [
        [
            "exp_edge_num",
            "exp_triangle_component_num",
            "exp_monochromatic_class_num",
            "nac_all_check_cycle_mask",
            "nac_all_check_is_NAC",
        ],
        # ["scaled_edge_num", "scaled_triangle_component_num", "scaled_monochromatic_class_num", "scaled_nac_all_check_cycle_mask"],
        [
            "inv_edge_num",
            "inv_triangle_component_num",
            "inv_monochromatic_class_num",
            "inv_nac_all_check_cycle_mask",
            "inv_nac_all_check_is_NAC",
        ],
        # ["new_edge_num",    "new_triangle_component_num",    "new_monochromatic_class_num",    "new_nac_all_check_cycle_mask" ],
    ]
    ops_aggregation = [
        "mean",
        "median",
        # "3rd quartile",
    ]

    nrows = len(ops_value_groups)
    ncols = len(ops_aggregation)

    for x_column in ops_x_column:
        if not LATEX_ENABLED:
            row = 0
            for title, value_columns in zip(
                [
                    "The number of checks",
                    # r"\#is_NAC_coloring() calls/\#NAC(G)",
                    "The number of NAC-colorings / The number of checks",
                    # f"Checks / {triang()}-connected components number",
                ],
                ops_value_groups,
            ):
                fig = figure(nrows * ncols, (20, 4 * nrows), layout="constrained")
                # fig.suptitle(
                #     f"Reduction of CycleMask and IsNACColoring checks against the naive algorithm",
                #     fontsize=20,
                # )
                figs.append(fig)

                axs = [
                    fig.add_subplot(nrows, ncols, i + ncols * row + 1)
                    for i in range(len(ops_aggregation))
                ]
                for ax, aggregation in zip(axs, ops_aggregation):
                    _plot_is_NAC_coloring_calls_groups(
                        title,
                        df,
                        ax,
                        x_column,
                        value_columns,
                        aggregation,
                        legend_rename_dict=rename_dict,
                    )
                row += 1
        ################################################################################
        else:
            row = 0
            for title, value_columns in zip(
                [
                    r"The number of checks",
                    # r"\#is_NAC_coloring() calls/\#NAC(G)",
                    r"The number of NAC-colorings / The number of checks",
                    # f"Checks / {triang()}-connected components number",
                ],
                ops_value_groups,
            ):
                for aggregation in ops_aggregation:
                    nrows = 1
                    ncols = 1

                    fig = figure(
                        nrows * ncols,
                        figsize=fig_size(
                            # width=DEFAULT_FIG_WIDTH if len(ops_aggregation) > 1 else DEFAULT_FIG_WIDTH * 3/4,
                            subplots=(nrows, ncols),
                        ),
                        layout="constrained",
                    )

                    # Used later to safe the figue to the correct file
                    fig.value_column = value_columns[0]
                    fig.x_column = x_column
                    fig.based_on = "checks"
                    fig.aggregations = aggregation  # "+".join(ops_aggregation)

                    figs.append(fig)
                    row = 0
                    axs = [fig.add_subplot(nrows, ncols, ncols * row + 1)]

                    for ax, aggregation in zip(axs, [aggregation]):
                        _plot_is_NAC_coloring_calls_groups(
                            title,
                            df,
                            ax,
                            x_column,
                            value_columns,
                            aggregation,
                            legend_rename_dict=rename_dict,
                        )
                    row += 1
    return figs


###############################################################################
