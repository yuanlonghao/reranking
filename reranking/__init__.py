import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .reranker import Reranker

__all__ = [
    "rerank",
    "rerank_multiprocessing",
]


def rerank(
    item_attribute: List[Any],
    desired_distribution: Dict[Any, float],
    k_max: Optional[int] = None,
    max_na: Optional[int] = None,
    algorithm: str = "det_greedy",
    verbose: bool = False,
) -> Union[List[int], pd.DataFrame]:
    """
    If n_workers is None, use single cpu.
    If n_workers is 0, use all cpus.
    """
    reranker = Reranker(algorithm, verbose)
    return reranker(item_attribute, desired_distribution, k_max, max_na)


def rerank_multiprocessing(
    item_attribute: List[List[Any]],
    desired_distribution: List[Dict[Any, float]],
    k_max: Optional[int] = None,
    max_na: Optional[int] = None,
    algorithm: str = "det_greedy",
    verbose: bool = False,
    n_workers: int = 0,
) -> Union[List[List[int]], List[pd.DataFrame]]:

    if n_workers == 0:
        n_workers = mp.cpu_count()

    reranker = Reranker(algorithm, verbose)
    inputs = [
        (ia, dd, k_max, max_na) for ia, dd in zip(item_attribute, desired_distribution)
    ]

    with mp.Pool(n_workers) as p:
        return p.map(reranker.multiprocessing_warpper, inputs)
