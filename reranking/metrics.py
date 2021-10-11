import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPSILON = 1e-5


def cal_proportion(
    item_attributes: List[Any], attributes: Optional[Any] = None
) -> List[float]:
    """
    Calculates proportions of each attributes in the recommended items.

    item_attributes: the attribute of each recommended item
    attributes: specified attributes which is optional
    """

    if attributes is None:
        attributes = list(set(item_attributes))
    proportion = [
        item_attributes.count(attr) / len(item_attributes) for attr in attributes
    ]
    return proportion


def cal_skew(p_1: float, p_2: float) -> float:
    """
    Calculates skew.

    p_1, p_2: two probabilities of the same attribute in two distributions
    """

    return math.log((p_1 + EPSILON) / (p_2 + EPSILON))


def cal_skew_static(
    distr_1: List[float], distr_2: List[float]
) -> Tuple[float, float, float]:
    """
    Calculates min, max, absolute mean skew of all the attributes.
    """

    skews = [0.0]
    for i, j in zip(distr_1, distr_2):
        skews.append(math.log((i + EPSILON) / (j + EPSILON)))
    skews_abs = [abs(i) for i in skews]
    return min(skews), max(skews), sum(skews_abs) / len(skews_abs)


def cal_kld(distr_1: List[float], distr_2: List[float]) -> float:
    """
    Calculates KL divergence.
    """

    vals = []
    for i, j in zip(distr_1, distr_2):
        vals.append(i * math.log((i + EPSILON) / (j + EPSILON)))
    return sum(vals)


def cal_reranking_ndcg(
    reranked_ranking: List[int], k_max: Optional[int] = None
) -> float:
    """
    Calculates the NDCG of the ranking change.
    Ranking before reranking: from 0 to k_max, i.e., [0, 1, 2, 3, ...]
    Re-ranked ranking: new ranking after the process, e.g., [0, 4, 2, 3, ...]
    """

    if k_max is None:
        k_max = len(reranked_ranking)
    original_ranking = list(range(k_max))
    reranked_ranking = reranked_ranking[:k_max]
    pred_list = np.array([1 if i in original_ranking else 0 for i in reranked_ranking])
    cg_factor = np.log2(np.arange(2, k_max + 2))
    pred_list_sorted = np.sort(pred_list)[::-1]
    dcg = np.sum(pred_list / cg_factor)
    idcg = np.sum(pred_list_sorted / cg_factor)
    ndcg: float = dcg / idcg if idcg != 0 else 0.0
    return ndcg


def cal_ndkl(item_attributes: List[Any], dict_p: Dict[Any, float]) -> float:
    """
    Calculates normalized discounted cumulative KL-divergence (NDKL).

    item_attributes: the attribute of each recommended item
    dict_p:  Dict[name/index of the attribute, desired_proportion]
    """

    n_items = len(item_attributes)
    Z = np.sum(1 / (np.log2(np.arange(1, n_items + 1) + 1)))

    total = 0.0
    for k in range(1, n_items + 1):
        item_attr_k = item_attributes[:k]
        item_distr = [
            item_attr_k.count(attr) / len(item_attr_k) for attr in dict_p.keys()
        ]
        total += (1 / math.log2(k + 1)) * cal_kld(item_distr, list(dict_p.values()))
    res: float = (1 / Z) * total
    return res


def cal_infeasible(
    item_attributes: List[Any],
    dict_p: Dict[Any, float],
    k_max: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Calculates the infeasible_index and infeasible_count.

    infeasible_index: from 1 to k, items saitisfy violation condition.
    infeasible_count: from 1 to k, count of insufficient attributes by violation condition.
    """

    # (this method is not general used though...)
    if k_max is None:
        k_max = len(item_attributes)
    infeasible_index = 0
    infeasible_count = 0

    for k in range(1, k_max):
        value_counts = pd.Series(item_attributes[:k]).value_counts().to_dict()
        count_attr = []
        for attr in dict_p:
            try:
                count_attr.append(value_counts[attr])
            except KeyError:
                count_attr.append(0)
        for j, val in enumerate(dict_p.values()):
            if count_attr[j] < math.floor(val * k):
                infeasible_count += 1
                if count_attr[j] != 0:
                    infeasible_index += 1
    return infeasible_index, infeasible_count
