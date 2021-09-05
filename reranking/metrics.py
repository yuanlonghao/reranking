import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def skew(
    item_attributes: List[Any],
    object_attribute: Any,
    desired_proportion: float,
    k: int,
) -> float:
    """
    item_attributes: name or index of the attribute of each recommended item
    object_attribute: skew of which attribute
    desired_proportion: desired propportion of the attribute
    k: top k ranked results
    """
    count = item_attributes[:k].count(object_attribute)
    propotion = count / k
    s = math.log((propotion + 1e-10) / (desired_proportion + 1e-10))
    return s


def min_max_skew(
    item_attributes: List[Any],
    dict_p: Dict[Any, float],
    k: Optional[int] = None,
    min_max: str = "min",
) -> float:
    """
    item_attributes: the attribute of each recommended item
    dict_p:  Dict[name/index of the attribute, desired_proportion]
    k: top k ranked results
    """

    if k is None:
        k = len(item_attributes)
    skew_list: List[float] = []
    for object_attribute, desired_proportion in dict_p.items():
        skew_list.append(skew(item_attributes, object_attribute, desired_proportion, k))
    if min_max == "min":
        return min(skew_list)
    elif min_max == "max":
        return max(skew_list)
    else:
        raise ValueError("Not MinSkew or MaxSkew.")


def kl_divergence(distri_1: List[float], distri_2: List[float]) -> float:
    """
    distri_1, distri_2: two list of distribution values
    """
    vals = []
    for i, j in zip(distri_1, distri_2):
        if i * j != 0:  # skip any 0 values
            vals.append(i * math.log(i / j))
    return sum(vals)


def ndkl(item_attributes: List[Any], dict_p: Dict[Any, float]) -> float:
    """
    Normalized discounted cumulative KL-divergence (NDKL)

    item_attributes: the attribute of each recommended item
    dict_p:  Dict[name/index of the attribute, desired_proportion]
    """
    n_items = len(item_attributes)

    Z = np.sum(1 / (np.log2(np.arange(1, n_items + 1) + 1)))
    total = 0.0

    for i in range(1, n_items + 1):
        value_counts = (
            pd.Series(item_attributes[:i]).value_counts(normalize=True).to_dict()
        )
        distr_1 = []
        distr_2 = []
        for attr in dict_p:
            try:
                distr_1.append(value_counts[attr])
            except KeyError:
                distr_1.append(0)
            distr_2.append(dict_p[attr])
        total += (1 / math.log2(i + 1)) * kl_divergence(distr_1, distr_2)
    res: float = (1 / Z) * total
    return res


def infeasible(
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
