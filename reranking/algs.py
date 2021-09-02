import math
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = getLogger(__name__)


class Reranking:
    """
    Re-ranking algorithms in the paper: Fairness-Aware Ranking in Search
    & Recommendation Systems with Application to LinkedIn Talent Search

    Attributes:
        attributes:
            the attributes corresponding to the each item ranked by
            recommendation system or search engine, from top to bottom.
        distribution:
            the disired distribution for each attribute
        item_ids:
            the item ids, optional input
    Usage:
        Call `re_rank` method.
    """

    def __init__(
        self,
        attributes: List[Union[str, int]],
        distribution: Dict[Union[str, int], float],
        item_ids: Optional[List[int]] = None,
    ) -> None:

        self.attr, self.distr, self.item_ids = self._process_init_inputs(
            attributes, distribution, item_ids
        )
        self.df_formatted = self._format_init()
        self.data, self.p = self._format_alg_inputs()

    def re_rank(
        self, k_max: int = 10, algorithm: str = "det_greedy", verbose: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """
        Processes all the four re-ranking algorithms.

        Reliability of the algorithms:
            1. `det_greedy`, `det_cons` and `det_relaxed` are guaranteed to be feasible if the category of
                the attributes is <=3.
            2. `det_greedy` is NOT guaranteed to be feasible if the category of the attributes is >=4.
            3. `det_const_sort` is guaranteed to be feasible.
        """

        if algorithm in ["det_greedy", "det_cons", "det_relaxed"]:
            re_ranking = self.re_rank_greedy(k_max, algorithm, verbose)
        elif algorithm == "det_const_sort":
            re_ranking = self.re_rank_ics(k_max, verbose)
        else:
            raise NotImplementedError("Invalid algorithm name.")
        return re_ranking

    def re_rank_greedy(
        self, k_max: int = 10, algorithm: str = "det_greedy", verbose: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """Implements the greedy-based algorithms: `DetGreedy`, `DetCons`, `DetRelaxed`."""

        re_ranked_ranking: List[int] = []
        counts = {i: 0 for i in range(len(self.p))}
        for k in range(1, k_max + 1):
            below_min = {
                attr
                for attr, cnt in counts.items()
                if cnt < math.floor(k * self.p[attr])
            }  # minimum requirement violation
            below_max = {
                attr
                for attr, cnt in counts.items()
                if cnt >= math.floor(k * self.p[attr])
                and cnt < math.ceil(k * self.p[attr])
            }  # maximum requirement violation
            if below_min or below_max:
                try:
                    next_attr = self._process_violated_attributes(
                        below_min, below_max, counts, k, algorithm
                    )
                except KeyError as ke:  # not enough item of desired attribute
                    attr_short = ke.args[0][0]
                    logger.debug(
                        f"Lack of item of attribute {attr_short}, input the current top rank item."
                    )
                    next_attr = self._get_top_rank_attr(re_ranked_ranking)
            else:  # below_min and below_max are empty
                next_attr = self._get_top_rank_attr(re_ranked_ranking)
            re_ranked_ranking.append(self.data[(next_attr, counts[next_attr])])
            counts[next_attr] += 1
        if verbose:
            return self._get_verbose(re_ranked_ranking)
        else:
            return re_ranked_ranking

    def re_rank_ics(
        self, k_max: int = 10, verbose: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """Implements `DetConstSort` algorithm."""

        counts = {i: 0 for i in range(len(self.p))}
        min_counts = {i: 0 for i in range(len(self.p))}

        re_ranked_ranking_dict = {}
        max_indices_dict = {}

        last_empty = 0
        k = 0

        while last_empty < k_max:
            k += 1
            temp_min_counts = {
                idx: math.floor(k * self.p[idx]) for idx in range(len(self.p))
            }
            changed_mins = {
                attr for attr, s in min_counts.items() if s < temp_min_counts[attr]
            }
            if changed_mins:
                vals = {}
                for attr in changed_mins:
                    try:
                        vals[attr] = self.data[(attr, counts[attr])]
                    except KeyError:  # not enough item of desired attribute
                        pass
                if vals:
                    ord_changed_mins = np.asarray(
                        (
                            sorted(
                                vals.items(),
                                key=lambda kv: (kv[1], kv[0]),
                                reverse=True,
                            )
                        )
                    )[:, 0].tolist()
                    for attr in ord_changed_mins:
                        re_ranked_ranking_dict[last_empty] = self.data[
                            (attr, counts[attr])
                        ]
                        max_indices_dict[last_empty] = k
                        start = last_empty
                        while (
                            start > 0
                            and max_indices_dict[start - 1] >= start
                            and re_ranked_ranking_dict[start - 1]
                            > re_ranked_ranking_dict[start]
                        ):
                            self._swap_dict_val(max_indices_dict, start - 1, start)
                            self._swap_dict_val(
                                re_ranked_ranking_dict, start - 1, start
                            )
                            start -= 1
                        counts[attr] += 1
                        last_empty += 1
                    min_counts = temp_min_counts.copy()

        re_ranked_ranking = list(re_ranked_ranking_dict.values())
        if verbose:
            return self._get_verbose(re_ranked_ranking)
        else:
            return re_ranked_ranking

    @staticmethod
    def _process_init_inputs(
        attributes: List[Union[str, int]],
        distribution: Dict[Union[str, int], float],
        item_ids: Optional[List[int]],
    ) -> Tuple[List[Union[str, int]], Dict[Union[str, int], float], List[int]]:

        attribute_unique = list(set(attributes))
        attribute_non_exist = [i for i in distribution if i not in attribute_unique]
        attribute_non_desired = [i for i in attribute_unique if i not in distribution]
        distri_sum = sum(distribution.values())

        if attribute_non_exist:
            raise ValueError(
                f"Wrong attribute name in desired distribution: {attribute_non_exist}."
            )
        if attribute_non_desired and abs(distri_sum - 1) < 1e-6:
            # Set non-specified attribute distribution value as 0
            for attr in attribute_non_desired:
                distribution[attr] = 0.0
        if distri_sum < 1 - 1e-6:
            # Set non-specified attribute name as "masked", values sum as 1
            attributes = [i if i in distribution else "masked" for i in attributes]
            distribution["masked"] = 1 - distri_sum
        if distri_sum > 1 + 1e-6:
            raise ValueError("Sum of desired attribute distribution larger than 1.")

        if item_ids is None:
            item_ids = [i for i in range(len(attributes))]

        return attributes, distribution, item_ids

    def _format_init(self) -> pd.DataFrame:

        rankings = [i for i in range(len(self.item_ids))]
        factorized_attribute = pd.factorize(self.attr)[0]
        df = pd.DataFrame(
            {
                "item_id": self.item_ids,
                "attribute": self.attr,
                "attribute_enc": factorized_attribute,
                "model_rank": rankings,
            }
        )
        df.sort_values(
            ["attribute_enc", "model_rank"], ascending=[True, True], inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        df["attri_rank"] = (
            df.groupby("attribute_enc")
            .apply(lambda x: [i for i in range(len(x))])
            .sum()
        )
        df_distr = pd.DataFrame(
            {"attribute": self.distr.keys(), "distr": self.distr.values()}
        )
        df = df.merge(df_distr, on="attribute", how="left")

        return df

    def _format_alg_inputs(self) -> Tuple[Dict[Tuple[int, int], int], List[float]]:
        """Formats algorithm inputs.

        Returns:
            data: in {(attribute_index, ranking in the attribute): overall ranking, ...} format.
            p: List[float] of distributions of each attribute.
        """

        data = {
            (attr, attr_rank): rank
            for attr, attr_rank, rank in zip(
                self.df_formatted.attribute_enc,
                self.df_formatted.attri_rank,
                self.df_formatted.model_rank,
            )
        }
        p = self.df_formatted.drop_duplicates("attribute_enc").distr.tolist()
        return data, p

    def _get_verbose(self, re_ranked_ranking: List[int]) -> pd.DataFrame:

        df_verbose = pd.DataFrame(
            {
                "model_rank": re_ranked_ranking,
                "re_rank": [i for i in range(len(re_ranked_ranking))],
            }
        )
        df_verbose = df_verbose.merge(self.df_formatted, on="model_rank", how="left")
        df_verbose.drop(["attribute_enc", "attri_rank"], axis=1, inplace=True)

        return df_verbose

    def _process_violated_attributes(
        self,
        below_min: Set[int],
        below_max: Set[int],
        counts: Dict[int, int],
        k: int,
        algorithm: str,
    ) -> int:
        """
        Finds the attribute of the next item based on the violations criterias.
        """

        s: Dict[int, Union[int, float]] = {}
        if below_min:
            for i in below_min:
                s[i] = self.data[(i, counts[i])]
            # Get the desired attribute with toppest rank
            next_attr = min(s, key=lambda x: s[x])
        else:
            if algorithm == "det_greedy":
                for i in below_max:
                    s[i] = self.data[(i, counts[i])]
                next_attr = min(s, key=lambda x: s[x])
            elif algorithm == "det_cons":
                for i in below_max:
                    s[i] = math.ceil(k * self.p[i]) / self.p[i]
                self.data[(i, counts[i])]  # catch KeyError exception
                next_attr = max(s, key=lambda x: s[x])
            elif algorithm == "det_relaxed":
                ns = {}
                for i in below_max:
                    ns[i] = math.ceil(math.ceil(k * self.p[i]) / self.p[i])
                temp = min(ns.values())
                next_attr_set = [key for key in ns if ns[key] == temp]
                for i in next_attr_set:
                    s[i] = self.data[(i, counts[i])]
                next_attr = min(s, key=lambda x: s[x])
            else:
                raise NotImplementedError("Invalid name for the greedy algorithms.")
        return next_attr

    def _get_top_rank_attr(self, re_ranked_ranking: List[int]) -> int:
        """
        Gets next attribute which is of the toppest rank item in the remained items.

        Use when not enough item for the required attribute or both `below_min` and
        `below_max` are empty.
        """

        data_rest: Dict[Tuple[int, int], int] = {
            idx: rank
            for idx, rank in self.data.items()
            if rank not in re_ranked_ranking
        }
        next_attr = min(data_rest, key=lambda k: data_rest[k])[0]
        return next_attr

    @staticmethod
    def _swap_dict_val(dic: Dict[int, Any], key_1: int, key_2: int) -> Dict[int, Any]:
        """Swaps values of two keys in a dictionary."""

        dic[key_1], dic[key_2] = dic[key_2], dic[key_1]
        return dic
