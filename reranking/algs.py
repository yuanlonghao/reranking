import math
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = getLogger(__name__)

## TODO: deal with this situation: attributes = ["f1", ["f1", "f2"], "f2", ...]


class Reranking:
    """
    Re-ranking algorithms in the paper: Fairness-Aware Ranking in Search
    & Recommendation Systems with Application to LinkedIn Talent Search

    Attributes:
        item_attr:
            The attributes in order of each item ranked by recommendation system
            or search engine, from top to bottom.
        distr:
            The disired distribution for each attribute.
        df_formatted:
            Dataframe containing all the processed information.
        data:
            In format of {(`attribute index`, `rank in the attribute`): overall rank}.
        p:
            Desired distribution in list format.

    """

    def __init__(
        self,
        item_attribute: List[Any],
        desired_distribution: Dict[Any, float],
        max_n_attribute: Optional[int] = None,  # TODO: implement this
    ) -> None:

        self.max_n_attribute = max_n_attribute

        self.item_attr, self.distr = self._process_init(
            item_attribute, desired_distribution
        )
        self.df_formatted, self.data, self.p = self._format_init()

    def _process_init(
        self,
        item_attribute: List[Any],
        desired_distribution: Dict[Any, float],
    ) -> Tuple[List[Any], Dict[Any, float]]:
        """
        Processes input item attributes and desired distribution to the proper form.

        Remark:
        set_1: attributes in item and distr
        set_2: attributes in item but not in distr
        set_3: attributes in distr but not in item

        Process logic:
            1. Sum of distr values is larger than 1: ValueError
            2. Mask the lowest value attrs in distr to remain `self.max_n_attribute` desired attrs
            3. set_1 False: NameError
            4. set_1 True:
                - set_2 True, set_3 True: mask set_2 in item and set_3 in distr
                - set_2 False, set_3 True: NameError (distr contains item, lack attribute in item)
                - set_2 False, set_3 False: pass (item set equals distr set)
                - set_2 True, set_3 False: pass (item contains distr, no need to mask)
        """

        item_attr = item_attribute.copy()
        distr = desired_distribution.copy()

        distri_sum = sum(distr.values())
        if distri_sum > 1 + 1e-6:
            raise ValueError("Sum of desired attribute distribution larger than 1.")

        if self.max_n_attribute is not None:
            n_merge = len(distr) - self.max_n_attribute
            if n_merge > 0:
                sorted_attrs = sorted(distr, key=lambda x: distr[x])
                distr = self._mask_distr(distr, sorted_attrs[: n_merge + 1])
            else:
                pass

        attr_in_item_in_distr = set(item_attr) & set(distr)
        attr_in_item_not_distr = set(item_attr) - attr_in_item_in_distr
        attr_in_distr_not_item = set(distr) - attr_in_item_in_distr
        if attr_in_item_in_distr:
            if attr_in_distr_not_item and not attr_in_item_not_distr:
                raise NameError(
                    f"Wrong attribute in distribution: {attr_in_distr_not_item}."
                )
            elif attr_in_distr_not_item and attr_in_item_not_distr:
                item_attr = [
                    "masked" if i in attr_in_item_not_distr else i for i in item_attr
                ]
                distr = self._mask_distr(distr, list(attr_in_distr_not_item))
            else:
                pass
        else:
            raise NameError("Item attribute and distribution have no intersection.")
        return item_attr, distr

    def _mask_distr(
        self, distr: Dict[Any, float], mask_attr: List[Any]
    ) -> Dict[Any, float]:
        valid_distr = {k: v for k, v in distr.items() if k not in mask_attr}
        valid_distr["masked"] = 1.0 - sum(valid_distr.values())
        return valid_distr

    def _format_init(
        self,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int], List[float]]:
        """Formats init inputs.

        Returns:
            df: dataframe contains processed information.
            data: in {(attribute_index, ranking in the attribute): overall ranking, ...} format.
            p: List[float] of distributions of each attribute.
        """

        factorized_attribute = pd.factorize(self.item_attr)[0]
        df = pd.DataFrame(
            {
                "model_rank": list(range(len(self.item_attr))),
                "attribute": self.item_attr,
                "attribute_enc": factorized_attribute,
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

        data = {
            (a, a_rank): rank
            for a, a_rank, rank in zip(
                df.attribute_enc,
                df.attri_rank,
                df.model_rank,
            )
        }
        p = df.drop_duplicates("attribute_enc").distr.tolist()

        return df, data, p

    def __call__(
        self, k_max: int = 10, algorithm: str = "det_greedy", verbose: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """
        Processes all the four re-ranking algorithms.

        Args:
            k_max: re-ranking top k_max items
            algorithm: name of one of the four algorithms
            verbose: if True, output dataframe with more infomation

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
            changed_mins = {a for a, s in min_counts.items() if s < temp_min_counts[a]}
            if changed_mins:
                vals = {}
                for a in changed_mins:
                    try:
                        vals[a] = self.data[(a, counts[a])]
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
                    for a in ord_changed_mins:
                        re_ranked_ranking_dict[last_empty] = self.data[(a, counts[a])]
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
                        counts[a] += 1
                        last_empty += 1
                    min_counts = temp_min_counts.copy()

        re_ranked_ranking = list(re_ranked_ranking_dict.values())
        if verbose:
            return self._get_verbose(re_ranked_ranking)
        else:
            return re_ranked_ranking

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
