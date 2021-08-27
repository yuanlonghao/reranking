import math
from logging import getLogger
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = getLogger(__name__)


class Reranking:
    ## TODO: add a attribute named unknown

    def __init__(
        self,
        item_ids: List[int],
        attributes: List[Union[str, int]],
        distribution: Dict[Union[str, int], float],
    ) -> None:
        attribute_unique = list(set(attributes))
        non_exist_attribute = [i for i in distribution if i not in attribute_unique]
        non_required_attribute = [i for i in attribute_unique if i not in distribution]
        distri_sum = sum(distribution.values()) 
        if non_exist_attribute:
            print(f"Wrong attribute name in distribution: {non_exist_attribute}.")
        if non_required_attribute:
            print(f"Distribution of feature {non_required_attribute} are not specified.")
        if distri_sum > 1:
            raise ValueError("Sum of required attribute distribution larger than 1.")
                
        self.df_formated = self.format_init_inputs(item_ids, attributes, distribution)
        self.data, self.p = self._format_alg_inputs()

    @staticmethod
    def format_init_inputs(
        item_ids: List[int],
        attributes: List[Union[str, int]],
        distribution: Dict[Union[str, int], float],
    ) -> pd.DataFrame:
        rankings = [i for i in range(len(item_ids))]
        factorized_attribute = pd.factorize(attributes)[0]
        df = pd.DataFrame(
            {
                "item_id": item_ids,
                "attribute": attributes,
                "attribute_enc": factorized_attribute,
                "model_rank": rankings,
            }
        )
        df.sort_values(
            ["attribute_enc", "model_rank"], ascending=[True, True], inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        df["attri_rank"] = (
            df.groupby("attribute_enc").apply(lambda x: [i for i in range(len(x))]).sum()
        )
        df_distr = pd.DataFrame(
            {"attribute": distribution.keys(), "distr": distribution.values()}
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
            (ai, ai_rank): rank
            for ai, ai_rank, rank in zip(
                self.df_formated.attribute_enc,
                self.df_formated.attri_rank,
                self.df_formated.model_rank,
            )
        }
        p = self.df_formated.drop_duplicates("attribute_enc").distr.tolist()
        return data, p

    def _get_verbose(self, re_ranked_ranking: List[int], k_max: int) -> pd.DataFrame:
        df_verbose = pd.DataFrame(
            {"model_rank": re_ranked_ranking, "re_rank": [i for i in range(k_max)]}
        )
        df_verbose = df_verbose.merge(self.df_formated, on="model_rank", how="left")
        df_verbose.drop(["attribute_enc", "attri_rank"], axis=1, inplace=True)

        return df_verbose

    def _process_violated_attributes(
        self,
        below_min: Set[int],
        below_max: Set[int],
        counts: Dict[int, int],
        method: str,
    ) -> int:
        s: Dict[int, Union[int, float]] = {}
        if len(below_min) != 0:
            for i in below_min:
                s[i] = self.data[(i, counts[i])]
            # get the required attribute with toppest rank
            next_attri = min(s, key=lambda k: s[k])
        else:
            if method == "det_greedy":
                for i in below_max:
                    s[i] = self.data[(i, counts[i])]
                next_attri = min(s, key=lambda k: s[k])
            elif method == "det_cons":
                for i in below_max:
                    s[i] = math.ceil(i * self.p[i]) / self.p[i]
                next_attri = max(s, key=lambda k: s[k])
            elif method == "det_relax":
                ns = {}
                for i in below_max:
                    ns[i] = math.ceil(math.ceil(i * self.p[i]) / self.p[i])
                temp = min(ns.values())
                next_attriSet = [key for key in ns if ns[key] == temp]
                for i in next_attriSet:
                    s[i] = self.data[(i, counts[i])]
                next_attri = min(s, key=lambda k: s[k])
            else:
                raise NotImplementedError("Not a valid name for the greedy methods.")
        return next_attri

    def re_rank_greedy(
        self, k_max: int = 10, method: str = "det_greedy", verbose: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """Implements three greedy-based algorithms: `det_greedy`, `dec_cons`, `det_relax`."""
        re_ranked_ranking: List[int] = []
        # if self.df_formated.attribute.nunique() == 1:
        #     re_ranked_ranking =  self.df_formated.model_rank.tolist()[:k_max]
        # else:
        counts = {k: 0 for k in range(len(self.p))}
        for k in range(1, k_max + 1):
            below_min = {
                ai for ai, v in counts.items() if v < math.floor(k * self.p[ai])
            }  # minimum requirement violation
            below_max = {
                ai
                for ai, v in counts.items()
                if v >= math.floor(k * self.p[ai]) and v < math.ceil(k * self.p[ai])
            }  # maximum requirement violation
            if below_min or below_max: 
                try:
                    next_attri = self._process_violated_attributes(
                        below_min, below_max, counts, method
                    )
                except KeyError:
                    # KeyError: not enough required attribute items
                    logger.debug("Lack of item of a required attribute.")
                    # print("Lack of item of a required attribute.")
                    rest_data: Dict[Tuple[int, int], int] = {
                        k: v for k, v in self.data.items() if v not in re_ranked_ranking
                    }
                    next_attri = min(rest_data, key=lambda k: rest_data[k])[0]
            else: # below_min and below_max are empty
                next_attri = 0
                
            re_ranked_ranking.append(self.data[(next_attri, counts[next_attri])])
            counts[next_attri] += 1
        if verbose:
            return self._get_verbose(re_ranked_ranking, k_max)
        else:
            return re_ranked_ranking

    @staticmethod
    def _swap_dict_val(dic: Dict[int, Any], key_1: int, key_2: int) -> Dict[int, Any]:
        """Swaps values of two keys in a dictionary."""
        dic[key_1], dic[key_2] = dic[key_2], dic[key_1]
        return dic

    def re_rank_const_sorting(
        self, k_max: int = 10, verbose: bool = False
    ) -> Union[List[int], pd.DataFrame]:
        """method: Interval Constrained Sorting"""

        re_ranked_ranking_dict = {}
        max_indices_dict = {}
        temp_min_counts = {}
        last_empty = 0
        k = 0
        counts = {k: 0 for k in range(len(self.p))}
        min_counts = {k: 0 for k in range(len(self.p))}

        while last_empty < k_max:
            k += 1
            for j in range(len(self.p)):
                temp_min_counts[j] = math.floor(k * self.p[j])

            changed_mins = {
                ai for ai, s in min_counts.items() if s < temp_min_counts[ai]
            }
            if len(changed_mins) != 0:
                vals = {}
                for ai in changed_mins:
                    vals[ai] = self.data[(ai, counts[ai])]
                ord_changed_mins = np.asarray(
                    (sorted(vals.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
                )[:, 0].tolist()
                for ai in ord_changed_mins:
                    re_ranked_ranking_dict[last_empty] = self.data[(ai, counts[ai])]
                    max_indices_dict[last_empty] = k
                    start = last_empty
                    while (
                        start > 0
                        and max_indices_dict[start - 1] >= max_indices_dict[start]
                        and re_ranked_ranking_dict[start - 1]
                        < re_ranked_ranking_dict[start]
                    ):
                        self._swap_dict_val(max_indices_dict, start - 1, start)
                        self._swap_dict_val(re_ranked_ranking_dict, start - 1, start)
                        start -= 1
                    counts[ai] += 1
                    last_empty += 1
                min_counts = temp_min_counts.copy()

        re_ranked_ranking = [v for v in re_ranked_ranking_dict.values()]
        if verbose:
            return self._get_verbose(re_ranked_ranking, k_max)
        else:
            return re_ranked_ranking
