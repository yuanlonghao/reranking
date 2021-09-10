from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from reranking.algs import Reranking


class TestReranking:
    @pytest.fixture
    def genders(self) -> List[str]:
        np.random.seed(seed=43)
        genders: List[str] = np.random.choice(
            ["male"] * 70 + ["female"] * 30, 100, replace=False
        ).tolist()
        return genders

    @pytest.mark.parametrize(
        "item_attributes, distribution",
        [
            ([1, 2, 3], {1: 0.3, 2: 0.4, 3: 0.5}),  # sum of distibution larger than 1
            ([1, 2, 3], {4: 0.3, 5: 0.4, 6: 0.3}),  # no intersection
            ([4, 5], {4: 0.3, 5: 0.4, 6: 0.3}),  # distribution attr contains item attr
        ],
    )
    def test_init_raise_exception(
        self, item_attributes: List[Any], distribution: Dict[Any, float]
    ) -> None:
        with pytest.raises((NameError, ValueError)):
            Reranking(item_attributes, distribution)

    @pytest.mark.parametrize(
        "item_attributes, distribution, max_n_attribute, expected_item_attr, expected_distr_key",
        [
            (
                [2, 4, 5],
                {4: 0.3, 5: 0.4, 6: 0.3},
                None,
                ["masked", 4, 5],
                ["masked", 4, 5],
            ),  # mask both
            (
                [4, 5, 6],
                {4: 0.3, 5: 0.4, 6: 0.3},
                None,
                [4, 5, 6],
                [4, 5, 6],
            ),  # mask nothing case 2
            (
                [4, 5, 6, 7],
                {4: 0.3, 5: 0.4, 6: 0.3},
                None,
                [4, 5, 6, 7],
                [
                    4,
                    5,
                    6,
                ],
            ),  # mask Nothing case 2
            (
                [2, 3, 4],
                {2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.1},
                3,
                [2, 3, "masked"],
                [2, 3, "masked"],
            ),  # max_n_attribute constraint case 1
            (
                [1, 2, 3, 4],
                {2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.1},
                3,
                ["masked", 2, 3, "masked"],
                [2, 3, "masked"],
            ),  # max_n_attribute constraint case 2
        ],
    )
    def test_process_init(
        self,
        item_attributes: List[Any],
        distribution: Dict[Any, float],
        max_n_attribute: Optional[int],
        expected_item_attr: List[Any],
        expected_distr_key: List[Any],
    ) -> None:
        r = Reranking(item_attributes, distribution, max_n_attribute=max_n_attribute)
        assert set(expected_item_attr) == set(r.item_attr)
        assert set(expected_distr_key) == set(r.distr)

    @pytest.mark.parametrize(
        "distribution, k_max, algorithm",
        [
            # test performce
            ({"female": 0.5, "male": 0.5}, 10, "det_greedy"),
            ({"female": 0.0, "male": 1.0}, 10, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 10, "det_cons"),
            ({"female": 0.0, "male": 1.0}, 10, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 10, "det_relaxed"),
            ({"female": 0.0, "male": 1.0}, 10, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 10, "det_const_sort"),
            ({"female": 0.0, "male": 1.0}, 10, "det_const_sort"),
            # test edge conditions
            ({"female": 0.5, "male": 0.5}, 70, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 100, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 70, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 100, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 70, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 100, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 70, "det_const_sort"),
            ({"female": 0.5, "male": 0.5}, 100, "det_const_sort"),
        ],
    )
    def test_algorithms(
        self,
        genders: List[Any],
        distribution: Dict[Any, float],
        k_max: int,
        algorithm: str,
    ) -> None:
        ranker = Reranking(genders, distribution)
        re_rankings = ranker(algorithm=algorithm, k_max=k_max)

        re_features = [genders[i] for i in re_rankings]
        acutal_male = re_features.count("male")
        actual_female = re_features.count("female")

        if k_max == 70:  # not enough items for female
            assert acutal_male == k_max - genders.count("female")
            assert actual_female == genders.count("female")
        elif k_max == 100:  # result distribution should be the overall distribution
            assert acutal_male == genders.count("male")
            assert actual_female == genders.count("female")
        else:  # enough item for each attribute
            assert acutal_male == distribution["male"] * k_max
            assert actual_female == distribution["female"] * k_max
