from typing import Any, Dict, List, Optional, Tuple

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
            ([4, 5], {4: 0.3, 5: 0.4, 6: 0.3}),  # distr contains item attr
        ],
    )
    def test_exception_returns(
        self, item_attributes: List[Any], distribution: Dict[Any, float]
    ) -> None:

        r = Reranking(item_attributes, distribution)
        with pytest.raises((NameError, ValueError)):
            r._format_alg_input()
        default_ranking = list(range(len(item_attributes)))
        reranking = r()
        assert default_ranking == reranking

    @pytest.mark.parametrize(
        "item_attributes, distribution, max_na, expected_item_attr, expected_distr_key, expected_data, expected_p",
        [
            (
                [2, 4, 5],
                {4: 0.3, 5: 0.4, 6: 0.3},
                None,
                ["masked", 4, 5],
                ["masked", 4, 5],
                {(0, 0): 0, (1, 0): 1, (2, 0): 2},
                [0.3, 0.3, 0.4],
            ),  # mask both
            (
                [4, 5, 6],
                {4: 0.3, 5: 0.4, 6: 0.3},
                None,
                [4, 5, 6],
                [4, 5, 6],
                {(0, 0): 0, (1, 0): 1, (2, 0): 2},
                [0.3, 0.4, 0.3],
            ),  # mask nothing
            (
                [4, 5, 6, 7],
                {4: 0.3, 5: 0.4, 6: 0.3},
                None,
                [4, 5, 6, "masked"],
                [
                    4,
                    5,
                    6,
                ],
                {(0, 0): 0, (1, 0): 1, (2, 0): 2, (3, 0): 3},
                [0.3, 0.4, 0.3, 0.0],
            ),  # mask item attr
            (
                [2, 3, 4],
                {2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.1},
                3,
                [2, 3, "masked"],
                [2, 3, "masked"],
                {(0, 0): 0, (1, 0): 1, (2, 0): 2},
                [0.3, 0.2, 0.5],
            ),  # max_na constraint case 1
            (
                [1, 2, 3, 4],
                {2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1, 6: 0.1},
                3,
                ["masked", 2, 3, "masked"],
                [2, 3, "masked"],
                {(0, 0): 0, (0, 1): 3, (1, 0): 1, (2, 0): 2},
                [0.5, 0.3, 0.2],
            ),  # max_na constraint case 2
        ],
    )
    def test__format_alg_input(
        self,
        item_attributes: List[Any],
        distribution: Dict[Any, float],
        max_na: Optional[int],
        expected_item_attr: List[Any],
        expected_distr_key: List[Any],
        expected_data: Dict[Tuple[int, int], int],
        expected_p: List[float],
    ) -> None:
        r = Reranking(item_attributes, distribution, max_na=max_na)

        # test masking
        acutal_item_attr, acutal_distr = r._mask_item_attr_and_distr()
        assert set(expected_item_attr) == set(acutal_item_attr)
        assert set(expected_distr_key) == set(acutal_distr)

        # test algorithm inputs
        _, actual_data, actual_p = r._format_alg_input()
        assert expected_data == actual_data
        np.testing.assert_almost_equal(expected_p, actual_p)

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
        reranking = ranker(algorithm=algorithm, k_max=k_max)

        re_features = [genders[i] for i in reranking]
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
