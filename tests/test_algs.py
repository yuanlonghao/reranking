from typing import Dict, List, Union

import numpy as np
import pytest

from reranking.algs import Reranking


class TestReranking:
    @pytest.fixture
    def rankings(self) -> List[int]:
        return [i for i in range(100)]

    @pytest.fixture
    def genders(self) -> List[str]:
        np.random.seed(seed=43)
        genders: List[str] = np.random.choice(
            ["male"] * 70 + ["female"] * 30, 100, replace=False
        ).tolist()
        return genders

    @pytest.mark.parametrize(
        "distribution, k_max, algorithm",
        [
            ({"female": 0.5, "male": 0.5}, 10, "det_greedy"),
            ({"female": 0.0, "male": 1.0}, 10, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 60, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 70, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 100, "det_greedy"),
            ({"female": 0.5, "male": 0.5}, 10, "det_cons"),
            ({"female": 0.0, "male": 1.0}, 10, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 60, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 70, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 100, "det_cons"),
            ({"female": 0.5, "male": 0.5}, 10, "det_relaxed"),
            ({"female": 0.0, "male": 1.0}, 10, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 60, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 70, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 100, "det_relaxed"),
            ({"female": 0.5, "male": 0.5}, 10, "det_const_sort"),
            ({"female": 0.0, "male": 1.0}, 10, "det_const_sort"),
            ({"female": 0.5, "male": 0.5}, 60, "det_const_sort"),
            ({"female": 0.5, "male": 0.5}, 70, "det_const_sort"),
            ({"female": 0.5, "male": 0.5}, 100, "det_const_sort"),
        ],
    )
    def test_algorithms(
        self,
        rankings: List[int],
        genders: List[Union[str, int]],
        distribution: Dict[Union[str, int], float],
        k_max: int,
        algorithm: str,
    ) -> None:
        ranker = Reranking(rankings, genders, distribution)
        re_rankings = ranker.re_rank(algorithm=algorithm, k_max=k_max)

        re_features = [genders[i] for i in re_rankings]
        acutal_male = re_features.count("male")
        actual_female = re_features.count("female")
        if k_max == 70:  # not enough item for female
            acutal_male == k_max - genders.count("female")
            actual_female = genders.count("female")
        elif k_max == 100:  # result should be overall distribution
            assert acutal_male == genders.count("male")
            assert actual_female == genders.count("female")
        else:  # enough item for each attribute
            assert acutal_male == distribution["male"] * k_max
            assert actual_female == distribution["female"] * k_max
