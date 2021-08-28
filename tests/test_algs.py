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
        return np.random.choice(["male"] * 70 + ["female"] * 30, 100, replace=False)

    @pytest.fixture
    def ages(self) -> List[str]:
        np.random.seed(seed=43)
        return np.random.choice(
            ["20s"] * 20 + ["30s"] * 40 + ["40s"] * 20 + ["others"] * 20,
            100,
            replace=False,
        )

    @pytest.fixture
    def locations(self) -> List[str]:
        np.random.seed(seed=43)
        return np.random.choice(
            ["Tokyo"] * 40 + ["Kanagawa"] * 30 + ["others"] * 30, 100, replace=False
        )

    @pytest.mark.parametrize(
        "distribution, k_max, expected_ranking",
        [({"female": 0.5, "male": 0.5}, 10, [0, 5, 1, 9, 2, 12, 3, 14, 4, 18])],
    )
    def test_det_greedy(
        self,
        rankings,
        genders,
        distribution: Dict[Union[str, int], float],
        k_max: int,
        expected_ranking: List[int],
    ) -> None:
        ranker = Reranking(rankings, genders, distribution)
        actual_ranking = ranker.re_rank_greedy(method="det_greedy", k_max=k_max)
        assert actual_ranking == expected_ranking
