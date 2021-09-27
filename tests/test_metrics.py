from typing import Any, Dict, List

import pytest

from reranking.metrics import (
    cal_infeasible,
    cal_kld,
    cal_ndcg_diff,
    cal_ndkl,
    cal_skew,
    cal_skews,
    kld,
    skew,
    skews,
)


class TestMetrics:
    @pytest.fixture
    def item_attributes(self) -> List[Any]:
        return [1, 1, 2, 3, 4]

    @pytest.fixture
    def dict_p(self) -> Dict[Any, float]:
        return {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.3}

    def test_skew(self) -> None:
        assert isinstance(skew(0.5, 0.5), float)

    def test_cal_skew(self, item_attributes: List[Any]) -> None:
        assert isinstance(cal_skew(item_attributes, 1, 0.5, 3), float)

    def test_skews(self) -> None:
        actual = skews([0.1, 0.3, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1])
        assert all(isinstance(i, float) for i in actual)

    def test_cal_skews(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        actual = cal_skews(item_attributes, dict_p, k=3)
        assert all(isinstance(i, float) for i in actual)

    def test_kld(self) -> None:
        assert isinstance(kld([0.1, 0.3, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1]), float)

    def test_cal_kld(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        assert isinstance(cal_kld(item_attributes, dict_p), float)

    def test_cal_ndcg_diff(self) -> None:
        assert isinstance(cal_ndcg_diff([0, 1, 4, 3], 4), float)
        assert cal_ndcg_diff([0, 1, 2, 3], 4) == 1.0
        assert cal_ndcg_diff([4, 5, 6, 7], 4) == 0.0

    def test_cal_ndkl(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        assert isinstance(cal_ndkl(item_attributes, dict_p), float)

    def test_cal_infeasible(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        infeasible_index, infeasible_count = cal_infeasible(item_attributes, dict_p, 5)
        assert isinstance(infeasible_index, int)
        assert isinstance(infeasible_count, int)
