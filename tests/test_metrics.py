from typing import Any, Dict, List

import pytest

from reranking.metrics import *


class TestMetrics:
    @pytest.fixture
    def item_attributes(self) -> List[Any]:
        return [1, 1, 2, 3, 4]

    @pytest.fixture
    def dict_p(self) -> Dict[Any, float]:
        return {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.3}

    def test_proportion(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        actual = proportion(item_attributes, list(dict_p.keys()))
        assert all(isinstance(i, float) for i in actual)

    def test_skew(self) -> None:
        assert isinstance(skew(0.5, 0.5), float)

    def test_skew_static(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        item_distr = proportion(item_attributes)
        actual = skew_static(item_distr, list(dict_p.keys()))
        assert all(isinstance(i, float) for i in actual)

    def test_kld(self) -> None:
        assert isinstance(kld([0.1, 0.3, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1]), float)

    def test_ndcg_diff(self) -> None:
        assert isinstance(ndcg_diff([0, 1, 4, 3], 4), float)
        assert ndcg_diff([0, 1, 2, 3], 4) == 1.0
        assert ndcg_diff([4, 5, 6, 7], 4) == 0.0

    def test_ndkl(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        assert isinstance(ndkl(item_attributes, dict_p), float)

    def test_infeasible(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        infeasible_index, infeasible_count = infeasible(item_attributes, dict_p, 5)
        assert isinstance(infeasible_index, int)
        assert isinstance(infeasible_count, int)
