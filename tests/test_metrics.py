from typing import Any, Dict, List

import pytest

from reranking.metrics import infeasible, kld, kld_at_k, min_max_skew, ndkl, skew


class TestMetrics:
    @pytest.fixture
    def item_attributes(self) -> List[Any]:
        return [1, 1, 2, 3, 4]

    @pytest.fixture
    def dict_p(self) -> Dict[Any, float]:
        return {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.3}

    def test_skew(self, item_attributes: List[Any]) -> None:
        isinstance(skew(item_attributes, 1, 0.5, 3), float)

    @pytest.mark.parametrize("min_max", ["min", "max"])
    def test_min_max_skew(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
        min_max: str,
    ) -> None:
        isinstance(min_max_skew(item_attributes, dict_p, 3, min_max=min_max), float)

    def test_kld(self) -> None:
        isinstance(kld([0.1, 0.3, 0.5, 0.0], [0.0, 0.5, 0.4, 0.1]), float)

    def test_kld_at_k(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        isinstance(kld_at_k(item_attributes, dict_p), float)

    def test_ndkl(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        isinstance(ndkl(item_attributes, dict_p), float)

    def test_infeasible(
        self,
        item_attributes: List[Any],
        dict_p: Dict[Any, float],
    ) -> None:
        infeasible_index, infeasible_count = infeasible(item_attributes, dict_p, 5)
        isinstance(infeasible_index, int)
        isinstance(infeasible_count, int)
