from typing import Dict, List, Union

import pytest

from reranking.metrics import (
    dcg_at_k,
    infeasible,
    kl_divergence,
    min_max_skew,
    ndcg_at_k,
    ndkl,
    skew,
)


class TestMetrics:
    @pytest.fixture
    def item_attributes(self) -> List[Union[str, int]]:
        return [1, 1, 2, 3, 4]

    @pytest.fixture
    def dict_p(self) -> Dict[int, float]:
        return {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.3}

    def test_skew(self, item_attributes: List[Union[str, int]]) -> None:
        isinstance(skew(item_attributes, 1, 0.5, 3), float)

    @pytest.mark.parametrize("min_max", ["min", "max"])
    def test_min_max_skew(
        self,
        item_attributes: List[Union[str, int]],
        dict_p: Dict[Union[str, int], float],
        min_max: str,
    ) -> None:
        isinstance(min_max_skew(item_attributes, dict_p, 3, min_max=min_max), float)
