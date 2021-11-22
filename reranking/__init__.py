from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .reranker import Reranker
from .metrics import *


def rerank(
    item_attribute: List[Any],
    desired_distribution: Dict[Any, float],
    max_na: Optional[int] = None,
    k_max: Optional[int] = None,
    algorithm: str = "det_greedy",
    verbose: bool = False,
) -> Union[List[int], pd.DataFrame]:
    rerank = Reranker(item_attribute, desired_distribution, max_na)
    return rerank(k_max, algorithm, verbose)
