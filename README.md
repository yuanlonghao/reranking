# reranking: fairness/personalization for recommendation and search

[![Python](https://img.shields.io/badge/python-3.7%7C3.8%7C3.9%7C3.10-red?logo=Python&logoColor=white)](https://www.python.org)
[![PyTest](https://github.com/yuanlonghao/reranking/actions/workflows/pytest.yml/badge.svg)](https://github.com/yuanlonghao/reranking/actions/workflows/pytest.yml)
[![pre-commit](https://github.com/yuanlonghao/reranking/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/yuanlonghao/reranking/actions/workflows/pre-commit.yml)
[![PyPI](https://img.shields.io/pypi/v/reranking?color=green)](https://pypi.org/project/reranking/)
[![GitHub](https://img.shields.io/github/license/yuanlonghao/reranking?color=blue)](https://github.com/yuanlonghao/reranking)


***reranking*** provides algorithms to re-rank the ranked items to any specified item attribute distribution.

This package can be used as a post-processing modular of recommendation systems or search engines.

Inspired by paper [Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search](https://dl.acm.org/doi/10.1145/3292500.3330691).

## Concept of usage

### Fairness: impose the same distribution to all user
Taking "recommend candidates to recruiters" as the example (this is the case of LinkedIn Recruiter service stressed in the paper), the search engine rankings are re-ranked by the distribution of the protected attributes like gender and demographic parity in consideration of the fair display of the candidates to recruiters.

### Personalization: impose the personalized distributions to each user
For example, when we recommend products to users, the product preference distribution for each user (which can be obtained by the purchase log or view log) can be used to re-rank the item rankings by recommendation systems to get a more personalized recommendation.

## Installation
```shell
$ pip install reranking
```

## Examples
### Re-rank
```python
import reranking
item_attribute = ["a1", "a1", "a1", "a2", "a1", "a1", "a1", "a2", "a2", "a1"]
desired_distribution = {"a1": 0.5, "a2": 0.5}
rerank_indices = reranking.rerank(
    item_attribute,  # attributes of the ranked items
    desired_distribution,  # desired item distribution
    k_max=None,  # length of output, if None, k_max is the length of `item_attribute`
    max_na=None,  # controls the max number of attribute categories applied
    algorithm="det_greedy",  # "det_greedy", "det_cons", "det_relaxed", "det_const_sort"
    verbose=False,  # if True, the output is with detailed information
)
print(rerank_indices)
```
The `rerank_indices` is `[0, 3, 1, 7, 2, 8, 4, 5, 6, 9]` which is the list of item indices after re-ranking by the desired distribution. The top items of the re-ranked list will have the same distribution as the desired distribution if there are enough desired items.
(`item_attribute` has the same order with the item list so it contains rank/score information.)

### Evaluate
```python
# Evaluate the before&after of the re-ranking result above
item_attribute_reranked = [item_attribute[i] for i in rerank_indices]
before = reranking.ndkl(item_attribute, desired_distribution)
after = reranking.ndkl(item_attribute_reranked, desired_distribution)
print(f"{before:.3f}, {after:.3f}")
```
The `before` and `after` are `0.412` and `0.172` respectively which are the normalized discounted cumulative KL-divergence (NDKL) of the ranked item attribute distribution and the desired distribution. (Lower is better.)

More examples can be found [here](examples/usage_example.ipynb).
