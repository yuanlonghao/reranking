# reranking: fairness/personalization for recommendation and search

[![Python](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9-red?logo=Python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/reranking?color=green)](https://pypi.org/project/reranking/)
[![GitHub](https://img.shields.io/github/license/yuanlonghao/reranking?color=blue)](https://github.com/yuanlonghao/reranking)


***reranking*** provides algorithms to re-rank the ranked items to any specified item attribute distribution.

This package can be used as a post-processing modular of recommendation system or search engine.

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
```python
from reranking.algs import Reranking

# Here are the attributes of the top-8 ranked items from a recommendation system
item_attribute = ["a1", "a1", "a1", "a2", "a1", "a1", "a1", "a2"]
# Here is the desired item distribution
desired_distribution = {"a1": 0.5, "a2": 0.5}
# We want items of "a1" and "a2" have equal proportions in top-4
r = Reranking(item_attribute, desired_distribution)
r(k_max=4)
```
The output is `[0, 3, 1, 7]` which is the indices of the top-4 items after re-ranking by the desired distribution.
(`item_attribute` has the same order with the ranked items so it contains score information.)

More examples can be found [here](examples/usage_example.ipynb).
