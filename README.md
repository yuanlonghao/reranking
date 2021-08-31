# reranking
[![Python](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9-red?logo=Python&logoColor=white)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/reranking?color=green)](https://pypi.org/project/reranking/)
[![GitHub](https://img.shields.io/github/license/yuanlonghao/reranking?color=blue)](https://github.com/yuanlonghao/reranking)


***reranking*** provides algorithms to re-rank the items to any desired item attribute distribution.

This package can be used in the post-processing step of recommendation system or search engine.

Inspired by paper [Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search](https://dl.acm.org/doi/10.1145/3292500.3330691).

## Concept of usage

### Fairness: impose a same distribution to the ranking of each user
Taking "recommend candidates to recruiters" as the example (this is the case of LinkedIn Recruiter service stressed in the paper), the search engine ranking are re-ranked by the distribution of the protected attributes like gender and demographic parity in consideration of the fair display of the candidates to recruiters.

### Personalization: impose distribution to each user based on their preference
For example, when we recommend products to users, the item preference distribution for each user (obtained by the purchase/view log) can be used to re-rank the item rankings calculated by recommendation systems.

## Installation
```shell
$ pip install reranking
```

## Examples
```python
from reranking.algs import Reranking
r = Reranking(["f1", "f1", "f1", "f2", "f1", "f1", "f1", "f2"], {"f1": 0.5, "f2": 0.5})
r.re_rank(k_max=4) # we want "f2" has equal proportion in top-4
```
The output is `[0, 3, 1, 7]` which is the re-ranked ranking of the top-4 items.

More examples can be found [here](examples/usage_example.ipynb).
