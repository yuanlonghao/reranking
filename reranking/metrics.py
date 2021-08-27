import math

import numpy as np


def proportion(df, k, ai):
    count = 0
    for i in range(k):
        if df[i] == ai:
            count = count + 1
    return count / k


def Skew(df, p, k, ai):
    s = math.log((proportion(df, k, ai) + 0.0001) / (p[ai] + 0.0001))
    return s


def MinSkew(df, p, k):
    min_skew = float("inf")
    for x in range(len(p)):
        m = Skew(list(df), p, k, x)
        if m < min_skew:
            min_skew = m
    return min_skew


def MaxSkew(df, p, k):
    max_skew = -float("inf")
    for x in range(len(p)):
        m = Skew(list(df), p, k, x)
        if m > max_skew:
            max_skew = m
    return max_skew


def KLD(D1, D2):
    a = np.asarray(D1, dtype=np.float)
    b = np.asarray(D2, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log((a + 0.00001) / (b + 0.00001)), 0))


def NDKL(df, p):
    Z = np.sum(1 / (np.log2(np.arange(1, len(df) + 1) + 1)))
    total = 0

    for i in range(1, len(df) + 1):
        value = df[:i].value_counts(normalize=True)
        value = value.to_dict()
        D1 = []
        for i in range(len(p)):
            if i in value.keys():
                D1.append(value[i])
            else:
                D1.append(0)

        total = total + (1 / math.log2(i + 1)) * KLD(D1, p)

    return (1 / Z) * total


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0


def ndcg_at_k(df, k, method=0):
    r = list(df)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def infeasibleIndex(df, p):

    """Params: df = dataFrame for Ranked List, a= No of Values of Protected Attribute
    desired_p = desired proportion of ai's (Array){Provisioned for Later Modification}
    k initial value need to set(default=10)

    """

    data = df[:100]
    a = len(p)
    desired_p = p
    tao_r = 100
    infeasibleFlag = False
    InfIndex_tao_r = 0
    InfCount_tao_r = 0

    for k in range(1, tao_r):
        data_temp_k = data[:k]
        infeasibleFlag = False
        for i in range(a):
            desired_p_ai = desired_p[i]
            observed_count_ai = len(data_temp_k[data_temp_k["ai"] == i])
            desired_count_ai = math.floor(desired_p_ai * k)
            if observed_count_ai < desired_count_ai:
                infeasibleFlag = True
                ## Increment Infeasible Count
                InfCount_tao_r += 1

        ## Increment Infeasible Index
        if infeasibleFlag == True:
            InfIndex_tao_r += 1
    if InfIndex_tao_r > 99:
        print(f"You are fucked!! at {a} and it is {InfIndex_tao_r}")
    Infeasible_Return_array = [InfIndex_tao_r, InfCount_tao_r]
    return Infeasible_Return_array
