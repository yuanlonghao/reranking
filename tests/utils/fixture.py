import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def df_recommend() -> pd.DataFrame:
    """
    Creates pseudo ranking of 100 items with specified feature distributions.

    `item` here can be persons to be recommended with their gender, location, and age features.
    """
    np.random.seed(seed=43)
    rankings = [i for i in range(100)]
    item_ids = np.random.choice(rankings, 100, replace=False)
    genders = np.random.choice(["male"] * 70 + ["female"] * 30, 100, replace=False)
    locations = np.random.choice(
        ["Tokyo"] * 40 + ["Kanagawa"] * 30 + ["others"] * 30, 100, replace=False
    )
    ages = np.random.choice(
        ["20s"] * 20 + ["30s"] * 40 + ["40s"] * 20 + ["others"] * 20, 100, replace=False
    )
    df_pseudo = pd.DataFrame(
        {
            "item_id": item_ids,
            "model_rank": rankings,
            "gender": genders,
            "age": ages,
            "location": locations,
        }
    )
    return df_pseudo
