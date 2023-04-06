import numpy as np
import pandas as pd
from typing import Union, Tuple


def data_splitter(
    dataframe: pd.DataFrame,
    *,
    train_size: float,
    test_size: float,
    random_seed: Union[int, np.random.BitGenerator, np.random.Generator],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function is intended to split the data frame with TESS flares to train / test / val samples
    so that there are no objects with the same TIC in different samples. It is done in order the future
    model will not over-trained to detect the templates.
    """

    rng = np.random.default_rng(random_seed)

    unique_tics = dataframe["TIC"].unique()
    n_tics = len(unique_tics)

    train_tics = rng.choice(unique_tics, size=(int(train_size * n_tics)), replace=False)
    remained_tics = list(set(train_tics).symmetric_difference(unique_tics))

    test_tics = rng.choice(remained_tics, size=(int(test_size * n_tics)), replace=False)
    val_tics = list(set(remained_tics).symmetric_difference(test_tics))

    train_data = dataframe[dataframe["TIC"].isin(train_tics)].reset_index(drop=True)
    test_data = dataframe[dataframe["TIC"].isin(test_tics)].reset_index(drop=True)
    val_data = dataframe[dataframe["TIC"].isin(val_tics)].reset_index(drop=True)

    assert len(dataframe) == len(train_data) + len(test_data) + len(val_data)

    return train_data, test_data, val_data
