import numpy as np
import pandas as pd

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def data_splitter(dataframe: pd.DataFrame, train_size: float, test_size: float):
    unique_tics = dataframe["TIC"].unique()
    n_tics = len(unique_tics)

    train_tics = rng.choice(unique_tics, size=(int(train_size * n_tics)), replace=False)
    remained_tics = list(set(train_tics).symmetric_difference(unique_tics))

    test_tics = rng.choice(remained_tics, size=(int(test_size * n_tics)), replace=False)
    val_tics = list(set(remained_tics).symmetric_difference(test_tics))

    train_data = dataframe[dataframe["TIC"].isin(train_tics)].reset_index(drop=True)
    test_data = dataframe[dataframe["TIC"].isin(test_tics)].reset_index(drop=True)
    val_data = dataframe[dataframe["TIC"].isin(val_tics)].reset_index(drop=True)

    return train_data, test_data, val_data
