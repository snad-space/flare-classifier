import ast
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from feature_extraction import *


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


def array_from_string(array_string):
    if not isinstance(array_string, str):
        return array_string

    array_string = array_string.replace("electron / s", "")
    array_string = ",".join(array_string.replace("[ ", "[").split())

    return np.array(ast.literal_eval(array_string))


def generate_datasets(
    positive_class_path: str,
    negative_class_path: str,
    *,
    train_size: float,
    test_size: float,
    random_seed: Union[int, np.random.BitGenerator, np.random.Generator],
    n_jobs=-1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Function to generate full train / test / val datasets with extracted features.
    Positive and negative classes in samples are balanced.
    Class label contains in column `is_flare`.
    """
    positive_extension = positive_class_path.split(".")[1]
    negative_extension = negative_class_path.split(".")[1]

    if positive_extension == "csv":
        positive_class = pd.read_csv(
            positive_class_path,
            converters={"mjd": array_from_string, "mag": array_from_string, "magerr": array_from_string},
        )
    elif positive_extension == "parquet":
        positive_class = pd.read_parquet(positive_class_path)
    else:
        raise TypeError(f"Incorrect format: {positive_extension}")

    if negative_extension == "csv":
        negative_class = pd.read_csv(
            negative_class_path,
            converters={"mjd": array_from_string, "mag": array_from_string, "magerr": array_from_string},
        )
    elif negative_extension == "parquet":
        negative_class = pd.read_parquet(negative_class_path)
    else:
        raise TypeError(f"Incorrect format: {negative_extension}")

    rng = np.random.default_rng(random_seed)

    positive_class["is_flare"] = np.ones(len(positive_class))
    negative_class["is_flare"] = np.zeros(len(negative_class))

    pos_train, pos_test, pos_val = data_splitter(
        positive_class, train_size=train_size, test_size=test_size, random_seed=rng
    )

    # we expect our negative class data is bigger than positive
    train_len, test_len, val_len = len(pos_train), len(pos_test), len(pos_val)

    neg_train, neg_test, neg_val = (
        negative_class[:train_len].reset_index(drop=True),
        negative_class[train_len : train_len + test_len].reset_index(drop=True),
        negative_class[train_len + test_len : train_len + test_len + val_len].reset_index(drop=True),
    )

    train_data = pd.concat([pos_train, neg_train], axis=0).sample(frac=1, random_state=rng).reset_index(drop=True)
    test_data = pd.concat([pos_test, neg_test], axis=0).sample(frac=1, random_state=rng).reset_index(drop=True)
    val_data = pd.concat([pos_val, neg_val], axis=0).sample(frac=1, random_state=rng).reset_index(drop=True)

    assert len(train_data) == len(pos_train) + len(neg_train)
    assert len(test_data) == len(pos_test) + len(neg_test)
    assert len(val_data) == len(pos_val) + len(neg_val)

    train_features = feature_extractor(train_data, n_jobs=n_jobs)
    print("train features extracted")

    test_features = feature_extractor(test_data, n_jobs=n_jobs)
    print("test features extracted")

    val_features = feature_extractor(val_data, n_jobs=n_jobs)
    print("val features extracted")

    feature_names = extractor.names
    feature_names.remove("bazin_fit_amplitude")
    feature_names.remove("bazin_fit_reference_time")

    return train_features, test_features, val_features, feature_names
