import pandas as pd
import numpy as np


def read_initial_csv(path_to_file:str, columns_list):
    return pd.read_csv(
        path_to_file,
        converters=dict.fromkeys(
            columns_list,
            lambda s: np.fromstring(s[1:-1], sep=' ', dtype=float)
        )
    )