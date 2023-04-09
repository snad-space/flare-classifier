import pandas as pd
import numpy as np


def read_initial_csv(path_to_file:str, columns_list):
    
    separator = ' '
    data = pd.read_csv(path_to_file, nrows=1)
    if len(data.loc[0,columns_list[0]].split(',')) > 1:
        separator = ','
    
    return pd.read_csv(
        path_to_file,
        converters=dict.fromkeys(
            columns_list,
            lambda s: np.fromstring(s[1:-1], sep=separator, dtype=float)
        )
    )

def save_to_parquet(path_to_file:str, columns_list, path_parquet_file:str):
    data = read_initial_csv(path_to_file, columns_list)
    data.to_parquet(
        path_parquet_file,
        compression='gzip'
    )
    return