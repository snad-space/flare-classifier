import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from scipy.interpolate import CubicSpline


@dataclass
class FeaturesExtractor:
    # name of Interpolator from scipy.interpolate
    interpolator: str = "CubicSpline"
    # path to csv file with data
    path_to_file: str = ""
    # number of grid points
    n_grid: int = 10

    def convert_row_to_array_to_float(self, item, column_name, verbose=0):
        """Convert string to float ndarray"""
        try:
            x = np.array([float(z) for z in re.split('\n | ', item[column_name][1:-1]) if z != '' ])
        except ValueError:
            if verbose==1:
                print(
                    f'Error convert {column_name}'
                    f'data for TIC={item["TIC"]} object'
                    f'cadence_id={item["cadence_id"]}'
                )
            x = np.array([])
        return x
        
    def fit_interpolator(self, x, y):
        """Fit interpolator for initial data"""
        interpolator_object = None
        if self.interpolator == "CubicSpline":
            interpolator_object = CubicSpline(x, y)
        else:
            raise ValueError("Undefined interpolator name")
        return interpolator_object

    def get_data(self, feature_label, verbose=0) -> pd.DataFrame:
        """Return dataframe with interpolated by interpolator object
        (cubic spline by default) n_grid points."""
        features_columns = [f'{feature_label}_{i}' for i in range(self.n_grid)]
        data = pd.DataFrame(columns=['TIC', 'cadence_id'] + features_columns)
        initial_data = pd.read_csv(self.path_to_file)
        
        for index, item in initial_data.iterrows():
            x = self.convert_row_to_array_to_float(item, column_name='mjd')
            y = self.convert_row_to_array_to_float(item, column_name=feature_label)
            # check input data
            if len(x)==0 or len(y) == 0 or len(x) != len(y):
                if verbose == 1 and len(x) != len(y):
                    print(
                        f'len time points={len(x)} not equl len mag points={len(y)}'
                        f' in data for TIC={item["TIC"]} object'
                        f' cadence_id={item["cadence_id"]}')
                    continue
            interpolator = self.fit_interpolator(x, y)
            xnew = np.linspace(x.min(), x.max(), self.n_grid)
            # add features to output datafrzme
            data.loc[index,:] = [item["TIC"], item["cadence_id"]] + list(interpolator(xnew))
        
        return data

if (__name__ == '__main__'):
    import time
    start_time = time.time()
    print('Start feature extract')
    FE = FeaturesExtractor(
        path_to_file='generated_250k.csv',
        n_grid = 20
    )
    data = FE.get_data(feature_label='mag')
    print(data.head())
    print(f'End features extract time execution = {(time.time() - start_time):.2f} seconds')