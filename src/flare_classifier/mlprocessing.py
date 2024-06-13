import os
import pickle

import light_curve as lc
import matplotlib.pyplot as plt
# import pyarrow.parquet as pq
from pyarrow.parquet import read_table
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from minio import Minio
from scipy.interpolate import Akima1DInterpolator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket
from tqdm import tqdm

load_dotenv()

DEBUG = False

N_GRID = 100
PATH_TO_DATA = '/home/jupyter/datasphere/s3/ztf-high-cadence-data/dr17/'
PATH_TO_REAL_DATA = '/home/jupyter/datasphere/s3/ztf-high-cadence-data/dr17/full_data/'

PATH_TO_MODELS = '../models/'
PATH_TO_TRAIN_DATA = '../data/train-data/'
PATH_TO_PREDICTED_DATA = '../data/predict-data/'

POSITIVE_DATA = 'dr17_generated_1m_v2.parquet'
NEGATIVE_DATA = 'dr17_negative_new.parquet'
REAL_FLARES_FILE = 'real_flares_validation.csv'

VAL_NEGATIVE_AMOUNT = 1_000
PCA_PROC = 90

LIGHT_CURVE_FEATURES = [
    'bazin_fit_rise_time',
    'median_buffer_range_percentage_10',
    'amplitude',
    'percent_difference_magnitude_percentile_5',
    'otsu_std_lower',
    'eta',
    'bazin_fit_reference_time',
    'magnitude_percentage_ratio_40_5',
    'anderson_darling_normal',
    'standard_deviation',
    'bazin_fit_reduced_chi2',
    'otsu_lower_to_all_ratio',
    'median',
    'bazin_fit_fall_time',
    'eta_e',
    'median_absolute_deviation',
    'beyond_1_std',
    'skew',
    'weighted_mean',
    'bazin_fit_baseline',
    'excess_variance',
    'stetson_K',
    'cusum',
    'inter_percentile_range_25',
    'mean',
    'otsu_mean_diff',
    'percent_amplitude',
    'beyond_2_std',
    'otsu_std_upper',
    'maximum_slope',
    'kurtosis',
    'mean_variance',
]


# this class need for calibrate original Catboost model
class CatBoostClassifierCalibrate(CatBoostClassifier):
    def predict_proba(self, X):
        return super().predict_proba(X)


def get_light_curve_features(ts_data: list) -> pd.DataFrame:
    """Calculate light curve features from time series data

    Args:
        ts_data (list of triple): list of triple (time, mag, mag_err)

    Returns:
        pd.DataFrame: dataframe with calc light curve features
    """

    amplitude = lc.Amplitude()
    anderson_test = lc.AndersonDarlingNormal()
    bazin_fit = lc.BazinFit(algorithm='mcmc')
    beyond_1_std = lc.BeyondNStd(1.0)
    beyond_2_std = lc.BeyondNStd(2.0)
    cusum = lc.Cusum()
    eta = lc.Eta()
    eta_e = lc.EtaE()
    excess_var = lc.ExcessVariance()
    interp_range = lc.InterPercentileRange(quantile=0.25)
    kurtosis = lc.Kurtosis()
    mag_perc_ratio = lc.MagnitudePercentageRatio(
        quantile_numerator=0.4, quantile_denominator=0.05
    )
    max_slope = lc.MaximumSlope()
    mean = lc.Mean()
    mean_var = lc.MeanVariance()
    median = lc.Median()
    median_abs_dev = lc.MedianAbsoluteDeviation()
    median_buff_range_perc = lc.MedianBufferRangePercentage(quantile=0.1)
    otsu_split = lc.OtsuSplit()
    percent_ampl = lc.PercentAmplitude()
    percent_diff_mag = lc.PercentDifferenceMagnitudePercentile(quantile=0.05)
    skew = lc.Skew()
    standard_dev = lc.StandardDeviation()
    stetson_k = lc.StetsonK()
    weighted_mean = lc.WeightedMean()

    features_set = [
        amplitude,
        anderson_test,
        bazin_fit,
        beyond_1_std,
        beyond_2_std,
        cusum,
        eta,
        eta_e,
        excess_var,
        interp_range,
        kurtosis,
        mag_perc_ratio,
        max_slope,
        mean,
        mean_var,
        median,
        median_abs_dev,
        median_buff_range_perc,
        otsu_split,
        percent_ampl,
        percent_diff_mag,
        skew,
        standard_dev,
        stetson_k,
        weighted_mean,
    ]

    EXTRACTOR = lc.Extractor(*features_set)
    results = EXTRACTOR.many(
        ts_data,
        n_jobs=32,
        sorted=True,
        check=False,
    )
    results_df = pd.DataFrame(results, columns=EXTRACTOR.names)
    return results_df[LIGHT_CURVE_FEATURES]


def read_initial_csv(path_to_file: str, columns_list: list) -> pd.DataFrame:
    """Read initial csv file and convert text fields from columns_list to float

    Args:
        path_to_file (str): full path to initial csv file
        columns_list (list): list of fields to convert

    Returns:
        pd.DataFrame: converted dataframe
    """
    separator = ' '
    data = pd.read_csv(path_to_file, nrows=1)
    if len(data.loc[0, columns_list[0]].split(',')) > 1:
        separator = ','

    return pd.read_csv(
        path_to_file,
        converters=dict.fromkeys(
            columns_list,
            lambda s: np.fromstring(s[1:-1], sep=separator, dtype=float),
        ),
    )


def read_and_interpolate_data(
    initial_data: pd.DataFrame, n_grid: int, target_column: list
) -> pd.DataFrame:
    """Read raw data, interpolate time series data
        and add light curve features to output dataframe.

    Args:
        initial_data (pd.DataFrame): dataframe from csv file
        n_grid (int): amount of interpolated points
        target_column (list): name of column with class label

    Returns:
        pd.DataFrame: converted and interpolated dataframe
    """

    # interpolate time series by Akima1DInterpolator
    columns = (
        ['class']
        # interpolated time series points
        + [f'mag_{i}' for i in range(n_grid)]
        # additional feature from time series
        + ['mag_min']
    )

    intertpolated_data = []
    list_lc = []

    print('Interpolate data')
    for _, item in tqdm(
        initial_data.iterrows(),
        total=initial_data.shape[0],
        desc='Interpolate progress',
    ):
        row = []
        row.append(int(item[target_column]))
        # interpolate magnitude data
        x = item['mjd']
        y = item['mag']
        errors = item['magerr']
        # light curve features
        list_lc.append(
            (
                np.array(x).astype(np.float64),
                np.array(y).astype(np.float64),
                np.array(errors).astype(np.float64),
            )
        )

        # subtract a minimum magnitude for normalize
        y_min = min(y)
        y = [(z - y_min) for z in y]

        # check input data
        if len(x) != len(y):
            if len(x) != len(y):
                print(
                    f'len time points={len(x)} not equal len mag points={len(y)}'
                    f' in data for obs_id={row[0]} object'
                )
                continue
        interpolator = Akima1DInterpolator(x, y)
        xnew = np.linspace(min(x), max(x), n_grid)
        ynew = interpolator(xnew)
        # add new time series points to output list
        row += list(ynew)
        # add simple time series feature
        row += [y_min]
        intertpolated_data.append(row)

    ts_df = pd.DataFrame(data=intertpolated_data, columns=columns)
    # get light curve features
    light_curve_features_df = get_light_curve_features(list_lc)
    # concat TS data and light curve features
    data_inter = pd.concat([ts_df, light_curve_features_df], axis=1)

    return data_inter


def run_preprocessing(n_grid, n_samples=1_000):
    """Performs steps of preprocessing

    Args:
        n_grid (int): The number of interpolation points
        n_samples (int, optional): Number samples for debug mode. Defaults to 1_000.

    Returns:
        preprocessing_out: dictionary with learning data and models for transform data
        keys of dict:
        - X_train: np.array with data for learing
        - X_test: np.array with data for test
        - y_train: np.array  target for X_train
        - y_test: np.array  target for X_test
        - scaler: model for scaling
        - mini_rocket: model for MINI Rocket transformation
        - pca: model for PCA transformation
        - features_list: list of features for training
        - X_val:  np.array with data for validation
        - y_val:  np.array target for validation
    """
    print('read data')
    if DEBUG:
        print(f'current path {os.getcwd()}')
    tab = read_table(PATH_TO_DATA + NEGATIVE_DATA)
    negative_all_data = tab.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    negative_all_data = negative_all_data[['mjd', 'mag', 'magerr']]
    tab = read_table(PATH_TO_DATA + POSITIVE_DATA)
    positive_data = tab.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
    positive_data = positive_data[['mjd', 'mag', 'magerr']]
    print(f'negative_class_shape {negative_all_data.shape}')
    print(f'positive_class_shape {positive_data.shape}')
    negative_data = negative_all_data.sample(
        n=positive_data.shape[0], random_state=42
    )
    negative_val = negative_all_data.drop(negative_data.index).sample(
        n=VAL_NEGATIVE_AMOUNT, random_state=42
    )

    if DEBUG:
        negative_data = negative_data.sample(n=n_samples, random_state=42)
        positive_data = positive_data.sample(n=n_samples, random_state=42)

    negative_val['class'] = 0
    negative_data['class'] = 0
    positive_data['class'] = 1
    initial_data = pd.concat([negative_data, positive_data])

    # interpolate time series by Akima1DInterpolator
    data = read_and_interpolate_data(initial_data, n_grid, 'class')

    data.to_parquet(PATH_TO_TRAIN_DATA + 'data_to_train.parquet')

    x = data.drop(['class'], axis=1)
    y = data['class']

    # # scaling data
    print('scale data')
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    # convert to 3D np.array for sktime format data
    X_ts_train = X_train[:, np.newaxis, 0:100]
    X_ts_test = X_test[:, np.newaxis, 0:100]

    print('mini rocket transform execute...')
    mini_rocket = MiniRocket(
        random_state=42, n_jobs=8
    )  # by default, MiniRocket uses 10,000 kernels
    mini_rocket.fit(X_ts_train)

    X_ts_train_transform = mini_rocket.transform(X_ts_train)
    X_ts_test_transform = mini_rocket.transform(X_ts_test)
    print('End mini rocket transform')

    print('PCA started...')
    pca = PCA(n_components=PCA_PROC / 100.0, random_state=42)
    pca.fit(X_ts_train_transform)
    print(
        f'PCA n_components={pca.n_components_} for {PCA_PROC}% of the variance in the train data'
    )

    pca_plot = PCA().fit(X_ts_train_transform)
    plt.plot(np.cumsum(pca_plot.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.xlim(0, 200)

    X_train_PCA = pca.transform(X_ts_train_transform)
    X_test_PCA = pca.transform(X_ts_test_transform)
    print('PCA end')

    # join time series data and features
    X_train_final = np.concatenate((X_train_PCA, X_train[:, 100:]), axis=1)
    X_test_final = np.concatenate((X_test_PCA, X_test[:, 100:]), axis=1)

    # get validation dataset
    print('Create validation dataset')
    X_val, y_val = get_validation_data(
        n_grid=100,
        negative_val=negative_val,
        scaler=scaler,
        mini_rocket=mini_rocket,
        pca=pca,
    )

    # save all preprocessing output
    # save models
    pickle.dump(scaler, open(PATH_TO_MODELS + 'scaler.pkl', 'wb'))
    pickle.dump(
        mini_rocket, open(PATH_TO_MODELS + 'MiniRocket_transform.pkl', 'wb')
    )
    pickle.dump(pca, open(PATH_TO_MODELS + 'PCA.pkl', 'wb'))
    # data
    np.savetxt(PATH_TO_TRAIN_DATA + 'X_train.csv', X_train_final, delimiter=',')
    np.savetxt(PATH_TO_TRAIN_DATA + 'X_test.csv', X_test_final, delimiter=',')
    np.savetxt(PATH_TO_TRAIN_DATA + 'y_train.csv', y_train, delimiter=',')
    np.savetxt(PATH_TO_TRAIN_DATA + 'y_test.csv', y_test, delimiter=',')
    np.savetxt(PATH_TO_TRAIN_DATA + 'X_val.csv', X_val, delimiter=',')
    np.savetxt(PATH_TO_TRAIN_DATA + 'y_val.csv', y_val, delimiter=',')
    # utility data
    features_list = [
        f'PCA_{i}' for i in range(1, pca.n_components_ + 1)
    ] + list(data.drop(['class'], axis=1).columns[100:])
    print(f'List of result features ({len(features_list)})  {features_list}')

    pickle.dump(features_list, open(PATH_TO_MODELS + 'features_list.pkl', 'wb'))

    print('End preprocessing')
    preproc_out = {}
    preproc_out['X_train'] = X_train_final
    preproc_out['X_test'] = X_test_final
    preproc_out['y_train'] = y_train
    preproc_out['y_test'] = y_test
    preproc_out['scaler'] = scaler
    preproc_out['mini_rocket'] = mini_rocket
    preproc_out['pca'] = pca
    preproc_out['features_list'] = features_list
    preproc_out['X_val'] = X_val
    preproc_out['y_val'] = y_val
    return preproc_out


def get_validation_data(n_grid, negative_val, scaler, mini_rocket, pca):
    """Prepare validation dataset

    Args:
        n_grid (int): _description_
        negative_val (pd.DataFrame): samples of negative class
        scaler (model): scaler transformer
        mini_rocket (model): MINI Rocket transformer
        pca (model): PCA transformer

    Returns:
        X_val, y_val: np.array features and target for validation
    """

    real_flares = pd.read_csv(
        PATH_TO_DATA + REAL_FLARES_FILE,
        converters=dict.fromkeys(
            ['mjd', 'mag', 'magerr'],
            lambda s: np.fromstring(s[1:-1], sep=' ', dtype=float),
        ),
    )
    negative_val['oid'] = None
    initial_data = pd.concat([real_flares, negative_val], axis=0)
    data = read_and_interpolate_data(initial_data, n_grid, 'class')
    print(f"Validation balance classes: {data.value_counts('class')}")

    x = data.drop(['class'], axis=1)
    y = data['class']

    # scaling data
    x = scaler.transform(x)

    # convert to 3D np.array for sktime format data
    x_ts = x[:, np.newaxis, 0:100]
    x_ts_val = mini_rocket.transform(x_ts)
    x_val_pca = pca.transform(x_ts_val)
    x_val = np.concatenate((x_val_pca, x[:, 100:]), axis=1)

    return x_val, y


def metrics_calc(
    df: pd.DataFrame,
    model_name: str,
    y: np.array,
    predict: np.array,
    predict_proba: np.array,
) -> pd.DataFrame:
    """Calculate metrics

    Args:
        df (pd.DataFrame): dataframe with calc results
        model_name (str): name of model
        y (np.array): ground truth
        predict (np.array): prediction
        predict_proba (np.array): proba of prediction

    Returns:
        pd.DataFrame: dataframe with calc results
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    df.loc[model_name, :] = [
        average_precision_score(y, predict_proba[:, 1]),
        accuracy_score(y, predict),
        roc_auc_score(y, predict_proba[:, 1]),
        f1_score(y, predict),
        precision_score(y, predict),
        sum(predict),
        recall_score(y, predict),
        fbeta_score(y, predict, beta=0.1),
        fbeta_score(y, predict, beta=0.3),
        fbeta_score(y, predict, beta=0.5),
    ]
    return df


def get_metric_threshold(
    y_pred_proba: np.array, y_val: np.array
) -> pd.DataFrame:
    """Calculates metrics for range of thresholds

    Args:
        y_pred_proba (np.array): predicted proba
        y_val (np.array): ground truth

    Returns:
        pd.DataFrame: dataframe with calc results
    """
    df_metric = pd.DataFrame(
        columns=[
            'average_precision_score',
            'accuracy',
            'ROC_AUC',
            'F1 score',
            'precision',
            'count_objects',
            'recall',
            'f-beta score (beta=0.1)',
            'f-beta score (beta=0.3)',
            'f-beta score (beta=0.5)',
        ]
    )

    for t in np.arange(0.1, 1.0, 0.1):
        threshold = 1 - t
        y_pred = np.where(y_pred_proba[:, 1] >= threshold, 1, 0)
        df_metric = metrics_calc(
            df_metric,
            f'threshold={threshold:.1f}',
            y_val,
            y_pred,
            y_pred_proba,
        )

    return df_metric


def interpolate_mag(row: tuple, n_grid: int = N_GRID):
    """Interpolate time series magniture data

    Args:
        tuple(np.array, np.array) - mjd and mag data
        n_grid:int number of interpolation points

    Returns:
        tuple(np.array, np.array, float) - interpolated mag and time, mag_min
    """
    mag = row[1]
    mag_min = min(mag)
    mag = [(z - mag_min) for z in mag]
    interpolator = Akima1DInterpolator(row[0], mag)
    t_new = np.linspace(min(row[0]), max(row[0]), n_grid)
    mag_new = interpolator(t_new)
    return (mag_new, t_new, mag_min)


def predict_real_data():
    """Perform process to predict class on real data.
    Procedure read files from PATH_TO_REAL_DATA, make prediction
    and saves result to csv files to PATH_TO_PREDICTED_DATA
    """
    columns_info = [
        'oid',
        'flare_start',
        'time_maximum',
        'fwhm_start',
        'fwhm_maximum',
        'RF_proba_class_0',
        'RF_proba_class_1',
        'CB_proba_class_0',
        'CB_proba_class_1',
        'min_prob',
    ]
    # load models
    scaler = pickle.load(open(PATH_TO_MODELS + 'scaler.pkl', 'rb'))
    mini_rocket = pickle.load(
        open(PATH_TO_MODELS + 'MiniRocket_transform.pkl', 'rb')
    )
    pca = pickle.load(open(PATH_TO_MODELS + 'PCA.pkl', 'rb'))
    clf_rf = pickle.load(
        open(PATH_TO_MODELS + 'calibrated_RandomForestIsotonic.pkl', 'rb')
    )
    clf_cb = pickle.load(
        open(PATH_TO_MODELS + 'calibrated_CatboostIsotonic.pkl', 'rb')
    )

    files = [f for f in os.listdir(PATH_TO_REAL_DATA)]

    if DEBUG:
        files = [
            'field_0257.parquet',
            'field_0258.parquet',
            'field_0259.parquet',
            'field_0260.parquet',
        ]

    amount_predicted_obs = 0
    chunk_size = 10_000
    for file in tqdm(files, total=len(files), desc='Data progress'):
        result = pd.DataFrame(columns=columns_info)

        if os.path.exists(
            PATH_TO_PREDICTED_DATA
            + file.replace('field', 'predict_field').replace(
                '.parquet', '.csv'
            )
        ):
            continue

        print(f'read file {PATH_TO_REAL_DATA + file}')
        # pfile = pd.read_parquet(PATH_TO_REAL_DATA + file)
        tab = read_table(PATH_TO_REAL_DATA + file)
        pfile = tab.to_pandas(types_mapper=pd.ArrowDtype, ignore_metadata=True)
        # split file on chunk, iter_batch don't work in DataSphere right now
        for i in range(0, pfile.shape[0] // chunk_size + 1):
            # file_name = file.replace('field', f'temp/field_{i}')
            data_field = pfile[
                i * chunk_size : (i + 1) * chunk_size
            ].reset_index(drop=True)

            data_field['flare_start'] = pd.Series(
                map(lambda x: x[0], data_field['mjd'])
            )
            data_field['argmin'] = pd.Series(
                map(np.argmin, data_field['mag'])
            )
            data_field['time_maximum'] = pd.Series(
                map(
                    lambda x, i: x[i],
                    data_field['mjd'],
                    data_field['argmin'],
                )
            )
            data_field['fwhm_start'] = pd.Series(
                map(lambda x: x[0], data_field['fwhm'])
            )
            data_field['fwhm_maximum'] = pd.Series(
                map(
                    lambda x, i: x[i],
                    data_field['fwhm'],
                    data_field['argmin'],
                )
            )
            data_field.drop(['argmin'], axis=1)

            triple_data = list(
                map(interpolate_mag, zip(data_field.mjd, data_field.mag))
            )
            temp_df = pd.DataFrame(
                data=triple_data, columns=['mag', 'time', 'mag_min']
            )

            ts_inter = np.stack(temp_df['mag'].values)
            mag_min = temp_df['mag_min'].to_numpy()
            mag_min = mag_min[:, np.newaxis]
            x_np = np.hstack((ts_inter, mag_min))
            x_full = np.concatenate(
                (x_np, np.stack(data_field[LIGHT_CURVE_FEATURES].values)),
                axis=1,
            )
            x_full = scaler.transform(x_full)

            # convert to 3D np.array for sktime format data
            x_ts = x_full[:, np.newaxis, 0:100]
            x_ts_rocket = mini_rocket.transform(x_ts)
            x_ts_pca = pca.transform(x_ts_rocket)

            x = np.concatenate((x_ts_pca, x_full[:, 100:]), axis=1)

            y_pred_proba_rf = clf_rf.predict_proba(x)
            y_pred_proba_cb = clf_cb.predict_proba(x)

            temp_result = data_field[columns_info[:-5]]
            temp_result['RF_proba_class_0'] = y_pred_proba_rf[:, 0]
            temp_result['RF_proba_class_1'] = y_pred_proba_rf[:, 1]
            temp_result['CB_proba_class_0'] = y_pred_proba_cb[:, 0]
            temp_result['CB_proba_class_1'] = y_pred_proba_cb[:, 1]
            temp_result['min_prob'] = np.minimum(
                temp_result['RF_proba_class_1'].to_numpy(),
                temp_result['CB_proba_class_1'].to_numpy(),
            )
            result = pd.concat([result, temp_result])

            amount_predicted_obs = (
                amount_predicted_obs + temp_result.shape[0]
            )

        result.to_csv(
            PATH_TO_PREDICTED_DATA
            + file.replace('field', 'predict_field').replace(
                '.parquet', '.csv'
            ),
            index=False,
        )

        if DEBUG:
            os.remove(file)


    print(f'Prediction done, amount prediction obs {amount_predicted_obs}')   


if __name__ == '__main__':
    # preproc_out = run_preprocessing(n_grid=100, n_samples=1_000)
    # predict_real_data()
