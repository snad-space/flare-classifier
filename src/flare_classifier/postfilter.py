import os
import time as time

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astroquery.imcce import Skybot
from tqdm import tqdm

PALOMAR = EarthLocation(
    lon=-116.863, lat=33.356, height=1706
)  # EarthLocation.of_site('Palomar')
# URLS_ZTF_DB = 'http://db.ztf.snad.space/api/v3/data/latest/oid/full/json?oid='

PATH_TO_PREDICT_DATA = '../data/predict-data/'
PATH_TO_MODELS = '../models/'


def get_coordinats_to_candidates(file_name: str) -> pd.DataFrame:
    import requests

    # url = 'https://db.ztf.snad.space/api/v3/data/latest/oid/full/json'
    url = 'https://sai.db.ztf.snad.space/api/v3/data/latest/oid/coord/json'
    chunk_size = 100
    print('Getting coord from ztf.snad.space...')
    df = pd.read_csv(file_name)

    ra = []
    dec = []
    amount_chunks = df.shape[0] // chunk_size + 1
    for i in tqdm(
        range(0, amount_chunks), total=amount_chunks, desc='Getting coords'
    ):
        try:
            temp_df = df[i * chunk_size : (i + 1) * chunk_size].reset_index(
                drop=True
            )
            list_oids = list(temp_df['oid'])
            if len(list_oids) == 0:
                continue

            params = dict(
                oid=list_oids,
            )
            resp = requests.get(url=url, params=params)
            data = resp.json()

            for oid in list_oids:
                ra.append(data[str(oid)]['ra'])
                dec.append(data[str(oid)]['dec'])
        except Exception as error:
            print(
                f'Error on read file, error description {error}, response text {resp.text}'
            )

            time.sleep(60)
            resp = requests.get(url=url, params=params)
            data = resp.json()

            for oid in list_oids:
                ra.append(data[str(oid)]['ra'])
                dec.append(data[str(oid)]['dec'])

    df.loc[:, 'ra'] = ra
    df.loc[:, 'dec'] = dec

    df.to_csv(
        file_name.replace('.csv', '_coord.csv'),
        index=False,
    )
    print('Done coord from ztf.snad.space')
    return df


def get_data_to_postfilter(threshold: float) -> pd.DataFrame:
    columns_info = [
        'field',
        'oid',
        'flare_start',
        'time_maximum',
        'fwhm_start',
        'fwhm_maximum',
        'min_prob',
    ]
    result = pd.DataFrame(columns=columns_info)
    files = [f for f in os.listdir(PATH_TO_PREDICT_DATA)]
    for file in tqdm(files, total=len(files), desc='Data progress'):
        # if not a prediction file - pass it
        if file.find('predict_field_') == -1:
            continue
        df = pd.read_csv(PATH_TO_PREDICT_DATA + file)
        field = file.replace('predict_field_', '').replace('.csv', '')
        df = df[df['min_prob'] > threshold].reset_index(drop=True)
        df['field'] = field
        df = df[columns_info]
        result = pd.concat([result, df]).reset_index(drop=True)

    result['diff_fwhm'] = result['fwhm_start'] - result['fwhm_maximum']
    result.to_csv(
        PATH_TO_PREDICT_DATA + f'all_fields_threshold-{threshold}.csv',
        index=False,
    )
    return result


def hmjd_to_earth(hmjd, coord):
    t = Time(hmjd, format='mjd')
    return t - t.light_travel_time(coord, kind='heliocentric', location=PALOMAR)


def convert_to_list(item):
    """Convert string representation of data to list"""

    data_row = (
        item.replace('[(', '#')
        .replace(')]', '#')
        .replace('),(', '#')
        .replace(',', '#')
        .replace('[', '#')
        .replace(']', '#')
        .split('#')
    )

    data_row = [element for element in data_row if element]
    return list(map(float, data_row))


def get_data_from_release(candidates_file, release_file, threshold=0.99):
    """Get time series data and ra/dec for candidates of flare"""

    candidates = pd.read_csv(candidates_file.file_name)
    candidates = candidates[
        candidates[candidates_file.predict_column] >= threshold
    ]
    candidates = candidates[
        ['oid', 'flare_start', candidates_file.predict_column]
    ]
    candidates.columns = ['oid', 'flare_start', 'predict_prob']

    data = pd.read_csv(release_file, chunksize=10_000)

    result = pd.DataFrame(
        {
            'oid': pd.Series(dtype='int'),
            'flare_start': pd.Series(dtype='float'),
            'predict_prob': pd.Series(dtype='float'),
            'time_series': pd.Series(dtype='object'),
            'ra': pd.Series(dtype='float'),
            'dec': pd.Series(dtype='float'),
        }
    )

    for chunk in tqdm(data, total=9729, desc='Data progress'):
        oids = pd.Series(chunk.iloc[:, 5]).reset_index(drop=True)
        data_list = pd.Series(map(convert_to_list, chunk.iloc[:, 6]))
        flare_start = pd.Series(map(lambda x: x[1], data_list)).reset_index(
            drop=True
        )
        data_list = data_list.reset_index(drop=True)
        dec = pd.Series(chunk.iloc[:, 7]).reset_index(drop=True)
        ra = pd.Series(chunk.iloc[:, 8]).reset_index(drop=True)

        chunk_df = pd.DataFrame(
            data={
                'oid': oids,
                'flare_start': flare_start,
                'time_series': data_list,
                'dec': dec,
                'ra': ra,
            }
        )

        merge_df = pd.merge(
            candidates, chunk_df, on=['oid', 'flare_start'], how='inner'
        )
        result = pd.concat([result, merge_df])

        if result.shape[0] == candidates.shape[0]:
            break

    result = result.reset_index(drop=True)
    print(f'result dataframe={result.shape}')
    return result


def asteroid_search(ra, dec, time_max):
    """Search asteroid in radius at time maximum of the light curve"""

    RADIUS_SEARCH_ASTEROID = 0.0041667

    field = SkyCoord(ra * u.deg, dec * u.deg)
    earthed_time = hmjd_to_earth(time_max, field)
    try:
        results = Skybot.cone_search(
            field, RADIUS_SEARCH_ASTEROID * u.deg, earthed_time, location='I41'
        )
        return len(results) > 0, results[0]['Name']
    except Exception as e:
        if len(e.args) > 0 and e.args[0][0:22] == 'No solar system object':
            return False, 'No object'
        else:
            print(f'Exception {e.args[0]}')
            return False, 'Service error'


def get_flares_time(df, add_flare_start=False):
    """Determine the time of the beginning and maximum of the light curve"""

    df['time_maximum'] = 0.0
    if add_flare_start:
        df['flare_start'] = 0.0

    for i, obs in tqdm(df.iterrows(), total=df.shape[0]):
        time = obs['time_series']
        mag = obs['time_series']
        if isinstance(time, str):
            time = convert_to_list(time)[1::4]
        if isinstance(mag, str):
            mag = convert_to_list(mag)[2::4]

        time_max = np.argmin([m for m, t in zip(mag, time)])

        df.loc[i, 'time_maximum'] = time[time_max]
        if add_flare_start:
            df.loc[i, 'flare_start'] = time[0]

    return df


def search_asteroids(df: pd.DataFrame):
    """Search asteroids in dataframe"""

    df['asteroid'] = False
    df['asteroid_name'] = ''

    for i, obs in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            asteroid, asteroid_name = asteroid_search(
                obs['ra'], obs['dec'], obs['time_maximum']
            )
        except Exception as e:
            if len(e.args) > 0:
                print(f'Exception message {e.args[0]}')

            time.sleep(60)
            asteroid, asteroid_name = asteroid_search(
                obs['ra'], obs['dec'], obs['time_maximum']
            )

        df.loc[i, 'asteroid'] = asteroid
        df.loc[i, 'asteroid_name'] = asteroid_name

    return df


def get_sharpness(df):
    """Get sharpness of source for all dataframe rows"""
    import io
    from urllib.request import urlopen

    from astropy.io.votable import parse

    URL_CALTECH = (
        'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='
    )
    HALF_EXPOSION_TIME = 0.000174  # 15 sec

    if 'sharpness_flare_start' not in df.columns:
        df['sharpness_flare_start'] = np.NaN
        df['sharpness_time_maximum'] = np.NaN
        df['exposure_id_start'] = 0
        df['exposure_id_maximum'] = 0

    for i, obs in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.notna(obs['sharpness_flare_start']):
            continue

        field = SkyCoord(obs['ra'] * u.deg, obs['dec'] * u.deg)
        start_time_request = (
            hmjd_to_earth(obs['flare_start'], field) - HALF_EXPOSION_TIME
        )
        end_time_request = (
            hmjd_to_earth(obs['time_maximum'], field) + HALF_EXPOSION_TIME
        )

        url_request = (
            URL_CALTECH
            + str(obs['oid'])
            + '&TIME='
            + str(start_time_request)
            + '%20'
            + str(end_time_request)
        )

        try:
            response = urlopen(url_request)
            stream = io.BytesIO(response.read())
            votable = parse(stream)
            table_data = votable.get_first_table().to_table(
                use_names_over_ids=True
            )

        except Exception as e:
            if len(e.args) > 0:
                print(f'Exception message {e.args[0]}')

            time.sleep(60)

            response = urlopen(url_request)
            stream = io.BytesIO(response.read())
            votable = parse(stream)
            table_data = votable.get_first_table().to_table(
                use_names_over_ids=True
            )

        if len(table_data) > 0:
            df.loc[i, 'sharpness_flare_start'] = table_data[0]['sharp']
            df.loc[i, 'sharpness_time_maximum'] = table_data[-1]['sharp']
            df.loc[i, 'exposure_id_start'] = table_data[0]['expid']
            df.loc[i, 'exposure_id_maximum'] = table_data[-1]['expid']

        if i % 100 == 0:
            # each 100 row will write to disk
            df.to_csv('temp_sharpness.csv')

    df['diff_sharpness'] = df.sharpness_flare_start - df.sharpness_time_maximum
    return df


def get_FWHM(prediction_file, exposure_file):
    """Get FWHM from exposure table of metadata"""
    data_expos = pd.read_csv(exposure_file)
    data_expos = data_expos[data_expos['filter'] == 'r']

    predict_data = pd.read_csv(prediction_file)

    s_df = pd.merge(
        predict_data[
            [
                'oid',
                'flare_start',
                'predict_prob',
                'time_series',
                'm_label',
                'ra',
                'dec',
                'asteroid',
                'asteroid_name',
                'time_maximum',
                'sharpness_flare_start',
                'sharpness_time_maximum',
                'diff_sharpness',
                'exposure_id_start',
                'exposure_id_maximum',
            ]
        ],
        data_expos[['expid', 'fwhm']],
        left_on='exposure_id_start',
        right_on='expid',
        how='left',
    )
    s_df.rename(columns={'fwhm': 'fwhm_start'}, inplace=True)
    s_df = pd.merge(
        s_df[
            [
                'oid',
                'flare_start',
                'predict_prob',
                'time_series',
                'm_label',
                'ra',
                'dec',
                'asteroid',
                'asteroid_name',
                'time_maximum',
                'sharpness_flare_start',
                'sharpness_time_maximum',
                'diff_sharpness',
                'exposure_id_start',
                'exposure_id_maximum',
                'fwhm_start',
            ]
        ],
        data_expos[['expid', 'fwhm']],
        left_on='exposure_id_maximum',
        right_on='expid',
        how='left',
    )
    s_df.rename(columns={'fwhm': 'fwhm_maximum'}, inplace=True)
    s_df['diff_fwhm'] = s_df['fwhm_start'] - s_df['fwhm_maximum']

    return s_df.drop('expid', axis=1)


def convert_db_to_csv_files(path_to_data_dir, file_name):
    """Convert sqlite database file to csv files for each table"""
    import sqlite3

    db = sqlite3.connect(path_to_data_dir + file_name)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query('SELECT * from %s' % table_name, db)
        table.to_csv(
            path_to_data_dir + table_name + '.csv', index_label='index'
        )
    cursor.close()
    db.close()


if __name__ == '__main__':
#     import os

#     import pandas as pd

#     print(os.getcwd())

#     result = get_coordinats_to_candidates(
#         '../data/predict-data/all_fields_0.98.csv'
#     )
    # result = get_data_to_postfilter(threshold=0.9)
    # result = get_coordinats_to_candidates(
    #     './data/predict-data/all_fields_threshold-0.9.csv'
    # )
    # result = pd.read_csv('./data/predict-data/all_fields_threshold-0.9_coord.csv')
    # result = search_asteroids(result)
    # result.to_csv('./data/predict-data/all_fields_threshold-0.9_coord.csv', index=False)
    # result_sharpness = get_sharpness(result)
    # result_sharpness.to_csv('./data/predict-data/all_fields_threshold-0.9_sharpness.csv', index=False)

    # postfilter_df = pd.read_csv(
    #     './data/predict-data/all_fields_threshold-0.9_sharpness.csv',
    # ).dropna()
    # # run only debug ----start
    # postfilter_df['diff_fwhm'] = (
    #     postfilter_df['fwhm_start'] - postfilter_df['fwhm_maximum']
    # )
    # # run only debug ----end
    # # load models
    # scaler = pickle.load(open(PATH_TO_MODELS + 'scaler_postfilter.pkl', 'rb'))
    # logreg = pickle.load(open(PATH_TO_MODELS + 'logreg_postfilter.pkl', 'rb'))
    # feature_cols = [
    #     'sharpness_flare_start',
    #     'sharpness_time_maximum',
    #     'diff_sharpness',
    #     'fwhm_start',
    #     'fwhm_maximum',
    #     'diff_fwhm',
    # ]
    # x = postfilter_df[feature_cols]
    # x_sc = scaler.transform(x)
    # postfilter_df['postfilter_prob'] = logreg.predict_proba(x_sc)[::, 1]
    # postfilter_df.to_csv(
    #     './data/predict-data/all_fields_postfilter_predict.csv', index=False
    # )
    # print('prediction done')
