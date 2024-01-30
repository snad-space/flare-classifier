import numpy as np
import pandas as pd
from tqdm import tqdm
import time as time

import astropy.units as u

from astroquery.imcce import Skybot
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

PALOMAR = EarthLocation(lon=-116.863, lat=33.356, height=1706)  # EarthLocation.of_site('Palomar')
# URLS_ZTF_DB = 'http://db.ztf.snad.space/api/v3/data/latest/oid/full/json?oid='

def hmjd_to_earth(hmjd, coord):
    t = Time(hmjd, format="mjd")
    return t - t.light_travel_time(coord, kind="heliocentric", location=PALOMAR)

def convert_to_list(item):
    """Convert string representation of data to list"""
    
    data_row = item.replace('[(','#') \
                    .replace(')]','#') \
                    .replace('),(','#') \
                    .replace(',','#') \
                    .replace('[','#') \
                    .replace(']','#') \
                    .split('#')
    
    data_row = [element for element in data_row if element]
    return list(map(float, data_row))

def get_data_from_release(candidates_file, release_file, threshold=0.99):
    """Get time series data and ra/dec for candidates of flare"""
    
    candidates = pd.read_csv(candidates_file.file_name)
    candidates = candidates[candidates[candidates_file.predict_column]>=threshold]
    candidates = candidates[['oid', 'flare_start', candidates_file.predict_column]]
    candidates.columns = ['oid', 'flare_start', 'predict_prob']


    data = pd.read_csv(
        release_file,
        chunksize=10_000
    )

    result = pd.DataFrame(
        {
            'oid': pd.Series(dtype='int'),
            'flare_start': pd.Series(dtype='float'),
            'predict_prob': pd.Series(dtype='float'),
            'time_series': pd.Series(dtype='object'),
            'ra': pd.Series(dtype='float'),
            'dec': pd.Series(dtype='float')
        }
    )

    for chunk in tqdm(data, total=9729, desc='Data progress'):
        
        oids =  pd.Series(chunk.iloc[:,5]).reset_index(drop=True)
        data_list = pd.Series(map(convert_to_list, chunk.iloc[:,6]))
        flare_start = pd.Series(map(lambda x: x[1], data_list)).reset_index(drop=True)
        data_list = data_list.reset_index(drop=True)
        dec = pd.Series(chunk.iloc[:,7]).reset_index(drop=True)
        ra  = pd.Series(chunk.iloc[:,8]).reset_index(drop=True)

        chunk_df = pd.DataFrame(data={
            'oid':oids,
            'flare_start':flare_start,
            'time_series':data_list,
            'dec':dec,
            'ra':ra,
        })
        
        merge_df = pd.merge(
            candidates, chunk_df,
            on=['oid', 'flare_start'],
            how='inner'
        )
        result = pd.concat([result, merge_df])
            
        if result.shape[0] == candidates.shape[0] :
            break

    result = result.reset_index(drop=True)
    print(f'result dataframe={result.shape}')
    return result


def asteroid_search(ra, dec, time_max):
    """Search asteroid in radius at time maximum of the light curve"""
    
    RADIUS_SEARCH_ASTEROID = 0.0041667

    field = SkyCoord(ra*u.deg, dec*u.deg)
    earthed_time = hmjd_to_earth(time_max, field)
    try:
        results = Skybot.cone_search(
            field,
            RADIUS_SEARCH_ASTEROID*u.deg,
            earthed_time,
            location='I41'
        )
        return len(results) > 0, results[0]['Name']
    except Exception as e:
        if len(e.args)>0 and e.args[0][0:22]=='No solar system object':
            return False, 'No object'
        else:
            print(f'Exception {e.args[0]}')     
            return False, 'Service error'

def get_flares_time(df):
    """Determine the time of the beginning and maximum of the light curve"""
    
    df['time_maximum'] = 0.0
    df['flare_start'] = 0.0

    for i, obs in tqdm(df.iterrows(), total=df.shape[0]):
        time = obs['time_series']
        mag = obs['time_series']
        if isinstance(time, str):
            time = convert_to_list(time)[1::4]
        if isinstance(mag, str):
            mag = convert_to_list(mag)[2::4]
            
        time_max = np.argmin([m for m,t in zip(mag,time)])
        df.loc[i,'flare_start'] = time[0]
        df.loc[i,'time_maximum'] = time[time_max]
        
    return df 

def search_asteroids(df:pd.DataFrame):
    """Search asteroids in dataframe"""
    
    df['asteroid'] = False
    df['asteroid_name'] = ''

    for i, obs in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            asteroid, asteroid_name  = asteroid_search(
                obs['ra'],
                obs['dec'],
                obs['time_maximum']
            )
        except Exception as e:
            if len(e.args)>0:
                print(f'Exception message {e.args[0]}')
            
            time.sleep(300)
            asteroid, asteroid_name  = asteroid_search(
                obs['ra'],
                obs['dec'],
                obs['time_maximum']
            )
        
        df.loc[i,'asteroid'] = asteroid
        df.loc[i,'asteroid_name'] = asteroid_name
        
    return df

def get_sharpness(df):
    """Get sharpness of source for all dataframe rows"""
    import io
    from astropy.io.votable import parse
    from urllib.request import urlopen

    URL_CALTECH = 'https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?ID='
    HALF_EXPOSION_TIME = 0.000174 # 15 sec

    df['sharpness_flare_start'] = np.NaN
    df['sharpness_time_maximum'] = np.NaN
    
    for i, obs in tqdm(df.iterrows(), total=df.shape[0]):
        
        field = SkyCoord(obs['ra']*u.deg, obs['dec']*u.deg)
        start_time_request = hmjd_to_earth(obs['flare_start'], field) - HALF_EXPOSION_TIME
        end_time_request = hmjd_to_earth(obs['time_maximum'], field) + HALF_EXPOSION_TIME
        
        url_request = URL_CALTECH + str(obs['oid']) + \
            '&TIME=' + str(start_time_request) + '%20' + str(end_time_request)
        
        try:    
            response = urlopen(url_request)
            stream = io.BytesIO(response.read())
            votable = parse(stream)
            table_data = votable.get_first_table().to_table(use_names_over_ids=True)
            
        except Exception as e:
            if len(e.args)>0:
                print(f'Exception message {e.args[0]}')
            
            time.sleep(300)

            response = urlopen(url_request)
            stream = io.BytesIO(response.read())
            votable = parse(stream)
            table_data = votable.get_first_table().to_table(use_names_over_ids=True)

        if len(table_data) > 0:
            df.loc[i,'sharpness_flare_start'] = table_data[0]['sharp']
            df.loc[i,'sharpness_time_maximum'] = table_data[-1]['sharp']
        
    return df        

if __name__ == "__main__":
    
    test_data = pd.read_csv('post filter/test_data.csv')
    test_data = get_flares_time(test_data)
    test_data = search_asteroids(test_data)
    test_data = get_sharpness(test_data)
    test_data.to_csv('postfiltered_test_data.csv', index=False)

