# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 17/12/2019

Description:
    Utility function
"""
import os
from functools import wraps
from time import time

import joblib
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

CODE_VERSION = "v1.0 alpha released at 2020.01"


def encrypt_model(path, model):
    def __encrypt(text):
        cryptor = AES.new(b"qz_secret_$#df21", AES.MODE_CBC,
                          b"qz_secret_$#df21")
        length = 16
        count = len(text)
        add = length - (count % length)
        text = text + (b'\0' * add)
        ciphertext = cryptor.encrypt(text)
        return b2a_hex(ciphertext)

    joblib.dump(model, path)
    text = open(path, "rb").read()
    with open(path, "wb") as f:
        f.write(__encrypt(text))


def decrypt_model(path):
    def __decrypt(text):
        cryptor = AES.new(b"qz_secret_$#df21", AES.MODE_CBC,
                          b"qz_secret_$#df21")
        plain_text = cryptor.decrypt(a2b_hex(text))
        return plain_text.rstrip(b'\0')

    text = open(path, "rb").read()
    with open(path + "1", "wb") as f:
        f.write(__decrypt(text))
    model = joblib.load(path + "1")
    os.remove(path + "1")
    return model


def back_fill(sub_df, back_fill_columns=['device_id', 'building_id', 'city', 'weather_morning', 'weather_evening',
                                         'meter_reading', 'temp_morning', 'temp_evening'],
              freq='D',
              start_date=None, end_date=None):
    """
    fill the missing value of a specific date with its nearest neighbour date,
    for rolling window
    :param sub_df:
    :param back_fill_columns: list of strings - filled with the nearest date data
    :param freq: str - frequency for filling missing date
    :param start_date: str - user-defined start date
    :param end_date: str - user_defined end date
    :return: df : pandas data-frame
    """
    assert len(back_fill_columns) > 0, 'back_fill_columns cannot be empty'

    # generate date range between start_date and end_date
    back_fill_columns = [col for col in back_fill_columns if col in sub_df.columns]
    sub_df = sub_df.sort_values('date')
    sub_df['date'] = pd.to_datetime(sub_df['date'])
    sub_df = sub_df.set_index('date')
    start_date, end_date = sub_df.index[0] if start_date is None else start_date, \
                           sub_df.index[-1] if end_date is None else end_date
    date_range = pd.date_range(start_date, end_date, freq=freq)

    # back fill missing values
    sub_back_df = sub_df[back_fill_columns]
    sub_non_back_columns = list(set(sub_df.columns) - set(sub_back_df.columns))
    if sub_non_back_columns:
        sub_non_back_df = sub_df[sub_non_back_columns]
        sub_back_df = sub_back_df.reindex(date_range, method='pad')
        sub_non_back_df = sub_non_back_df.reindex(date_range, fill_value=0)
        df = pd.concat([sub_back_df, sub_non_back_df], axis=1).reset_index().rename(columns={'index': 'date'})
        return df
    else:
        return sub_back_df.reindex(date_range, method='pad').reset_index().rename(columns={'index': 'date'})


def timer(logger):
    """
    decorator for calculate the running time of a func
    :param logger: LogManager object - for logging
    :return:
    """

    def real_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info('{} starts'.format(func.__name__))
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logger.info('{} took: {:.2f}s'.format(func.__name__, (end - start)))
            logger.info('{} finishs\n'.format(func.__name__))
            return result

        return wrapper


def preprocess_weather_data(weather_data_input_path, weather_data_output_path, start_date, end_date, city):
    weather_df = pd.read_csv(weather_data_input_path)
    weather_df.drop(columns=['Unnamed: 0'], inplace=True)

    column_name_c = ['时间', '省份', '城市', '白天天气', '白天风力', '白天最高温', '夜间天气', '夜间风力', '夜间最低温']
    weather_df.rename(columns=dict(zip(weather_df.columns, column_name_c)), inplace=True)
    column_name_e = ['date', 'city', 'weather_morning', 'temp_morning', 'weather_evening', 'temp_evening']
    sub_weather_df = weather_df[['时间', '城市', '白天天气', '白天最高温', '夜间天气', '夜间最低温']]
    sub_weather_df.columns = column_name_e

    mask = sub_weather_df['date'] >= start_date
    mask &= sub_weather_df['date'] <= end_date
    mask &= sub_weather_df['city'] == city
    sub_weather_df = sub_weather_df[mask]

    sub_weather_df.to_csv(weather_data_output_path, index=False)


def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.

    .. note:: Should only be used to wrap a method where first arg is `self`
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("Method %s forces keyword arguments." % func.__name__)
        self._input_kwargs = kwargs
        return func(self, **kwargs)

    return wrapper


def check_columns(col_dict):
    """Check columns type"""
    index_cols, cate_cols, cont_cols, label_cols = [], [], [], []
    for col in col_dict:
        if col in ['model', 'dt', 'serial_number', 'manufacturer']:
            index_cols.append(col)
        elif col=="tag":
            label_cols.append(col)
        elif col_dict[col] == float or col_dict[col] == int or col_dict[col]=='float16' or col_dict[col]=='float32':
            cont_cols.append(col)
        else:
            cate_cols.append(col)
    return index_cols, cate_cols, cont_cols, label_cols



def transform_category_column(fe_df, cate_transform_dict):
    for cate in cate_transform_dict:
        cate_set = cate_transform_dict[cate]
        fe_df.loc[:, cate] = fe_df[cate].apply(lambda x: x if x in cate_set else 'other')
        fe_df[cate] = fe_df[cate].astype('category')
    return fe_df


def overrides(interface_class):
    """
    overrides decorate for readability
    :param interface_class:
    :return:
    """

    def overrider(method):
        assert method.__name__ in dir(interface_class), '%s is not in %s' % (method.__name__, interface_class.__name__)
        return method

    return overrider


def get_latest_model(dir_path, file_prefix=None):
    files = sorted(os.listdir(dir_path))
    if file_prefix is not None:
        files = [x for x in files if x.startswith(file_prefix)]
    return os.path.join(dir_path, files[-1])


def get_time_diff(start_time, end_time):
    """cal the time func consumes"""
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))

def correct_colum_type(fe_df):
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    
    for cont in cont_cols:
        fe_df[cont] = fe_df[cont].astype(np.float32)
        
    fe_df['model'] = fe_df['model'].astype(np.int8)
    
    if 'tag' in fe_df.columns:
        fe_df['tag'] = fe_df['tag'].astype(np.int8)
        
    fe_df['dt'] = pd.to_datetime(fe_df['dt'], format='%Y%m%d')
    print(fe_df.dtypes)

def check_category_column(fe_df, cate_cols, num_cates_threshold=5):
    cate_transform_dict = {}
    total_samples = fe_df.shape[0]
    ret_cate_cols = []
    for cate in cate_cols:
        if fe_df[[cate]].drop_duplicates().shape[0] >= num_cates_threshold:
            cate_stat = fe_df.groupby(cate, as_index=False)[['date']].count()
            cate_stat['date'] = cate_stat['date'].apply(lambda x: round(x / total_samples, 3))
            select_cates = set(cate_stat[cate_stat['date'] > 0.005][cate])  # 至少占比0.5%的类别特征才会被选择
            cate_transform_dict[cate] = select_cates
            ret_cate_cols += [cate]

    return cate_transform_dict, ret_cate_cols

def check_nan_value(fe_df, threshold=30):
    drop_na_cols = []
    for col in fe_df.columns:
        miss_ratio = round(fe_df[col].isnull().sum() / fe_df.shape[0]*100,2)
        print('%s - %s ' %(col, miss_ratio))
        if miss_ratio>=threshold:
            drop_na_cols += [col]
    return drop_na_cols