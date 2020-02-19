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


def read_data_dir(dir_path):
    def __reformat_input_df(df):
        if 'date' in df.columns:
            df.loc[:, 'date'] = pd.to_datetime(df['date'])
        return df

    ret = []
    for file_name in ['building_info.csv', 'device_info.csv', 'meter_reading_data.csv', 'weather_data.csv']:
        file_path = os.path.join(dir_path, file_name)
        ret.append(__reformat_input_df(pd.read_csv(file_path)) if os.path.exists(file_path) else None)
    return ret


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
    index_cols, cate_cols, cont_cols, label_cols = [], [], [], []
    for col in col_dict:
        if col in ['building_id', 'device_id', 'date', 'building_name', 'device_name']:
            index_cols.append(col)
        elif col.startswith("y_"):
            label_cols.append(col)
        elif col_dict[col] == float or col_dict[col] == int:
            cont_cols.append(col)
        else:
            cate_cols.append(col)
    return index_cols, cate_cols, cont_cols, label_cols


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


def get_dataset(x, y):
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float()
    )


def get_dataloader(x: np.array, y: np.array, batch_size: int, shuffle: bool = True,
                   num_workers: int = 0):
    dataset = get_dataset(x, y)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def apply_df(args):
    df, index_cols, time_step = args
    return generate_time_step_data(df, index_cols, time_step)


def generate_time_step_data(sub_df, index_cols, time_step):
    """
    # generate time step features based on building and device combo
    :param sub_df:
    :return:
    """
    feats = sub_df[list(filter(lambda x: x not in set(index_cols), sub_df.columns))].reset_index(drop=True)
    dates = sub_df[['date']].reset_index(drop=True)
    time_step_feats, time_step_start_date = [], [],
    for i in range(max(1, len(feats) - time_step)):
        time_step_feats += [feats.iloc[i:i + time_step].values]
        time_step_start_date += [dates.iloc[i].values[0]]

    return pd.DataFrame({'feats': time_step_feats,
                         'date': time_step_start_date})
