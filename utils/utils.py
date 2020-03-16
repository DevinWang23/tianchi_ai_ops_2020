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
from datetime import timedelta, datetime
import sys

import joblib
# from Crypto.Cipher import AES
# from binascii import b2a_hex, a2b_hex
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .LogManager import LogManager 
sys.path.append('../')
import conf

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'mlpipeline.log')
logger = LogManager.get_logger(__name__)

# def encrypt_model(path, model):
#     def __encrypt(text):
#         cryptor = AES.new(b"qz_secret_$#df21", AES.MODE_CBC,
#                           b"qz_secret_$#df21")
#         length = 16
#         count = len(text)
#         add = length - (count % length)
#         text = text + (b'\0' * add)
#         ciphertext = cryptor.encrypt(text)
#         return b2a_hex(ciphertext)

#     joblib.dump(model, path)
#     text = open(path, "rb").read()
#     with open(path, "wb") as f:
#         f.write(__encrypt(text))
        
def save_model(path, model):
    joblib.dump(model, path)

# def decrypt_model(path):
#     def __decrypt(text):
#         cryptor = AES.new(b"qz_secret_$#df21", AES.MODE_CBC,
#                           b"qz_secret_$#df21")
#         plain_text = cryptor.decrypt(a2b_hex(text))
#         return plain_text.rstrip(b'\0')

#     text = open(path, "rb").read()
#     with open(path + "1", "wb") as f:
#         f.write(__decrypt(text))
#     model = joblib.load(path + "1")
#     os.remove(path + "1")
#     return model

def load_model(path):
    model = joblib.load(path)
    return model

def timer(logger):
    def real_timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            logger.info('%s开始'%func.__name__)
            start_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            logger.info('%s已完成，共用时%s'%(
                                     func.__name__, 
                                     get_time_diff(start_time, end_time)))
            return result
        return wrapper
    return real_timer

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
        if col in ['model', 'dt', 'serial_number']:
            index_cols.append(col)
        elif col in ["tag" ,"flag",'30_day']:
            label_cols.append(col) 
        # judge cont cols type by its type prefix    
        elif str(col_dict[col])[:5] == 'float' or str(col_dict[col])[:3] == 'int':
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

@timer(logger)
def correct_column_type(fe_df, use_float16=False):
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    def __reduce_cont_cols_mem_by_max_min_value():
        for col in cont_cols:
            c_min = fe_df[col].min()
            c_max = fe_df[col].max()
            if str(fe_df[col].dtypes)[:3] == "int":  # judge col_type by type prefix
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    fe_df[col] = fe_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    fe_df[col] = fe_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    fe_df[col] = fe_df[col].astype(np.int32)
                else:
                    fe_df[col] = fe_df[col].astype(np.int64)
            else:
                
                # space and accuracy trade-off
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    fe_df[col] = fe_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    fe_df[col] = fe_df[col].astype(np.float32)
                else:
                    fe_df[col] = fe_df[col].astype(np.float64)
                
    __reduce_cont_cols_mem_by_max_min_value()
    fe_df['model'] = fe_df['model'].astype(np.int8)
    if 'tag' in fe_df.columns:
        fe_df['tag'] = fe_df['tag'].astype(np.int8)
    if 'flag' in fe_df.columns:
        fe_df['flag'] = fe_df['flag'].astype(np.int8)
    if '30_day' in fe_df.columns:
        fe_df['30_day'] = fe_df['30_day'].astype(np.int8)
    fe_df['dt'] = pd.to_datetime(fe_df['dt'], format='%Y%m%d')
    fe_df.sort_values(by='dt',inplace=True)
    logger.info('col_types: %s'%fe_df.dtypes)

@timer(logger)
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

# @timer(logger)
def check_nan_value(fe_df, threshold=30):
    nan_cols = []
    for col in fe_df.columns:
        miss_ratio = round((fe_df[col].isnull().sum() / fe_df.shape[0])*100,2)       
        logger.info("%s - %s%%" % (col, miss_ratio))
        if miss_ratio>=threshold:
            nan_cols += [col]
    return nan_cols

@timer(logger)
def remove_cont_cols_with_small_std(fe_df, cont_cols, threshold=1):
    assert not fe_df.empty and len(cont_cols)>0, 'fe_df and cont_cols cannot be empty'
    small_std_cols = []
    for col in cont_cols:
        col_std = round(fe_df[col].std(),2)
        logger.info('%s - %s ' %(col, col_std))
        if col_std<=threshold:
            small_std_cols += [col]
    return small_std_cols

@timer(logger)
def remove_cont_cols_with_unique_value(train_fe_df, test_fe_df, cont_cols, threshold=3):
    assert not (train_fe_df.empty and test_fe_df.empty) and len(cont_cols)>0, 'fe_df and cont_cols cannot be empty'
    unique_cols = []
    for col in cont_cols:
        train_num_unique = len(train_fe_df[col].unique())
        test_num_unique = len(test_fe_df[col].unique())
        logger.info('train: %s - %s ' %(col, train_num_unique))
        logger.info('test: %s - %s ' %(col, test_num_unique))
        if train_num_unique<=threshold or test_num_unique<=threshold:
             unique_cols += [col]
    logger.info('drop cols: %s' % unique_cols)
    return unique_cols

@timer(logger)
def standard_scale(cont_cols,
                   train_fe_df,
                   valid_fe_df=pd.DataFrame()):
        scaler = StandardScaler().fit(train_fe_df[cont_cols])
        train_fe_df.loc[:,cont_cols] = scaler.transform(train_fe_df[cont_cols])
        if not valid_fe_df.empty:
            valid_fe_df.loc[:,cont_cols] = scaler.transform(valid_fe_df[cont_cols])
            return train_fe_df, valid_fe_df
        return train_fe_df, scaler

@timer(logger)  
def log_scale(cont_cols,
              train_fe_df,
              valid_fe_df=pd.DataFrame()):
        train_fe_df.loc[:,cont_cols] = train_fe_df[cont_cols].apply(np.log2).fillna(0)
#         train_fe_df.loc[:,cont_cols] = train_fe_df[cont_cols].apply(np.log2)
        
#         train_fe_df.loc[:,cont_cols] = train_fe_df[cont_cols].replace(0, np.nan)
        if not valid_fe_df.empty:
            valid_fe_df.loc[:,cont_cols] =  valid_fe_df[cont_cols].apply(np.log2).fillna(0)
#             valid_fe_df.loc[:,cont_cols] =  valid_fe_df[cont_cols].apply(np.log2)
#             valid_fe_df.loc[:,cont_cols] =  valid_fe_df[cont_cols].replace(0, np.nan)
        return train_fe_df, valid_fe_df
     