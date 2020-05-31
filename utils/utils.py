# -*- coding: utf-8 -*-
"""
Author: Devin Wang
Email: k.tracy.wang@gmail.com
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
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.sparse import vstack, csr_matrix

from .LogManager import LogManager 
sys.path.append('../')
import conf

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'utils.log')
logger = LogManager.get_logger(__name__)
        
def save_model(path, model):
    joblib.dump(model, path)

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
    fe_columns = fe_df.columns
    if 'model' in fe_columns:
        fe_df['model'] = fe_df['model'].astype(np.int8)
    if 'tag' in fe_columns:
        fe_df['tag'] = fe_df['tag'].astype(np.int8)
    if 'flag' in fe_columns:
        fe_df['flag'] = fe_df['flag'].astype(np.int8)
    if '30_day' in fe_columns:
        fe_df['30_day'] = fe_df['30_day'].astype(np.int8)
    if 'dt' in fe_columns:
        fe_df['dt'] = pd.to_datetime(fe_df['dt'], format='%Y%m%d')
        fe_df.sort_values(by='dt',inplace=True)
    if 'fault_time' in fe_columns:
        fe_df['fault_time'] = pd.to_datetime(fe_df['fault_time'], format='%Y%m%d')
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

@timer(logger)
def check_nan_value(fe_df, threshold=30):
    nan_cols = []
    for col in fe_df.columns:
        miss_ratio = round((fe_df[col].isnull().sum() / fe_df.shape[0])*100,2)       
        logger.info("%s - %s%%" % (col, miss_ratio))
        if miss_ratio>=threshold:
            nan_cols += [col]
    logger.info('drop cols: %s' % nan_cols)
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
def remove_cont_cols_with_unique_value(fe_df, cont_cols, threshold=3):
    assert not fe_df.empty and len(cont_cols)>0, 'fe_df and cont_cols cannot be empty'
    unique_cols = []
    for col in cont_cols:
        num_unique = len(fe_df[col].unique())
        logger.info('%s - %s ' %(col, num_unique))
        if num_unique<=threshold:
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
        for cont_col in tqdm(cont_cols):
            train_fe_df.loc[:,cont_col] = np.log2(train_fe_df[cont_col] + 1)
        if not valid_fe_df.empty:
            for cont_col in tqdm(cont_cols):
                valid_fe_df.loc[:,cont_col] =  np.log2(valid_fe_df[cont_col] + 1)
        return train_fe_df, valid_fe_df

@timer(logger)
def form_sparse_onehot(y_pred, num_leaves):
#     output_list = []
#     for i in int_df:
#         row = []
#         for j in i:
#             tmp = np.zeros(num_leaves)
#             tmp[j]=1
#             row.extend(tmp)
#         output_list += [row]
#     return pd.DataFrame(output_df)
        transformed_matrix = np.zeros([len(y_pred), len(y_pred[0])*num_leaves],dtype=np.int8)
        for i in range(len(y_pred)):
            tmp = np.arange(len(y_pred[0])) * num_leaves - 1 + np.array(y_pred[i])
            transformed_matrix[i][tmp] += 1
        return csr_matrix(transformed_matirx)
