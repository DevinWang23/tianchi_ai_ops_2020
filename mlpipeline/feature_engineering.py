# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 23/10/2019

Description:
    Do the feature engineering

"""
import sys
import os 
from time import time
import gc
from datetime import datetime, timedelta
import traceback

import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

sys.path.append('../')
import conf
from utils import (
    check_columns, 
    plot_dist_of_cols,
    check_nan_value,
    correct_column_type,
    remove_cont_cols_with_small_std,
    remove_cont_cols_with_unique_value,
    get_time_diff,
    LogManager,
    timer,
)


# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'feature_engineering.log')
logger = LogManager.get_logger(__name__)

# global varirable
POWER_ON_HOURS_CATE_LIST = [-9999,  452.75 ,  905.5  , 1358.25 , 9999.]  # [-1.811,  452.75 ,  905.5  , 1358.25 , 1811.]
BINS_FOR_CUT_POWER_ON_HOURS_FEAT = 4
DEFAULT_MISSING_FLOAT = -1.234
DEFAULT_MISSING_STRING = 'U'
STD_THRESHOLD_FOR_REMOVING_COLUMNS = 1
FAULT_LABEL = 1
USING_LABEL = 'tag'
DROP_UNIQUE_COL_THRESHOLD = 1  # i.e. the unique number of this column should be larger than threshold 
DROP_NAN_COL_THRESHOLD = 30  # i.e. 30%
DIFF_WINDOW_SIZE = 7 
DIFF_WINDOW_COLS = ['smart_5raw',
                    'err_weight',
                    'seek_err_weight',
                    'degradation_err_weight']
# SELECTED_CONT_COLS = ['smart_1_normalized','smart_7_normalized',
#                       'smart_197_normalized','smart_198_normalized',
#                       'smart_1raw', 'smart_4raw', 'smart_5raw', 
#                       'smart_7raw', 'smart_9raw',                                                     'smart_12raw','smart_184raw', 'smart_187raw', 
#                       'smart_188raw', 'smart_189raw', 'smart_190raw', 
#                       'smart_192raw', 'smart_193raw', 'smart_194raw',
#                       'smart_195raw', 'smart_197raw', 'smart_198raw', 
#                       'smart_199raw','smart_240raw', 'smart_241raw',
#                       'smart_242raw','smart_240_normalized']
NORMALIZED_SMART_COLS = ['smart_1_normalized','smart_3_normalized',
                         'smart_7_normalized', 'smart_184_normalized',
                         'smart_9_normalized','smart_187_normalized',
#                        'smart_189_normalized','smart_193_normalized',
                         'smart_191_normalized', 
                         'smart_195_normalized']

RAW_SMART_COLS = [ 'smart_4raw','smart_5raw',
                      'smart_9raw','smart_12raw',
                      'smart_188raw',
                      'smart_192raw','smart_194raw',
                      'smart_197raw','smart_198raw',
                      'smart_199raw',
                     ]

SELECTED_CONT_COLS = NORMALIZED_SMART_COLS + RAW_SMART_COLS           
SELECTED_INDEX_COLS = ['dt','serial_number','model']
SELECTED_CATE_COLS = []
SELECTED_LABEL_COLS = ['tag','flag']

# dict values are the corrleation scores related to tag
ERR_RECORD_COLS = {
                   'smart_1_normalized':0.0016,
                   'smart_5raw':0.025,
                   'smart_184_normalized':0.004,
                   'smart_187_normalized':0.018,
                   'smart_195_normalized':0.004,
                   'smart_197raw':0.019,
                   'smart_198raw':0.019
}
SEEK_ERR_COLS = { 
                  'smart_7_normalized':0.0053,
#                   'smart_189_normalized':0.00099,
                  'smart_191_normalized':0.0058,
                  'smart_194raw':0.0011,
              }
DEGRADATION_ERR_COLS = {
                        'smart_3_normalized':0.0049,
                        'smart_9_normalized':0.006,
                        'smart_192raw':0.0068,
#                         'smart_193_normalized':0.00097
} 

# cols used to cal slope features
SLOPE_COLS = [
          'err_weight',
          'smart_5raw'
          ]  

NON_SLIDING_COLS = [
                    'smart_9_normalized',
                    'smart_197raw',
                    'smart_198raw',
                    'smart_187_normalized',
                    'smart_184_normalized',
                    'smart_191_normalized',
                    'smart_188raw',
                    'smart_199raw',
                   ]

POWER_ON_HOURS_COL = 'smart_9raw'
# NORMALIZED_POWER_ON_HOURS_COL = 'smart_9_normalized'
ERR_COMBINATION_COLS = [
                    'err_weight',
                    'seek_err_weight',
                    'degradation_err_weight']
def _apply_df(args):
    df, index_cols, cont_cols, cate_cols = args
    return _create_daily_features(df, 
                                   index_cols, 
                                   cont_cols, 
                                   cate_cols)
def _back_fill(sub_df, 
              back_fill_columns,
              freq='D',
              start_date=None, 
              end_date=None):
    """
    fill the missing value of a specific date with its nearest neighbour date,
    for rolling window
    :param sub_df:
    :param back_fill_columns: list of strings - filled with the nearest date data
    :param freq: str - frequency for filling missing date
    :param start_date: str - user-defined start date for sliding window
    :param end_date: str - user_defined end date for sliding window
    :return: df : pandas data-frame
    """
    back_fill_columns = [col for col in back_fill_columns if col in sub_df.columns]
    sub_df = sub_df.sort_values('dt')
    sub_df = sub_df.set_index('dt')
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
        df = pd.concat([sub_back_df, sub_non_back_df], axis=1).reset_index().rename(columns={'index': 'dt'})
        return df
    else:
        return sub_back_df.reindex(date_range, method='pad').reset_index().rename(columns={'index': 'dt'})
    
def _get_combination_weight(df,
                            weight_dict,
                            weight_col_name):
    
    total_weight = sum(weight_dict.values())
    df.loc[:,weight_col_name] = 0  # assign 0 as default value
    for key in weight_dict.keys():
        df[weight_col_name] += df[key]*(weight_dict[key]/total_weight)

def _linear_fit(x, slope, bias):
    return slope*x + bias

def _create_daily_features(df, 
                            index_cols,
                            cont_cols,
                            cate_cols,
                            window_list=[1],  # [1,2,3]
                            window_size=7,
                            min_periods=1,
                            ):
    """
    create min, max, mean and std for different sliding window sizes
    :param df:
    :param window_list:
    :param window_size:
    :param min_periods:
    :return:
    """
    assert len(cont_cols)>0 and len(index_cols)>0 and not df.empty, \
    'dataframe, cont_cols, index_cols and label_cols cannot be empty'
   
    init_date = df['dt'].values
    back_fill_columns = index_cols + cont_cols
    df = _back_fill(df, back_fill_columns=back_fill_columns, freq='D')
    df_index = df[index_cols + ['dt']]
    if len(cate_cols):
        df_cate = df[cate_cols]
        index_cate_dfs = [df_index, df_cate]
    else:
        index_cate_dfs = [df_index]   
        
    _get_combination_weight(df,
                             ERR_RECORD_COLS,
                             'err_weight')
    _get_combination_weight(df,
                             SEEK_ERR_COLS,
                            'seek_err_weight')
    _get_combination_weight(df,
                            DEGRADATION_ERR_COLS,
                            'degradation_err_weight')
    cont_cols += ERR_COMBINATION_COLS
    
    df_sliding_cols = df[cont_cols].drop(columns=[POWER_ON_HOURS_COL] + NON_SLIDING_COLS
                                                  , inplace=False)
    
    cont_dfs = []
#     mean_data = df_sliding_cols.rolling(window_list[0]*window_size, min_periods=min_periods).mean()
    for i_window in window_list:
        target_df = df_sliding_cols
        cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).min() 
                    ).rename(
            columns=dict(zip(cont_cols, [col + "_min_%s" % (i_window * window_size) for col in cont_cols]))))
        cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).max() 
                    ).rename(
            columns=dict(zip(cont_cols, [col + "_max_%s" % (i_window * window_size) for col in cont_cols]))))
        cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).std()
                    ).rename(
            columns=dict(zip(cont_cols, [col + "_std_%s" % (i_window * window_size) for col in cont_cols]))))
        cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).mean() 
                    ).rename(
            columns=dict(zip(cont_cols, [col + "_mean_%s" % (i_window * window_size) for col in cont_cols]))))

# Cal the diff value between last period and the tendency of window
#     for i_window in window_list:
    for col in DIFF_WINDOW_COLS:
                cont_dfs.append((df_sliding_cols[[col]].diff(periods=DIFF_WINDOW_SIZE))                            .rename(columns=dict({col:'%s_diff_for_last_period_%s' % (col, DIFF_WINDOW_SIZE)})))
    
    # Cal the slope feature
    for i_window in window_list:
            target_df = pd.concat([df_sliding_cols[SLOPE_COLS],df[[POWER_ON_HOURS_COL]]],axis=1)
            for col in target_df.columns:
                if col != POWER_ON_HOURS_COL:
                    tmp_df = target_df[[col,POWER_ON_HOURS_COL]].dropna(inplace=False).drop_duplicates(inplace=False)
                    tmp_index = tmp_df.index
                    tmp_df.set_index(POWER_ON_HOURS_COL, inplace=True)
                    cont_dfs.append((tmp_df[[col]].rolling(i_window * window_size,                                           min_periods=window_size).apply(lambda x:curve_fit(_linear_fit, x.index.values//24, x.values)[0][0],                  raw=False)).set_index(tmp_index.values,inplace=False).rename(columns=dict({col:'%s_slope_for_last_duration_%s'%(col, i_window * window_size)})))

#     the operating duration of the disk, dt has been sorted 
#     cont_dfs.append(pd.DataFrame((df_index['dt'] - df_index['dt'].iloc[0]).apply(lambda                                       x:x.days)).astype(np.int8).rename(columns=dict({'dt':'operation_duration'}))) 

    cont_dfs.append(pd.DataFrame(df[POWER_ON_HOURS_COL]//24). \
    rename(columns=dict({POWER_ON_HOURS_COL:'power_on_hours_in_day_unit'})))
    cont_dfs.append(df[NON_SLIDING_COLS])
    
    cont_dfs.append(df_sliding_cols)
    cont_df = pd.concat(cont_dfs, axis=1)
    
    # fill all cont features with its own mean 
    cont_df.fillna(method='pad', inplace=True)
    df = pd.concat([cont_df] + index_cate_dfs, axis=1)
    init_date_df = df[df.dt.isin(init_date)]  # we do not use the data generated by back_fill
    
    return init_date_df
 
@timer(logger)
def _sliding_window(fe_df, 
                    group_cols,
                    cont_cols,
                    cate_cols,
                    num_processes):
    sub_dfs = dict(tuple(fe_df.groupby(group_cols)))
    results = []
    back_fill_index_cols = ['model', 'serial_number']
    with multiprocessing.Pool(processes=num_processes) as p:
        with tqdm(total=len(sub_dfs)) as pbar:
            for result in (p.imap_unordered(_apply_df, [(sub_dfs[key], back_fill_index_cols, cont_cols, cate_cols) \
                                                     for key in sub_dfs.keys()])):
                results += [result]              
                pbar.update()
        
    fe_df = pd.concat(results)
    del results
    gc.collect()
    logger.info('构造滑窗特征后，当前维度(包括index_cols):%s' % 
                                            fe_df.shape[1])
    return fe_df

@timer(logger)
def _get_pred_data(data_path):
    """
    load pred data from tcdata folder into dataframe
    """
    def __get_date_range():
        start_date = '2018-08-11'
        end_date = '2018-09-30'  # '2018-09-30'
        start_date = datetime.strptime(start_date,'%Y-%m-%d')
        while (1):
            str_start_date = datetime.strftime(start_date, '%Y-%m-%d')
            start_date = start_date + timedelta(days = 1)
            if str_start_date > end_date:
                break
            yield str_start_date
            
    df = pd.DataFrame()
    for date in __get_date_range():
        date = pd.to_datetime(date).strftime('%Y%m%d')
        tmp_df = pd.read_csv(os.path.join(data_path,'disk_sample_smart_log_%s_round2.csv'% (date)),
                         usecols=SELECTED_CONT_COLS + 
                                 SELECTED_INDEX_COLS +        
                                 SELECTED_CATE_COLS)
        df = pd.concat([df, tmp_df])
    return df
        
@timer(logger)
def _load_data_into_dataframe(filename, is_train):
    """
    
    """
    data_path = os.path.join(conf.DATA_DIR, filename) if is_train else conf.PRED_DATA_DIR
    logger.info('加载数据集: %s' % data_path)
    start_time = time()
    if not is_train:
        disk_smart_df = _get_pred_data(data_path)
    
    # the store format for pre-processed train data is h5
    else:
        disk_smart_df = pd.read_hdf(data_path, columns=SELECTED_CONT_COLS + 
                                                       SELECTED_INDEX_COLS +        
                                                       SELECTED_CATE_COLS + 
                                                       SELECTED_LABEL_COLS,
                                                       )
    logger.info('使用的cols: %s'%disk_smart_df.columns)
    end_time = time()
    logger.info('加载数据集完成,共用时: %s' % get_time_diff(start_time, end_time))
    return disk_smart_df

@timer(logger)
def _fill_cont_cols_na_value_by_mean(fe_df, 
                                     cont_cols):
    values = dict(zip(cont_cols, fe_df[cont_cols].mean().tolist()))
    fe_df.fillna(value=values, inplace=True)
    return fe_df

@timer(logger)
def _fill_cont_cols_na_value_by_default_value(train_test_fe_df, 
                                     cont_cols):
    values = dict(zip(cont_cols, [DEFAULT_MISSING_FLOAT for _ in range(len(cont_cols))]))
    train_test_fe_df.fillna(value=values, inplace=True)
    return train_test_fe_df

@timer(logger)
def _data_preprocess(clip_start_date,
                     clip_end_date,
                     disk_smart_df, 
                     use_model_id,
                     use_2017_fault_data,
                     is_train,
                    ):
    """
    
    """
    if use_model_id:
        disk_smart_df = disk_smart_df[disk_smart_df.model==use_model_id] 
    disk_smart_df = disk_smart_df[disk_smart_df['dt'] >= clip_start_date] if clip_start_date is not None \
    else disk_smart_df
    disk_smart_df = disk_smart_df[disk_smart_df['dt'] <= clip_end_date] if clip_end_date is not None \
    else disk_smart_df
    
    if use_2017_fault_data:
        fault_data_2017_path = os.path.join(conf.DATA_DIR, '2017_fault_data.h5')
        fault_2017_df = pd.read_hdf(fault_data_2017_path, columns=SELECTED_CONT_COLS + 
                                                       SELECTED_INDEX_COLS +        
                                                       SELECTED_CATE_COLS + 
                                                       SELECTED_LABEL_COLS,
                                                       )
        disk_smart_df = pd.concat([disk_smart_df,fault_2017_df],axis=0)
    
    # some task-specific clean rules
    correct_column_type(disk_smart_df)
    index_cols, cate_cols, cont_cols, label_cols = check_columns(disk_smart_df.dtypes.to_dict())
    disk_smart_df.drop_duplicates(index_cols, keep='first',inplace=True)
    mask = (disk_smart_df[POWER_ON_HOURS_COL]!=0)
    disk_smart_df = disk_smart_df[mask]
    disk_smart_df.dropna(subset=[POWER_ON_HOURS_COL], inplace=True)
    logger.info(disk_smart_df[POWER_ON_HOURS_COL].isnull().sum())
    
    # for normalized cols,  closer to 100 means better condition 
    for col in NORMALIZED_SMART_COLS:
        if col in cont_cols:
            disk_smart_df[col] = 100 - disk_smart_df[col]
            
    if is_train:
        cols_with_unique_number = remove_cont_cols_with_unique_value(disk_smart_df, 
                                                 cont_cols,
                                                 threshold=DROP_UNIQUE_COL_THRESHOLD)
        disk_smart_df.drop(columns=cols_with_unique_number, inplace=True)
        drop_na_cols = check_nan_value(disk_smart_df,threshold=DROP_NAN_COL_THRESHOLD)
        disk_smart_df.drop(columns=drop_na_cols, inplace=True) 
        disk_smart_df.loc[disk_smart_df[USING_LABEL]!=0,USING_LABEL] = FAULT_LABEL
    return disk_smart_df

@timer(logger)
def _merge_non_fe_df_and_fe_df(non_fe_df, fe_df, index_cols):
        non_fe_df.set_index(index_cols,inplace=True)
        fe_df.set_index(index_cols,inplace=True)
        fe_df = fe_df.join(non_fe_df, how='left')
        fe_df.reset_index(drop=False, inplace=True)
        return fe_df

@timer(logger)
def feature_engineering(filename='',
                fe_save_filename='train_fe.feather',
                is_train=True,
                use_2017_fault_data=False,
                clip_start_date=None, 
                clip_end_date=None, 
                pred_start_date='2018-09-01',
                pred_end_date ='2018-09-30',
                use_model_id=None,
                num_processes = 10):
    """
    
    :return:
    """
    logger.info('训练数据特征工程: %s，数据集截断起始日期：%s, 数据集截断结束日期：%s'%(is_train, 
                                                            clip_start_date, 
                                                            clip_end_date))
    
    # load dataset
    disk_smart_df = _load_data_into_dataframe(filename,
                                is_train)
    
    # preprocess data
    disk_smart_df = _data_preprocess(clip_start_date,
                           clip_end_date,
                           disk_smart_df, 
                           use_model_id,
                           use_2017_fault_data,
                           is_train)
   
    """ generate cont feats"""
    # sliding window feature, can be used after all conts features generated 
    group_cols = ['model', 'serial_number']
    index_cols, cate_cols, cont_cols, label_cols = check_columns(disk_smart_df.dtypes.to_dict())
    if is_train:
        non_fe_df = disk_smart_df[index_cols + label_cols]    # for further joining with feature engineered data 
    fe_df = disk_smart_df[index_cols + cate_cols + cont_cols]
    del disk_smart_df
    gc.collect()
    fe_df = _sliding_window(fe_df,
                            group_cols,
                            cont_cols,
                            cate_cols,
                            num_processes)
    
    """generate cate feats"""
    # TODO: change model id into cate feat, try discrete continuous value into bin
    fe_df['model_type'] = fe_df['model'].map({1:0,2:1}).astype('category')
    fe_df.loc[fe_df['smart_198raw']>0,'smart_198raw']=1
    fe_df['smart_198raw'] =  fe_df['smart_198raw'].astype('category')
    fe_df.loc[fe_df['smart_197raw']>0,'smart_197raw']=1
    fe_df['smart_197raw'] =  fe_df['smart_197raw'].astype('category')
    fe_df.loc[fe_df['smart_184_normalized']>0,'smart_184_normalized']=1
    fe_df['smart_184_normalized'] =  fe_df['smart_184_normalized'].astype('category')
    fe_df.loc[fe_df['smart_188raw']>0,'smart_188raw']=1
    fe_df['smart_188raw'] =  fe_df['smart_188raw'].astype('category')
    fe_df.loc[fe_df['smart_187_normalized']>0,'smart_187_normalized']=1
    fe_df['smart_187_normalized'] =  fe_df['smart_187_normalized'].astype('category')
    fe_df.loc[fe_df['smart_191_normalized']>0,'smart_191_normalized']=1
    fe_df['smart_191_normalized'] =  fe_df['smart_191_normalized'].astype('category')
    fe_df.loc[fe_df['smart_199raw']>0,'smart_199raw']=1
    fe_df['smart_199raw'] =  fe_df['smart_199raw'].astype('category')
#     fe_df.loc[fe_df['smart_5raw']>0,'smart_5raw']=1
#     fe_df['smart_5raw'] =  fe_df['smart_5raw'].astype('category')
#     fe_df.dropna(subset=['power_on_hours_in_day_unit'], inplace=True)
    if is_train:
        fe_df['power_on_hours_in_day_unit_cate'] = pd.cut(fe_df['power_on_hours_in_day_unit'],bins=BINS_FOR_CUT_POWER_ON_HOURS_FEAT, labels=False)
    else:
        fe_df['power_on_hours_in_day_unit_cate'] = pd.cut(fe_df['power_on_hours_in_day_unit'], POWER_ON_HOURS_CATE_LIST, labels=False)
    fe_df['power_on_hours_in_day_unit_cate'] = fe_df['power_on_hours_in_day_unit_cate'].astype('category')
    fe_df.drop(columns=['power_on_hours_in_day_unit'],inplace=True)
    
    # drop the col with too many nan
    if is_train:
        drop_na_cols = check_nan_value(fe_df,threshold=DROP_NAN_COL_THRESHOLD)
        fe_df.drop(columns=drop_na_cols, inplace=True)
    
    # fill up the nan value
#     _ ,_ , cont_cols, _ = check_columns(fe_df.dtypes.to_dict())
#     fe_df = _fill_cont_cols_na_value_by_default_value(fe_df, cont_cols)
    #      train_test_fe_df = _fill_cont_cols_na_value_by_mean(train_test_fe_df, cont_cols)
                                        
     # # remove cols with small std
#     small_std_cols = remove_cont_cols_with_small_std(fe_df, 
#                                                      cont_cols, 
#                                                      STD_THRESHOLD_FOR_REMOVING_COLUMNS)
#     fe_df = fe_df.drop(columns=small_std_cols)

    if is_train:
        fe_df = _merge_non_fe_df_and_fe_df(non_fe_df, fe_df, index_cols)     # get the label cols back
        del non_fe_df
        gc.collect()
        fe_df.reset_index(drop=True, inplace=True)
        save_path = os.path.join(conf.DATA_DIR, fe_save_filename)
        fe_df.to_feather(save_path)
        logger.info('特征工程文件文件已保存至%s'%save_path)
    else:
    # get the prediction duration for predict data
        mask = fe_df.dt >=pred_start_date
        mask &= fe_df.dt<=pred_end_date
        fe_df = fe_df[mask] 
        fe_df.reset_index(drop=True, inplace=True)
    return fe_df

if __name__ == "__main__":
    pass
    