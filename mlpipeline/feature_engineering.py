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
from datetime import datetime

import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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
# LogManager.created_filename = os.path.join(conf.LOG_DIR, 'mlpipeline.log')
logger = LogManager.get_logger(__name__)

# global varirable
DEFAULT_MISSING_FLOAT = -1.234
DEFAULT_MISSING_STRING = 'U'
STD_THRESHOLD_FOR_REMOVING_COLUMNS = 1
FAULT_LABEL = 1
MAX_SAMPLING_DISKS = 10000
USING_LABEL = 'tag'
SELECTED_CONT_COLS = ['smart_1_normalized','smart_7_normalized',
                      'smart_197_normalized','smart_198_normalized',
                      'smart_1raw', 'smart_4raw', 'smart_5raw', 
                      'smart_7raw', 'smart_9raw', 'smart_12raw','smart_184raw', 'smart_187raw', 
                      'smart_188raw', 'smart_189raw', 'smart_190raw', 
                      'smart_192raw', 'smart_193raw', 'smart_194raw',
                      'smart_195raw', 'smart_197raw', 'smart_198raw', 
                      'smart_199raw','smart_240raw','smart_241raw',
                      'smart_242raw']
SELECTED_INDEX_COLS = ['dt','serial_number','model']
SELECTED_CATE_COLS = []
SELECTED_LABEL_COLS = ['tag','flag']

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

def _create_daily_features(df, 
                            index_cols,
                            cont_cols,
                            cate_cols,
                            window_list=[3],  # [1,2,3]
                            window_size=10,
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
    assert len(cont_cols)>0 and len(index_cols)>0, \
    'cont_cols, index_cols and label_cols cannot be empty'
    
    init_date = df['dt'].values
    back_fill_columns = index_cols + cont_cols
    df = _back_fill(df, back_fill_columns=back_fill_columns, freq='D')
    df_index = df[index_cols + ['dt']]
    if len(cate_cols):
        df_cate = df[cate_cols]
        tmp_dfs = [df_index, df_cate]
    else:
        tmp_dfs = [df_index]   
    df_sliding_cols = df[cont_cols]
    cont_dfs = []
    
#     mean_data = df_sliding_cols.rolling(window_list[0]*window_size//2, min_periods=min_periods).mean()
#     cont_dfs.append(mean_data)
    for i_window in window_list:
        target_df = df_sliding_cols
        cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).min()
                    ).rename(
            columns=dict(zip(cont_cols, [col + "_min_%s" % (i_window * window_size) for col in cont_cols]))))
        cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).max()
                    ).rename(
            columns=dict(zip(cont_cols, [col + "_max_%s" % (i_window * window_size) for col in cont_cols]))))
#         cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).std()
#                     ).rename(
#             columns=dict(zip(cont_cols, [col + "_std_%s" % (i_window * window_size) for col in cont_cols]))))
#         cont_dfs.append((target_df.rolling(i_window * window_size, min_periods=min_periods).mean()
#                     ).rename(
#             columns=dict(zip(cont_cols, [col + "_mean_%s" % (i_window * window_size) for col in cont_cols]))))

#     The diff value between last period   
    for i_window in window_list:
            for col in cont_cols:
                cont_dfs.append((df_sliding_cols[[col]] -                                                                                                        df_sliding_cols[[col]].shift(i_window *                                                                                        window_size)).rename(columns=dict({col: \
                                 '%s_diff_for_last_period_%s' % (col, i_window * window_size)})))                                      
    # the operating duration of the disk, dt has been sorted 
#     cont_dfs.append(pd.DataFrame((df_index['dt'] - df_index['dt'].iloc[0]).apply(lambda                                             x:x.days)).astype(np.int8).rename(columns=dict({'dt':'operation_duration'})))                                               
    cont_dfs.append(pd.DataFrame(df_sliding_cols['smart_9raw']//24).rename(columns=dict({'smart_9raw':'smart_9raw_in_day_unit'})))
    df_sliding_cols.drop(columns=['smart_9raw'], inplace=True)
    
    cont_dfs.append(df_sliding_cols)
    cont_df = pd.concat(cont_dfs, axis=1)
    
    # fill all cont features with its own mean 
    cont_cols = cont_df.columns
    values = dict([(col_name, col_mean) for col_name, col_mean in zip(cont_cols,                   
                                                                      cont_df[cont_cols].mean().tolist())])
    cont_df.fillna(value=values, inplace=True)
    df = pd.concat([cont_df] + tmp_dfs, axis=1)
    init_date_df = df[df.dt.isin(init_date)]  # we do not use the data generated by back_fill
#     del cont_df, cont_dfs
#     gc.collect()
    return init_date_df

@timer(logger)
def _sampling(train_df, 
              valid_start_date,
              valid_end_date,
              sample_validset,
              group_cols):
    # do not do sampling for valid set
    valid_df = pd.DataFrame()
    if not sample_validset:
        logger.info('不对验证集采样')
        mask = train_df.dt >=valid_start_date
        mask &= train_df.dt <=valid_end_date
        valid_df = train_df[mask]
        non_valid_df = train_df[~mask]
        train_df = non_valid_df
        
    mask = train_df[USING_LABEL] == FAULT_LABEL
    train_fault_df = train_df[mask]

    train_sub_dfs = dict(tuple(train_df.groupby(group_cols)))
    train_fault_sub_dfs = dict(tuple(train_fault_df.groupby(group_cols)))

    # sample the normal disk by the num of fault disk
    np.random.seed(1234)
    if len(train_sub_dfs) > MAX_SAMPLING_DISKS and len(train_sub_dfs) > len(train_fault_sub_dfs):
            sample_rate = MAX_SAMPLING_DISKS * 1.0 / len(train_sub_dfs)
            train_sample_sub_dfs = dict([(x, train_sub_dfs[x]) \
                                            for x in tqdm(train_sub_dfs) \
                                            if np.random.random() < sample_rate \
                                            or x in train_fault_sub_dfs])  # here, we keep the records of all fault disk
            
            train_sample_df = pd.concat([train_sample_sub_dfs[key] for key in train_sample_sub_dfs])
            train_sample_df = pd.concat([train_sample_df, valid_df]) if not sample_validset else train_sample_df
            logger.info('采样前数据集正负样本数：%s : %s'%(
                                                    len(train_df[train_df[USING_LABEL]==FAULT_LABEL]), \
                                                    len(train_df[train_df[USING_LABEL]!=FAULT_LABEL])))
            logger.info('采样后数据集正负样本数：%s : %s'%(
                                                    len(train_sample_df[train_sample_df[USING_LABEL]==FAULT_LABEL]), \
                                                    len(train_sample_df[train_sample_df[USING_LABEL]!=FAULT_LABEL])))
            return train_sample_df
    else:  
            train_sample_df = pd.concat([train_df, valid_df]) if not sample_validset else train_df 
            logger.info('采样前数据集正负样本数：%s : %s'%(
                                                    len(train_df[train_df[USING_LABEL]==FAULT_LABEL]), \
                                                    len(train_df[train_df[USING_LABEL]!=FAULT_LABEL])))
            logger.info('采样后训练集正负样本数：%s : %s'%(
                                                    len(train_sample_df[train_sample_df[USING_LABEL]==FAULT_LABEL]), \
                                                    len(train_sample_df[train_sample_df[USING_LABEL]!=FAULT_LABEL])))
            return train_sample_df 

@timer(logger)
def _sliding_window(train_fe_df, 
                    test_fe_df,
                    group_cols,
                    cont_cols,
                    cate_cols,
                    num_processes):
    train_test_fe_df = pd.concat([train_fe_df, test_fe_df])
    sub_dfs = dict(tuple(train_test_fe_df.groupby(group_cols)))
    results = []
    back_fill_index_cols = ['model', 'serial_number']
    with multiprocessing.Pool(processes=num_processes) as p:
        with tqdm(total=len(sub_dfs)) as pbar:
            for result in (p.imap_unordered(_apply_df, [(sub_dfs[key], back_fill_index_cols, cont_cols, cate_cols) \
                                                     for key in sub_dfs.keys()])):
                results += [result]              
                pbar.update()
        
    train_test_fe_df = pd.concat(results)
    del results
    gc.collect()
    logger.info('构造滑窗特征后，当前维度:%s' % 
                                            train_test_fe_df.shape[1])
    return train_test_fe_df

@timer(logger)
def _load_dataset_by_filename(train_filename, test_filename):
    """
    
    """
    train_data_path = os.path.join(conf.DATA_DIR, train_filename)
    logger.info('加载训练数据集: %s' % train_data_path)
    start_time = time()
    disk_smart_train_df = pd.read_hdf(train_data_path)
    end_time = time()
    logger.info('加载训练数据集完成,共用时: %s' % get_time_diff(start_time, end_time))
    
    test_data_path = os.path.join(conf.DATA_DIR, test_filename)
    logger.info('加载测试数据集: %s' % test_data_path)
    start_time = time()
    disk_smart_test_df = pd.read_hdf(test_data_path)
    end_time = time()
    logger.info('加载测试数据集完成,共用时: %s' % get_time_diff(start_time, end_time))
    
    return disk_smart_train_df, disk_smart_test_df

@timer(logger)
def _fill_cont_cols_na_value_by_mean(train_test_fe_df, 
                                     cont_cols):
    values = dict(zip(cont_cols, train_test_fe_df[cont_cols].mean().tolist()))
    train_test_fe_df.fillna(value=values, inplace=True)
    return train_test_fe_df

@timer(logger)
def _fill_cont_cols_na_value_by_default_value(train_test_fe_df, 
                                     cont_cols):
#     values = dict([(col_name, col_mean) for col_name, col_mean in zip(cont_cols,                   
#                                                                       train_test_fe_df[cont_cols].mean().tolist())])
    values = dict(zip(cont_cols, [DEFAULT_MISSING_FLOAT for _ in range(len(cont_cols))]))
    train_test_fe_df.fillna(value=values, inplace=True)
    return train_test_fe_df

@timer(logger)
def _data_preprocess(train_start_date,
                     train_end_date,
                     disk_smart_train_df, 
                     disk_smart_test_df,
                     use_model_one):
    """
    
    """
    disk_smart_train_df = disk_smart_train_df[disk_smart_train_df.model==1] if use_model_one else disk_smart_train_df 
    disk_smart_train_df = disk_smart_train_df[disk_smart_train_df['dt'] >= train_start_date] if train_start_date is not None \
    else disk_smart_train_df
    disk_smart_train_df = disk_smart_train_df[disk_smart_train_df['dt'] <= train_end_date] if train_end_date is not None \
    else disk_smart_train_df
    
    # remove cols which are similar betweem normal and fault disks
    train_selected_cols = SELECTED_CATE_COLS + SELECTED_CONT_COLS + SELECTED_LABEL_COLS + SELECTED_INDEX_COLS
    test_selected_cols  = SELECTED_CATE_COLS + SELECTED_CONT_COLS + SELECTED_INDEX_COLS
    disk_smart_train_df = disk_smart_train_df[train_selected_cols]
    disk_smart_test_df = disk_smart_test_df[test_selected_cols]
    logger.info('train使用的cols: %s'%train_selected_cols)
    logger.info('test使用的cols: %s'%test_selected_cols)
    
    correct_column_type(disk_smart_train_df)
    correct_column_type(disk_smart_test_df)
    index_cols, cate_cols, cont_cols, label_cols = check_columns(disk_smart_train_df.dtypes.to_dict())
    
    disk_smart_train_df.drop_duplicates(index_cols, keep='first',inplace=True)
    disk_smart_test_df.drop_duplicates(index_cols, keep='first', inplace=True)
    
    cols_with_unique_number = remove_cont_cols_with_unique_value(disk_smart_train_df, 
                                                                 disk_smart_test_df, 
                                                                 cont_cols,
                                                                 threshold=4)
    disk_smart_train_df.drop(columns=cols_with_unique_number, inplace=True)
    disk_smart_test_df.drop(columns=cols_with_unique_number, inplace=True)
    
    drop_na_cols = check_nan_value(disk_smart_train_df,threshold=70)
    disk_smart_train_df.drop(columns=drop_na_cols, inplace=True)
    disk_smart_test_df.drop(columns=drop_na_cols, inplace=True)
    
    disk_smart_train_df.loc[disk_smart_train_df[USING_LABEL]!=0,USING_LABEL] = FAULT_LABEL
    
    
    train_df = disk_smart_train_df
    test_fe_df = disk_smart_test_df
    return train_df, test_fe_df


@timer(logger)
def feature_engineering(train_filename,
                        test_filename,
                        train_fe_save_filename='train_fe_df.h5',
                        test_fe_save_filename='test_fe_df.h5',
                        test_start_date='2018-08-01',
                        valid_start_date='2018-06-01',
                        valid_end_date='2018-06-31',
                        use_sampling= True,
                        sample_validset=False,
                        train_start_date=None, 
                        train_end_date=None, 
                        use_model_one=True,
                        num_processes = 10,):
    """
    
    :return:
    """
    # load dataset
#     print('reload successfully')
    disk_smart_train_df, disk_smart_test_df = _load_dataset_by_filename(train_filename,
                                                                        test_filename)
    
    # preprocess train data and test data
    train_df, test_fe_df =  _data_preprocess(train_start_date,
                                             train_end_date,
                                             disk_smart_train_df, 
                                             disk_smart_test_df,
                                             use_model_one)
    del disk_smart_train_df, disk_smart_test_df
    gc.collect()
    
    # sampling train data
    group_cols = ['model', 'serial_number']
    train_sample_df = pd.DataFrame()
    if use_sampling:
        train_sample_df = _sampling(train_df,
                                    valid_start_date,
                                    valid_end_date,
                                    sample_validset,
                                    group_cols)
    else:
        train_sample_df = train_df 
        
    del train_df
    gc.collect()
   
   # sliding window feature, can be used after all conts features generate 
    index_cols, cate_cols, cont_cols, label_cols = check_columns(train_sample_df.dtypes.to_dict())
    train_label_df = train_sample_df[index_cols + label_cols]  # for further joining with feature engineered data
    train_fe_df = train_sample_df[index_cols + cate_cols + cont_cols] 
    train_test_fe_df = _sliding_window(train_fe_df,
                                       test_fe_df,
                                       group_cols,
                                       cont_cols,
                                       cate_cols,
                                       num_processes = num_processes)
    del train_sample_df, test_fe_df
    gc.collect()
    
    # drop the col with too many nan
    drop_na_cols = check_nan_value(train_test_fe_df,threshold=70)
    train_test_fe_df.drop(columns=drop_na_cols, inplace=True)
    
    # TODO: rethink how to fill up the nan value
    _ ,_ , cont_cols, _ = check_columns(train_test_fe_df.dtypes.to_dict())
#     train_test_fe_df = _fill_cont_cols_na_value_by_mean(train_test_fe_df, cont_cols)
    train_test_fe_df = _fill_cont_cols_na_value_by_default_value(train_test_fe_df, cont_cols)
                                        
     # remove cols with small std
#     small_std_cols = remove_cont_cols_with_small_std(train_test_fe_df, 
#                                                      cont_cols, 
#                                                      STD_THRESHOLD_FOR_REMOVING_COLUMNS)
#     train_test_fe_df = train_test_fe_df.drop(columns=small_std_cols)
    
    # change serial_number as cate feat
#     train_test_fe_df['serial_number_feat'] = train_test_fe_df['serial_number'].apply(lambda x:x.split('_')      [1]).astype('category')
    
    # get the label cols back
    train_label_df.set_index(index_cols,inplace=True)
    train_test_fe_df.set_index(index_cols,inplace=True)
    train_test_fe_df = train_test_fe_df.join(train_label_df, how='left')
    train_test_fe_df.reset_index(drop=False, inplace=True)
#     train_test_fe_df.to_feather(os.path.join(conf.DATA_DIR, 'train_test_fe.feather'),key='train')
#     logger.info('train_test_fe_df has been dumped')
    
    # divide df into train and test  
    test_mask = train_test_fe_df.dt>=test_start_date 
    train_fe_df = train_test_fe_df[~test_mask]
    test_fe_df = train_test_fe_df[test_mask]
    del train_test_fe_df, train_label_df
    gc.collect()
    
    # save train and test features dataframe
    train_fe_df.reset_index(drop=True, inplace=True)
    test_fe_df.reset_index(drop=True, inplace=True)
    save_path_train = os.path.join(conf.DATA_DIR, train_fe_save_filename)
    save_path_test = os.path.join(conf.DATA_DIR, test_fe_save_filename)
    train_fe_df.to_feather(save_path_train)
    test_fe_df.to_feather(save_path_test)
    logger.info('训练文件已保存至%s'%save_path_train)
    logger.info('测试文件已保存至%s'%save_path_test)
    
    return train_fe_df, test_fe_df

if __name__ == "__main__":
    pass
    