"""
@File	: data_preparing.py
@Author	: Chen Jingzhi
@Time	: 2020-03-13 14:57:27
@Email	: jingzhichen@yahoo.com
"""
import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
sys.path.append('../')
from utils import (
    check_columns, 
    plot_dist_of_cols,
    check_nan_value,
    correct_column_type,
    remove_cont_cols_with_small_std,
    remove_cont_cols_with_unique_value,
    get_time_diff,
    LogManager,
    timer
)
import conf

LogManager.created_filename = os.path.join(conf.LOG_DIR, 'data_prerocess.log')
logger = LogManager.get_logger(__name__)

def get_fault_info(filenames=['disk_sample_fault_tag.csv', 'disk_sample_fault_tag_201808.csv']):
    """
    Generate a dictionary to store fault date and fault tag of each disk.
    Use tuples (manufacturer, model, serial_number) as keys.
    """
    fault_df1 = pd.read_csv(os.path.join(conf.DATA_DIR, filenames[0]))
    fault_df2 = pd.read_csv(os.path.join(conf.DATA_DIR, filenames[1]))
    fault_df2.drop(['key'], axis=1,inplace=True)
    fault_tag_df = pd.concat([fault_df1, fault_df2], ignore_index=True)
    fault_dic = {}
    
    for _, row in fault_tag_df.iterrows():
        f_time = row["fault_time"]
        tag = row["tag"]
        key = tuple([row["manufacturer"], row["model"], row["serial_number"]])
        if key not in fault_dic.keys():
            sub_dic = {}
            sub_dic["date"] = f_time
            sub_dic["tag"] = tag
            fault_dic[key] = sub_dic
    return fault_dic

def tag_data(df):
    """
    Remove columns with missing ration larger than threshold.
    Add 3 types of tag for input dataframe and change dtype.
    Correct the data type.
    """
    fault_dic = get_fault_info()
    #nan_cols = check_nan_value(df)
    #df.drop(columns=nan_cols, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["tag"] = 0
    df["flag"] = 0
    df["30_day"] = 0
    for row in df.itertuples():
        number = getattr(row, "serial_number")
        manufacturer = getattr(row, "manufacturer")
        model = getattr(row, "model")
        key = tuple([manufacturer, model, number]) #get keys to look up in fault_dic
        if key in fault_dic:
            fault_date = fault_dic[key]["date"]
            date = str(getattr(row, "dt"))
            days_diff = (pd.to_datetime(fault_date) - pd.to_datetime(date)).days
            if days_diff >= 0 and days_diff <= 30:
                df.loc[row[0], "tag"] = fault_dic[key]["tag"] + 1
            if days_diff == 0:
                df.loc[row[0], "flag"] = 1
            if days_diff == 30:
                df.loc[row[0], "30_day"] = 1
    correct_column_type(df)   
    return df 

@timer(logger)
def save_data_to_hdf(time_period, save_filename):
    """
    Preprocess all csv data during given time period into one hdf5 file.
    time_period: list of time in yyyymm form.
    save_filename: name of target .h5 file.
    """
    hdf_file = pd.HDFStore(os.path.join(conf.DATA_DIR,save_filename),'w')
    try:
        for i in time_period:
            input_file = "disk_sample_smart_log_%s.csv" % i
            df = pd.DataFrame()
            for sub_df in pd.read_csv(os.path.join(conf.DATA_DIR, input_file), chunksize=1e+5, index_col=0):
                df = pd.concat([df, sub_df])  
            logger.info('%s 的数据读入完成，开始准备标记' % i )
            df = tag_data(df)
            logger.info('%s 的数据标记完成，存入h5文件' % i )
            hdf_file.append(key='data', value=df,format='table', data_columns=True)
            del df
            logger.info('%s 的数据处理完成' % i )
    finally:
        hdf_file.close()

def convert_test_to_hdf(test_file, save_filename):
    """
    Remove nan columns in test data and save as hdf file.
    """
    hdf_file = pd.HDFStore(os.path.join(conf.DATA_DIR,save_filename),'w')
    try:
        df = pd.read_csv(os.path.join(conf.DATA_DIR, test_file))
        nan_cols = check_nan_value(df)
        df.drop(columns=nan_cols, inplace=True)
        correct_column_type(df)
        hdf_file.append(key='data', value=df,format='table', data_columns=True)
        del df
    finally:
        hdf_file.close()

def extract_2017_fault_data():
    fe_2017_df = pd.read_hdf(os.path.join(conf.DATA_DIR,'data_2017_new_all.h5'))
    fe_2017_df.loc[fe_2017_df.tag!=0,'tag'] = 1
    temp_fault_df = fe_2017_df[fe_2017_df.flag==1]
    mask = fe_2017_df.serial_number.isin(temp_fault_df.serial_number)
    mask &= fe_2017_df.model.isin(temp_fault_df.model)
    ret_df = fe_2017_df[mask]
    ret_df.to_hdf(os.path.join(conf.DATA_DIR,'2017_fault_data.h5'),key='train',format='table')
    
if __name__ == "__main__":
    
    period_2017 = ["201707", "201708", "201709", "201710", "201711", "201712"]
    period_2018 = ["201801", "201802", "201803", "201804", "201805", "201806", "201807", "201808"]

    save_data_to_hdf(period_2017, "data_2017_all.h5")
    save_data_to_hdf(period_2018, "data_2018_all.h5")
    extract_2017_fault_data()

    #convert_test_to_hdf("disk_sample_smart_log_test_a.csv", "data_201808_test_all.h5")
    convert_test_to_hdf("disk_sample_smart_log_test_b.csv", "data_201808_test_b.h5")
