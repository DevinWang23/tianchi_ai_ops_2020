# -*- coding: utf-8 -*-
"""
Author: Devin Wang
Email: k.tracy.wang@gmail.com
Date: 29/10/2019

Description:
    Test feature engineering
"""
import sys
import unittest

def test_sliding_window():
    pass

def test_sampling():
    train_chunk_df = pd.read_hdf(os.path.join(conf.DATA_DIR, 'chunk_data_for_test.h5'),key='train' )
    test_chunk_df = pd.read_hdf(os.path.join(conf.DATA_DIR, 'chunk_data_for_test.h5'), key='test')
    
    np.random.seed(1234)
    max_normal_disk_samples = 100
    mask = train_chunk_df.tag == FAULT_LABEL
    train_fault_fe_df = train_chunk_df[mask]

    group_cols = ['manufacturer', 'model', 'serial_number']
    train_sub_dfs = dict(tuple(train_chunk_df.groupby(group_cols)))
    train_fault_sub_dfs = dict(tuple(train_fault_fe_df.groupby(group_cols)))

# sample the normal disk
    if len(train_sub_dfs) > max_normal_disk_samples:
        sample_rate =  max_normal_disk_samples * 1.0 / len(train_sub_dfs)
        train_sample_sub_dfs = dict([(x, train_sub_dfs[x]) \
                                            for x in train_sub_dfs \
                           if np.random.random() < sample_rate or x in train_fault_sub_dfs] \
                         )
    assert set(list(train_sub_dfs)).intersection(set(list(train_fault_sub_dfs))) == \
    set(list(train_sample_sub_dfs)).intersection(set(list(train_fault_sub_dfs))), 'wrong sampling '

def test_feature_engineering():
    train_chunk_df = pd.read_hdf(os.path.join(conf.DATA_DIR, 'chunk_data_for_test.h5'),key='train' )
    test_chunk_df = pd.read_hdf(os.path.join(conf.DATA_DIR, 'chunk_data_for_test.h5'), key='test')
    
    params = {'disk_smart_train_df': train_chunk_df,
          'disk_smart_test_df': test_chunk_df,
          'start_date': '2018-01-01', # include start date
          'end_date': '2018-01-31', # include end date
          'use_model_one': True,
    }
_,_ = feature_engineering(**params)


if __name__ == "__main__":
    pass
