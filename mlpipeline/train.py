# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 28/09/2019

Description: 
   Train the model
"""
import sys
import os

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
import numpy as np

from .feature_engineering import feature_engineering_pandas
from .models.ensemble_tree import Lightgbm, Xgboost
from .models.neural_network import LstmModel, LstmAttentionModel, TcnModel

sys.path.append('../')
from utils.utils import read_data_dir, check_columns, check_category_column, encrypt_model, transform_category_column, \
    apply_df
import conf

# global default values
MIN_X = -1000000
MAX_X = 1000000
MIN_Y = -1
MAX_Y = 100
SMALL_VALUE = 0.01
DEFAULT_MISSING_FLOAT = -1.234
DEFAULT_MISSING_STRING = 'U'


def generate_period_training_set(data_dir, train_duration, logger, predict_targets=[1, 2, 7],
                                 start_date='2000-01-01', end_date='2100-01-01', ):
    building_info_df, device_info_df, meter_reading_df, weather_df = read_data_dir(data_dir)
    meter_reading_df = meter_reading_df[
        (meter_reading_df['date'] >= start_date) & (meter_reading_df['date'] <= end_date)]
    fe_df = feature_engineering_pandas(meter_reading_df, building_info_df, device_info_df, weather_df, logger=logger,
                                       predict_targets=predict_targets, is_train=True)
    fe_df.to_csv(os.path.join(conf.output_dir, "fe_df_%s.csv" % train_duration), index=False)


def train_pipeline_ensemble_tree(fe_df, model_params, train_params, model, logger,
                                 predict_targets=[1, 2, 7], model_path=None):
    def __train(model_class):
        trained_models = []
        valild_fold = list(TimeSeriesSplit(n_splits=5).split(train_x.values))
        for target in predict_targets:
            model_ = model_class(train_x, fe_df['y_%s' % target].clip(MIN_Y, MAX_Y),
                                 valild_fold, model_params, train_params)
            best_params, num_boost_round = model_.bayesian_optimization()
            trained_model = model_.train(best_params, num_boost_round)
            model_.feature_imp_plot(trained_model, target)
            trained_models.append(trained_model)
        if model_path is not None:
            encrypt_model(model_path, (index_cols, cate_cols, cont_cols, label_cols, cate_transform_dict,
                                       train_x.columns, trained_models))

    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    cate_transform_dict, cate_cols = check_category_column(fe_df, cate_cols)
    fe_df = transform_category_column(fe_df, cate_transform_dict)
    fe_df = fe_df.sort_values('date')  # for TimeSeriesSplit
    logger.info('筛选后的类别特征%s' % cate_cols)
    logger.info('筛选后的数值特征%s' % cont_cols)
    logger.info('筛选后的index特征%s' % index_cols)
    logger.info('筛选后的标签%s' % label_cols)

    if model == 'xgb':
        train_x = pd.concat([pd.get_dummies(fe_df[cate_cols]).fillna(0.0), fe_df[cont_cols]],
                            axis=1).fillna(DEFAULT_MISSING_FLOAT).clip(MIN_X, MAX_X)
        logger.info('输入特征维度为%s' % train_x.shape[1])
        __train(Xgboost)
    elif model == 'lgb':
        train_x = pd.concat(
            [fe_df[cate_cols], fe_df[cont_cols].fillna(DEFAULT_MISSING_FLOAT).clip(MIN_X, MAX_X)],
            axis=1)
        logger.info('输入特征维度为%s' % train_x.shape[1])
        __train(Lightgbm)
    else:
        raise ValueError('%s has not been implemented' % model)


def train_pipeline_neural_network(fe_df, train_params, model_params, model, logger, time_step,
                                  predict_targets=[1, 2, 7], valid_split=0.8, model_path=None):
    def __train(model_class):
        trained_models = []
        _, seq_len, input_size = train_x.shape
        for target in predict_targets:
            model_ = model_class(train_x, train_y['y_%s' % target].values.reshape(-1, 1),
                                 (val_x, val_y['y_%s' % target].values.reshape(-1, 1)), train_params, model_params,
                                 logger, model=model)
            trained_model = model_.train()
            trained_models.append(trained_model)
        if model_path is not None:
            encrypt_model(model_path, (index_cols, cate_cols, cont_cols, label_cols, cate_transform_dict,
                                       train_features.columns, trained_models))

    # TODO: 对离散特征做有意义的embedding
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    cate_transform_dict, cate_cols = check_category_column(fe_df, cate_cols)
    fe_df = transform_category_column(fe_df, cate_transform_dict)
    logger.info('筛选后的类别特征%s' % cate_cols)
    logger.info('筛选后的数值特征%s' % cont_cols)
    logger.info('筛选后的index特征%s' % index_cols)
    logger.info('筛选后的标签%s' % label_cols)

    # generate time step data
    logger.info('开始生成time step data')
    train_features = pd.concat([pd.get_dummies(fe_df[cate_cols]).fillna(0.0), fe_df[cont_cols]],
                               axis=1).fillna(DEFAULT_MISSING_FLOAT).clip(MIN_X, MAX_X)
    logger.info('输入特征维度为%s' % train_features.shape[1])
    tmp_df = pd.concat([train_features, fe_df[index_cols]]
                       , axis=1)
    sub_dfs = dict(tuple(tmp_df.groupby(['building_id', 'device_id'])))
    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1))
    result = pool.map_async(apply_df, [(sub_dfs[key], index_cols, time_step) for key in sub_dfs.keys()])
    pool.close()
    time_step_df = pd.concat(list(result.get())).fillna(DEFAULT_MISSING_FLOAT)
    logger.info('总共%s条time_step数据，time step为%s' % (time_step_df.shape[0], time_step))

    # split data into train and validation
    logger.info('开始划分训练集与验证集')
    split_date = sorted(time_step_df['date'])[int(valid_split * time_step_df.shape[0])]
    logger.info('分隔日期为%s' % split_date)
    train_dates = time_step_df[time_step_df['date'] <= split_date]['date'].values
    train_x = np.asarray(time_step_df[time_step_df['date'].isin(train_dates)]['feats'].values.tolist())
    train_y = fe_df[fe_df['date'].isin(train_dates)][['y_%s' % x for x in predict_targets]].clip(MIN_Y, MAX_Y)
    valid_date = time_step_df[time_step_df['date'] > split_date]['date'].values
    val_x = np.asarray(time_step_df[time_step_df['date'].isin(valid_date)]['feats'].values.tolist())
    val_y = fe_df[fe_df['date'].isin(valid_date)][['y_%s' % x for x in predict_targets]].clip(MIN_Y, MAX_Y)

    logger.info('开始训练%s模型' % model)
    if model == 'lstm':
        __train(LstmModel)
    elif model == 'tcn':
        __train(TcnModel)
    elif model == 'lstm_attention':
        __train(LstmAttentionModel)
    else:
        raise ValueError('%s has not been implemented' % model)


def offline_train(train_duration, train_params, correct_types, model_params, model, model_type,
                  predict_targets=[1, 2, 7], time_step=5,
                  train_date='2100-01-01', business_type="sample_data", logger=None):
    logger.info("开始离线训练，业态：%s，日期: %s" % (business_type, train_date))
    fe_df = pd.read_csv(os.path.join(conf.output_dir, "fe_df_%s.csv" % train_duration), dtype=correct_types)
    fe_df = fe_df[fe_df['date'] <= train_date]
    model_path = os.path.join(conf.output_dir, "%s.model.%s" % (model, train_date))

    if model_type == 'neural':
        train_pipeline_neural_network(fe_df, logger=logger, predict_targets=predict_targets,
                                      model_path=model_path, model=model, model_params=model_params,
                                      train_params=train_params, time_step=time_step)
    elif model_type == 'ensemble':
        train_pipeline_ensemble_tree(fe_df, model_params=model_params, train_params=train_params, model=model,
                                     predict_targets=predict_targets, model_path=model_path, logger=logger)
    elif model_type == 'stacking':
        # TODO: 增加stacking部分train pipeline
        raise NotImplementedError
    else:
        raise ValueError('%s has not been implemented' % model_type)
    logger.info("%s模型训练完成!模型保存至:%s" % (model, model_path))
