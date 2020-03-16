# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 25/10/2019

Description:
    Use for predict test data
"""
import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import torch
import multiprocessing
import numpy as np
import xgboost as xgb

from time_series_pipeline.feature_engineering import feature_engineering_pandas
sys.path.append('../')
from utils.utils import decrypt_model, transform_category_column, read_data_dir, get_latest_model, apply_df
import conf

DEFAULT_MISSING_FLOAT = -1.234
MIN_X = -1000000
MAX_X = 1000000


def inference_pipeline_ensemble_tree(fe_df, model, predict_targets=[1, 2, 7], logger=None, model_path=None):
    index_cols, cate_cols, cont_cols, label_cols, cate_transform_dict, features, models = decrypt_model(model_path)
    fe_df = transform_category_column(fe_df, cate_transform_dict)
    if model == 'lgb':
        test_features = pd.concat(
            [fe_df[cate_cols], fe_df[cont_cols].fillna(DEFAULT_MISSING_FLOAT).clip(MIN_X, MAX_X)],
            axis=1)
    else:
        test_features = pd.concat([pd.get_dummies(fe_df[cate_cols]).fillna(0.0), fe_df[cont_cols]],
                                  axis=1).fillna(DEFAULT_MISSING_FLOAT)

    # fill up features
    for col in set(features) - set(test_features.columns):
        if col in cont_cols:
            test_features.loc[:, col] = DEFAULT_MISSING_FLOAT
        else:
            test_features.loc[:, col] = 0.0
    test_features = test_features[features]
    logger.info('输入特征维度为%s' % test_features.shape[1])

    ret = fe_df[index_cols]
    ret.loc[:, 'meter_reading_base'] = test_features['meter_reading']
    for i, target in enumerate(predict_targets):
        ret.loc[:, 'predict_%s' % target] = models[i].predict(xgb.DMatrix(test_features)) * test_features[
            'meter_reading'] * target if model == 'xgb' else models[i].predict(test_features) * test_features[
            'meter_reading'] * target
        if "y_%s" % target in fe_df.columns:
            ret.loc[:, 'label_%s' % target] = fe_df['y_%s' % target] * test_features['meter_reading']
    return ret


def inference_pipeline_neural_net(test_fe_df, time_step, predict_date, model,
                                  correct_types, train_duration, predict_targets=[1, 2, 7],
                                  logger=None,
                                  model_path=None):
    # TODO: 如何高效的为神经网络构建time step 测试集, 构建数据库，从数据库中做查询
    def ___reshape_tensor_based_on_model():
        if model == 'tcn':
            return torch.from_numpy(test_x).float().reshape(-1, len(features), time_step)
        return torch.from_numpy(test_x).float().reshape(-1, time_step, len(features))

    index_cols, cate_cols, cont_cols, label_cols, cate_transform_dict, features, models = decrypt_model(model_path)
    time_step_begin_date = datetime.strptime(predict_date, '%Y-%m-%d') - timedelta(time_step)
    history_fe_df = pd.read_csv(os.path.join(conf.output_dir, "fe_df_%s.csv" % train_duration),
                                dtype=correct_types, parse_dates=['date'])
    history_fe_df = history_fe_df[
        (history_fe_df.date > str(time_step_begin_date)) & (history_fe_df.date < predict_date)]
    test_fe_df = pd.concat([history_fe_df, test_fe_df], axis=0).reset_index(drop=True)
    test_features = pd.concat([pd.get_dummies(test_fe_df[cate_cols]).fillna(0.0), test_fe_df[cont_cols]],
                              axis=1).fillna(
        DEFAULT_MISSING_FLOAT).T.drop_duplicates().T  # drop duplicate columns with same name

    # fill up the features
    for col in set(features) - set(test_features.columns):
        if col in cont_cols:
            test_features.loc[:, col] = DEFAULT_MISSING_FLOAT
        else:
            test_features.loc[:, col] = 0.0
    test_features = test_features[features].clip(MIN_X, MAX_X)
    logger.info('输入特征维度为%s' % test_features.shape[1])

    # generate time step data
    logger.info('开始生成time step data')
    tmp_df = pd.concat([test_features, test_fe_df[index_cols]]
                       , axis=1)
    sub_dfs = dict(tuple(tmp_df.groupby(['building_id', 'device_id'])))
    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1))
    result = pool.map_async(apply_df, [(sub_dfs[key], index_cols, time_step) for key in sub_dfs.keys()])
    pool.close()
    time_step_df = pd.concat(list(result.get())).fillna(DEFAULT_MISSING_FLOAT)
    logger.info('总共%s条time_step数据，time step为%s' % (time_step_df.shape[0], time_step))

    # change numpy into tensor
    test_x = np.asarray(time_step_df['feats'].values.tolist())  # make sure each array in feats with same length
    test_x = ___reshape_tensor_based_on_model()

    # predict
    time_step_test_fe_df = test_fe_df[test_fe_df.date.isin(time_step_df['date'].values)]
    ret = time_step_test_fe_df[index_cols]
    ret.loc[:, 'meter_reading_base'] = time_step_test_fe_df['meter_reading']
    for i, target in enumerate(predict_targets):
        with torch.no_grad():
            prediction = models[i](test_x).numpy().reshape(-1)
        ret.loc[:, 'predict_%s' % target] = prediction * time_step_test_fe_df['meter_reading'] * target
        if "y_%s" % target in test_fe_df.columns:
            ret.loc[:, 'label_%s' % target] = time_step_test_fe_df['y_%s' % target] * time_step_test_fe_df[
                'meter_reading'] * target
    return ret


def predict(predict_date, model, model_type, data_dir, business_type, correct_types,
            train_duration, time_step=5, predict_targets=[1, 2, 7], logger=None):
    if logger is None:
        logger.info("开始预测，业态：%s，日期: %s," % (business_type, predict_date))
    building_info_df, device_info_df, meter_reading_df, weather_df = read_data_dir(data_dir)
    test_df = feature_engineering_pandas(meter_reading_df, building_info_df, device_info_df, weather_df,
                                         end_date=predict_date, is_train=False, logger=logger)
    test_df.to_csv(os.path.join(conf.output_dir, "test_df.csv"), index=False)
    logger.info("预测样本数:%s" % test_df.shape[0])
    model_path = get_latest_model(conf.output_dir, '%s.model' % model)
    if model_type == 'ensemble':
        ret = inference_pipeline_ensemble_tree(test_df, model=model, logger=logger, predict_targets=predict_targets,
                                               model_path=model_path)
    elif model_type == 'neural':
        ret = inference_pipeline_neural_net(test_df,
                                            predict_targets=predict_targets,
                                            model_path=model_path,
                                            time_step=time_step, predict_date=predict_date,
                                            correct_types=correct_types,
                                            train_duration=train_duration,
                                            model=model,
                                            logger=logger)
    elif model_type == 'stacking':
        # TODO:增加stacking部分inference pipeline
        raise NotImplementedError
    else:
        raise ValueError('%s has not been implemented' % model_type)
    output_path = os.path.join(conf.output_dir, "%s_predict_result_%s" % (model, predict_date))
    ret.to_csv(output_path, index=False)
    logger.info("%s预测完成！结果保存至:%s" % (model, output_path))
