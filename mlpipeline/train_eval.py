# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 28/09/2019

Description: 
   Train and eval the model
"""
import sys
import os
from time import time 
from datetime import timedelta, datetime

import lightgbm as lgb
from sklearn import metrics
import pandas as pd

sys.path.append('../')
from utils import (
    check_columns, 
    check_category_column, 
    encrypt_model, 
    transform_category_column,
    get_time_diff,
    decrypt_model,
)
import conf

# global default values
MIN_X = -1000000
MAX_X = 1000000
MIN_Y = -1
MAX_Y = 100
SMALL_VALUE = 0.01
DEFAULT_MISSING_FLOAT = -1.234
DEFAULT_MISSING_STRING = 'U'
CLASS_NAME = ['无故障','有故障']

def _precision(eval_df):
    ntpp = ''
    npp = len(eval_df[eval_df.prediction==1])
    return  

def _recall(eval_df):
#     npr = eval_df[eval_df.prediction==1]
#     ntpr = 
    pass 
    
def _f1_score(percision, recall):
     return 2* precision * recall / (precision + recall)

def train_pipeline_lightgbm(fe_df, 
                            split_date, 
                            params,
                            cls_threshold=0.4,
                            model_path=os.path.join(conf.DATA_DIR,'lgb')):
    
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
#     cate_transform_dict = check_category_column(fe_df, cate_cols)
#     fe_df = transform_category_column(fe_df, cate_transform_dict)

    if cate_cols and not cont_cols:
        train_x = pd.get_dummies(fe_df[fe_df['dt'] < split_date][cate_cols]).fillna(0.0)
        val_x = pd.get_dummies(fe_df[fe_df['dt'] >= split_date][cate_cols]).fillna(0.0)
    elif not cate_cols and cont_cols:
        fe_df.loc[:,cont_cols] = fe_df[cont_cols].fillna(fe_df[cont_cols].mean(), inplace=False)
        train_x = fe_df[fe_df['dt'] < split_date][cont_cols]
        val_x = fe_df[fe_df['dt'] >= split_date][cont_cols]    
    else:
        fe_df.loc[:,cont_cols] = fe_df[cont_cols].fillna(fe_df[cont_cols].mean(), inplace=False)
        train_x = pd.concat([pd.get_dummies(fe_df[fe_df['dt'] < split_date][cate_cols]).fillna(0.0),
                             fe_df[fe_df['dt'] < split_date][cont_cols]], axis=1)
        val_x = pd.concat([pd.get_dummies(fe_df[fe_df['dt'] >= split_date][cate_cols]).fillna(0.0),
                           fe_df[fe_df['dt'] >= split_date][cont_cols]], axis=1)
       
    train_y = fe_df[fe_df['dt'] < split_date][['tag']]
    
    val_x_index = fe_df[fe_df['dt'] >= split_date][index_cols]
    val_y = fe_df[fe_df['dt'] >= split_date][['tag']]
    
    # unify the features of dev set and train set 
    for col in list(train_x.columns) + list(val_x.columns):
        if col not in train_x.columns:
            train_x[col] = 0.0
        if col not in val_x.columns:
            val_x[col] = 0.0
    val_x = val_x[train_x.columns]
    
    train_set = lgb.Dataset(data=train_x, label=train_y['tag'])
    val_set = lgb.Dataset(data=val_x, label=val_y["tag"], reference=train_set)
    print('开始训练')
    start_time = time()
    pos_num = len(train_y[train_y.tag==1])
    neg_num = len(train_y[train_y.tag==0])
    
#     params = {"objective": "binary", "learning_rate": 0.01,'scale_pos_weight':100,
#                              'metric':'auc', 'subsample':0.7, 'max_bin':255, 'n_thread':3}
    model = lgb.train(params=params, 
                      train_set=train_set, 
                      valid_sets=[val_set],
                      num_boost_round=1000, 
                      early_stopping_rounds=10)
    print('训练结束')
    end_time = time()
    print('Time Usage:%s'%get_time_diff(start_time,end_time))
    
    # eval on dev set
    eval_df = pd.concat([val_y[['tag']], val_x_index], axis=1)
    eval_df.loc[:, 'prediction'] = model.predict(data=val_x) 
    eval_df.loc[:,'prediction'] = eval_df['prediction'].apply(lambda x:1 if x>=cls_threshold else 0)
    
    acc = metrics.accuracy_score(eval_df['tag'], eval_df['prediction'])
    report = metrics.classification_report(eval_df['tag'], eval_df['prediction'], target_names=CLASS_NAME, digits=4)
    confusion = metrics.confusion_matrix(eval_df['tag'], eval_df['prediction'])
    msg = 'Val Acc: {0:>6.2%}'
    print(msg.format(acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    
    # # get best iteration
    #  n_estimators = model.best_iteration
    #  print("n_estimators : ", n_estimators)
    #  print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    if model_path is not None:
        encrypt_model(model_path, (index_cols, cate_cols, cont_cols, label_cols, 
                                   train_x.columns, model))
#   return cate_transform_dict, models
    return model, eval_df

def pipeline_inference(fe_df, pred_threshold=0.4, model_path=os.path.join(conf.DATA_DIR,'lgb')):
    index_cols, cate_cols, cont_cols, label_cols, features, model = decrypt_model(model_path)
#     fe_df = transform_category_column(fe_df, cate_transform_dict)
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
    
    if cate_cols and not cont_cols:
        test_features = pd.get_dummies(fe_df[cate_cols]).fillna(0.0)
    elif not cate_cols and cont_cols:
        fe_df.loc[:,cont_cols] = fe_df[cont_cols].fillna(fe_df[cont_cols].mean(), inplace=False)
        test_features = fe_df[cont_cols]
    else:
        fe_df.loc[:,cont_cols] = fe_df[cont_cols].fillna(fe_df[cont_cols].mean(), inplace=False)
        test_features = pd.concat([pd.get_dummies(fe_df[cate_cols]).fillna(0.0), fe_df[cont_cols]],
                                  axis=1)
            
    for col in set(features) - set(test_features.columns):
        if col in cont_cols:
            test_features.loc[:, col] = DEFAULT_MISSING_FLOAT
        else:
            test_features.loc[:, col] = 0.0
    test_features = test_features[features]
    ret = fe_df[index_cols]
    ret.loc[:, 'prediction'] = model.predict(test_features) 
    ret.loc[:,'prediction'] = ret['prediction'].apply(lambda x:1 if x>=pred_threshold else 0)
    
    # save submission
    output_df = ret[ret.prediction==1][['manufacturer','model', 'serial_number','dt']]
    output_filename = 'submission_%s.csv'%datetime.now().isoformat()
    output_path = os.path.join(conf.SUBMISSION_DIR, output_filename)
    output_df.to_csv(output_path, index=False, header=False)
    print('%s已保存至%s'%(output_filename,output_path))
    
    return ret, output_df


# def offline_train(train_duration, train_params, correct_types, model_params, model, model_type,
#                   predict_targets=[1, 2, 7], time_step=5,
#                   train_date='2100-01-01', business_type="sample_data", logger=None):
#     logger.info("开始离线训练，业态：%s，日期: %s" % (business_type, train_date))
#     fe_df = pd.read_csv(os.path.join(conf.output_dir, "fe_df_%s.csv" % train_duration), dtype=correct_types)
#     fe_df = fe_df[fe_df['date'] <= train_date]
#     model_path = os.path.join(conf.output_dir, "%s.model.%s" % (model, train_date))

#     if model_type == 'neural':
#         train_pipeline_neural_network(fe_df, logger=logger, predict_targets=predict_targets,
#                                       model_path=model_path, model=model, model_params=model_params,
#                                       train_params=train_params, time_step=time_step)
#     elif model_type == 'ensemble':
#         train_pipeline_ensemble_tree(fe_df, model_params=model_params, train_params=train_params, model=model,
#                                      predict_targets=predict_targets, model_path=model_path, logger=logger)
#     elif model_type == 'stacking':
#         # TODO: 增加stacking部分train pipeline
#         raise NotImplementedError
#     else:
#         raise ValueError('%s has not been implemented' % model_type)
#     logger.info("%s模型训练完成!模型保存至:%s" % (model, model_path))
