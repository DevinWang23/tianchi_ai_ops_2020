# -*- coding: utf-8 -*-
"""
Author:  Devin Wang
Email: k.tracy.wang@gmail.com
Date: 28/09/2019

Description: 
   Train and eval the model
"""
import sys
import os
from time import time 
from datetime import timedelta, datetime
import gc
from collections import defaultdict

import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tqdm import tqdm
from imblearn.over_sampling import SMOTE 

sys.path.append('../')
from utils import (
    check_columns, 
    check_category_column, 
    save_model, 
    load_model,
    transform_category_column,
    get_time_diff,
    load_model,
    LogManager,
    timer,
    standard_scale,
    log_scale,
    remove_cont_cols_with_small_std,
    correct_column_type,
    form_sparse_onehot,
    get_latest_model,
    correct_column_type,
)

import conf
from mlpipeline.feature_engineering import (
    USING_LABEL,
    FAULT_LABEL,
)

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'train.log')
# LogManager.log_handle = 'file'
logger = LogManager.get_logger(__name__)

# global varirable
CLASS_NAME = ['无故障','有故障']
EARLY_STOPPING_ROUNDS=5
CLS_RANKING = 0.996  # 0.996,0.994
NUM_SUBMISSION = 40
DEFAULT_MISSING_FLOAT = -1.234
DEFAULT_MISSING_CATE = 0 

def _f1_score( eval_df):
    try:
        def __precision():
                tmp_df = eval_df[eval_df['pred']==1]
                mask = tmp_df['pred']==tmp_df['tag']
                ntpp = len(tmp_df[mask])
                npp = len(tmp_df)
                return  ntpp / npp
        def __recall():
                fault_tmp_df = eval_df[eval_df['flag']==1] 
                npr = len(fault_tmp_df)
                pred_tmp_df = eval_df[eval_df['pred']==1]
                mask =  (pred_tmp_df['model'].isin(fault_tmp_df['model'])) & (pred_tmp_df['serial_number'].isin(fault_tmp_df['serial_number']))
                ntpr = len(pred_tmp_df[mask])
                return ntpr / npr
        precision, recall = __precision(), __recall()
        f1_score = 2* precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0, 0, 0
        
    return precision, recall, f1_score
        
def _focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    """
    Adapation of the Focal Loss for lightgbm to be used as evaluation loss
    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-                        y_true)*np.log(1-p) )
    return 'focal_loss', np.mean(loss), False

def _focal_loss_lgb(y_pred, dtrain, alpha, gamma):
        """
        Focal Loss for lightgbm
        Parameters:
        -----------
        y_pred: numpy.ndarray
            array with the predictions
        dtrain: lightgbm.Dataset
        alpha, gamma: float
            See original paper https://arxiv.org/pdf/1708.02002.pdf
        """
        a,g = alpha, gamma
        y_true = dtrain.label
        def __fl(x,t):
            p = 1/(1+np.exp(-x))
            return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p) + (1-t)*np.log(1-p) )
        partial_fl = lambda x: __fl(x, y_true)
        grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
        hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
        return grad, hess
        
def _log_best_round_of_model(model, 
                             evals_result,
                             valid_index,
                             metric):
        assert hasattr(model, 'best_iteration'), 'just can logger object that has best_iteration attribute'
        n_estimators = model.best_iteration
        logger.info('eval最优轮数: %s, eval最优%s: %s' %(
                                                       n_estimators, 
                                                       metric,            
                                                       evals_result[valid_index][metric][n_estimators - 1]))
        return n_estimators
    
@timer(logger)
def _get_index_for_cv(train_x,
                      train_y,
                      train_date_list,
                      val_date_list,
                      n_fold):
    """ 
    get data index for cross validation, index of input fe_df should have been reset.
    Parameters:
    -----------
    train and val date list should be like:
    train_list = [['2018-02-01','2018-03-31'], ['2018-03-01','2018-04-30']]
    val_list = [['2018-05-01','2018-05-31'], ['20180601','20180630']]
    return fold as a list of tuples (train index,val index).
    """ 
    assert len(train_date_list) == n_fold and len(val_date_list) == n_fold, 'length of input list must be same as number of folds'   
    folds = []
  
    for i in range(n_fold):        
        train_start_date = train_date_list[i][0]
        train_end_date = train_date_list[i][1]
        train_mask = train_x['dt'] >= train_start_date 
        train_mask &= train_x['dt'] <= train_end_date
        train_idx = train_x[train_mask].index
    
        val_start_date = val_date_list[i][0]
        val_end_date = val_date_list[i][1]
        val_mask = train_x['dt'] >= val_start_date 
        val_mask &= train_x['dt'] <= val_end_date   
        val_idx = train_x[val_mask].index
        
        yield train_x.iloc[train_idx], train_y.iloc[train_idx], train_x.iloc[val_idx], train_y.iloc[val_idx]

# def __learning_rate_010_decay_power_0995(current_iter):
#     base_learning_rate = 0.1
#     lr = base_learning_rate  * np.power(.995, current_iter)
#     return lr if lr > 1e-3 else 1e-3

@timer(logger)
def _sampling(
                   fe_df,
                   is_eval,
                   use_sampling_by_month_with_weight,
                   use_sampling_by_power_on_hours,
                   use_sampling_by_clustering_label,
                   valid_start_date,
                   valid_end_date,
                   train_start_date,
                   train_end_date,
                   train_sample_ratio,
                   valid_sample_ratio,
                   next_month_start_date,
                   next_month_end_date,
                   use_next_month_fault_data,
                   use_2017_fault_data,
                   random_state
                   ):
    
    sample_dfs = []
    if is_eval:
        valid_mask = (fe_df['dt']>=valid_start_date) & (fe_df['dt']<=valid_end_date)
        valid_fe_df = fe_df[valid_mask]
        if valid_sample_ratio:
            valid_sample_num = int(valid_sample_ratio * len(valid_fe_df)) if valid_sample_ratio <= 1 else valid_sample_ratio
            logger.info('需采样验证集负样本ratio或个数：%s,样本数:%s'%(valid_sample_ratio,
                                                             valid_sample_num))
            mask = valid_fe_df[USING_LABEL]==FAULT_LABEL
            sample_dfs += [valid_fe_df[mask]]
            sample_dfs += [valid_fe_df[~mask].sample(min(len(valid_fe_df[~mask]),valid_sample_num), 
                                                     random_state=random_state)]
        else:
            sample_dfs += [valid_fe_df]
        del valid_fe_df
        gc.collect()
        
    # for train data, just sample the data in 2018, data size in 2017 is small  
    train_mask = (fe_df['dt']>=train_start_date) & (fe_df['dt']<=train_end_date) & (fe_df['dt']>='2018-01-01')
    train_fe_df = fe_df[train_mask]
    fault_mask = train_fe_df[USING_LABEL] == FAULT_LABEL
    normal_disk_df =  train_fe_df[~fault_mask]
    fault_disk_df =  train_fe_df[fault_mask]
    normal_sample_num = int(train_sample_ratio*len(normal_disk_df)) if train_sample_ratio <= 1 else train_sample_ratio
    total_normal_num =len(normal_disk_df)
    logger.info('负样本数：%s, 需采样训练集负样本ratio或个数：%s,样本数:%s'%(total_normal_num,
                                                                    train_sample_ratio,
                                                                    normal_sample_num))
    sample_dfs += [fault_disk_df]
    del train_fe_df
    gc.collect()
    
    if use_2017_fault_data:
        fault_2017_df = fe_df[fe_df['dt']<'2018-01-01']
        if not fault_2017_df.empty:
                sample_dfs += [fault_2017_df[fault_2017_df[USING_LABEL]==FAULT_LABEL]]
        
    if use_next_month_fault_data:
        next_month_fe_df = fe_df[(fe_df['dt']>=next_month_start_date) & (fe_df['dt']<=next_month_end_date)]
        if not next_month_fe_df.empty:
            next_month_fault_disk_df = next_month_fe_df[next_month_fe_df['flag']==1]
            next_month_tag_df = next_month_fe_df[next_month_fe_df['tag']==1]
            mask = next_month_tag_df.model.isin(next_month_fault_disk_df.model)
            mask &= next_month_tag_df.serial_number.isin(next_month_fault_disk_df.serial_number)
            sample_dfs += [next_month_tag_df[mask]]
    del fe_df
    gc.collect()
    
    def __sampling_by_month_with_weight(
                                        normal_disk_df, 
                                        valid_start_date,
                                        sample_dfs,
                                        normal_sample_num
    ):
            normal_disk_df.loc[:,'year'] = normal_disk_df['dt'].dt.year.astype(np.int16)
            normal_disk_df.loc[:,'month'] = normal_disk_df['dt'].dt.month.astype(np.int8)
            normal_disk_sub_dfs = dict(tuple(normal_disk_df.groupby(['year','month'])))
         
            
            # cal the decay ratio by the gap between current date and valid or test date
            valid_year, valid_month = tuple(map(lambda x: int(x),valid_start_date.split('-')[:2]))
            total_month_gap = 0
            sample_weight_dict = defaultdict(float)
            for year,month in normal_disk_sub_dfs.keys():
                    month_gap = (valid_year - year) * 12 + (valid_month - month)
                    total_month_gap += month_gap
                    sample_weight_dict[(year,month)] = month_gap

            # sort by month gap and then swap the month gap to be weight score
            sorted_weight_list = sorted(sample_weight_dict.items(),key=lambda x:x[1],reverse=False)  
            for i in range(len(sorted_weight_list)):
                key = sorted_weight_list[i][0]
                value = sorted_weight_list[-i-1][1]
                sample_weight_dict[key] = value/total_month_gap 
            logger.info('权重词典:%s'%sample_weight_dict)

            # do sampling for normal disks of train data in 2018
            for year_and_month in tqdm(normal_disk_sub_dfs):
                tmp_df = normal_disk_sub_dfs[year_and_month]
                weight_ratio = sample_weight_dict[year_and_month]
                sample_dfs+=[tmp_df.sample(min(len(tmp_df),int(normal_sample_num*(1/len(normal_disk_sub_dfs)))),random_state=random_state)]
            del normal_disk_sub_dfs
            gc.collect()

            # do sampling for normal
            sample_df = pd.concat(sample_dfs, axis=0)
            if 'cluster_label' in sample_df.columns: 
                sample_df.drop(columns=['year','month','cluster_label'], inplace=True)
            else:
                sample_df.drop(columns=['year','month'], inplace=True)
            return sample_df
    
    def  __sampling_by_clustering(normal_disk_df,
                                  sample_dfs,
                                  normal_sample_num,
                                  total_normal_num):
        
        cluster_labels = normal_disk_df['cluster_label'].unique()
#         weight_ratio = 1/len(cluster_labels)
        for cluster in tqdm(cluster_labels):
            tmp_df =  normal_disk_df[normal_disk_df['cluster_label']==cluster]
            weight_ratio = len(tmp_df) / total_normal_num
            sample_dfs +=                                                                                                                              [tmp_df.sample(min(len(tmp_df),int(normal_sample_num*weight_ratio)),random_state=random_state)]
        
        sample_df = pd.concat(sample_dfs, axis=0)
        sample_df.drop(columns=['cluster_label'], inplace=True)
        return sample_df
    
    def  __sampling_by_power_on_hours(normal_disk_df,
                                      sample_dfs,
                                      normal_sample_num,
                                      total_normal_num):
        
        normal_disk_df.dropna(subset=['power_on_hours_in_day_unit_cate'],inplace=True)
        power_on_hours_labels = normal_disk_df['power_on_hours_in_day_unit_cate'].unique()
        weight_ratio = 1/len(power_on_hours_labels)

        for label in tqdm(power_on_hours_labels):
            tmp_df =  normal_disk_df[normal_disk_df['power_on_hours_in_day_unit_cate']==label]
#             weight_ratio = len(tmp_df) / total_normal_num
            sample_dfs +=                                                                                                                              [tmp_df.sample(min(len(tmp_df),int(normal_sample_num*weight_ratio)),random_state=random_state)]
        
        sample_df = pd.concat(sample_dfs, axis=0)
#         sample_df.drop(columns=['smart_9raw_in_day_unit_cate'], inplace=True)
        return sample_df
    
    if use_sampling_by_month_with_weight:
        sample_df =  __sampling_by_month_with_weight(normal_disk_df, valid_start_date, sample_dfs,normal_sample_num)
    elif  use_sampling_by_power_on_hours:
         sample_df =  __sampling_by_power_on_hours(normal_disk_df,sample_dfs,normal_sample_num, total_normal_num)
    elif use_sampling_by_clustering_label:
        sample_df =  __sampling_by_clustering(normal_disk_df,sample_dfs,normal_sample_num, total_normal_num)
    else:
        pass
    sample_df.sort_values('dt',inplace=True)
    sample_df.reset_index(drop=True,inplace=True)
    return sample_df

def _train_valid_split(
                       fe_df,
                       train_start_date='2018-01-01',
                       train_end_date='2018-05-31',
                       valid_start_date='2018-07-01',
                       valid_end_date='2018-07-31',  
                       train_on_model_id=None,
                       eval_on_model_id=None,
                       use_next_month_fault_data=False,
                       next_month_start_date='2018-06-01',
                       next_month_end_date='2018-06-30',
                       use_2017_fault_data=False,
                       use_up_sampling_by_smote=False,
                       use_cv=False,
                       use_log=False,
):
    
    
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
#     cate_transform_dict = check_category_column(fe_df, cate_cols)
#     fe_df = transform_category_column(fe_df, cate_transform_dict)
    logger.info('连续性特征数量: %s' % len(cont_cols))
    logger.info('离散性特征数量: %s' % len(cate_cols))
    
#     fe_df[cont_cols].fillna(DEFAULT_MISSING_FLOAT,inplace=True)
#     fe_df[cate_cols].fillna(0,inplace=True)    
    train_fe_df = fe_df[(fe_df['dt'] >= train_start_date) & (fe_df['dt']<=train_end_date)] 
    
    # add 2017 fault data into train
    if use_2017_fault_data:
        fault_2017_df = fe_df[fe_df['dt']<'2018-01-01']
        if not fault_2017_df.empty:
                train_fe_df = pd.concat([fault_2017_df[fault_2017_df[USING_LABEL]==FAULT_LABEL],train_fe_df], axis=0)
                
    if use_cv:
        if train_on_model_id !=None:
            train_fe_df = train_fe_df[train_fe_df.model==train_on_model_id]
        train_fe_df.reset_index(drop=True,inplace=True)
        del fe_df
        gc.collect()

        if cate_cols and not cont_cols:
            train_x = train_fe_df[index_cols + cate_cols]
        elif not cate_cols and cont_cols:
            if use_log:
                    logger.info("使用log: %s"%use_log)
                    train_fe_df, _ = log_scale(cont_cols, 
                                               train_fe_df, 
                                               )
            train_x = train_fe_df[index_cols + cont_cols]
        else:
            if use_log:
                    logger.info("使用log: %s"%use_log)
                    train_fe_df, _ = log_scale(cont_cols, 
                                               train_fe_df)

            train_x = pd.concat([train_fe_df[cate_cols],
                                 train_fe_df[cont_cols],
                                 train_fe_df[index_cols]],
                                 axis=1)

        train_y = train_fe_df[label_cols]
        if use_up_sampling_by_smote:
            train_x, train_y = _up_sampling_by_smote(train_x, 
                                                     train_y, 
                                                     cont_cols,
                                                     cate_cols)
        return (index_cols, cate_cols, cont_cols, label_cols), train_x, train_y
    else:
        # add next_month_fault_data into train
        if use_next_month_fault_data:
                next_month_fe_df = fe_df[(fe_df['dt']>=next_month_start_date) & (fe_df['dt']<=next_month_end_date)]
                if not next_month_fe_df.empty:
                    next_month_fault_disk_df = next_month_fe_df[next_month_fe_df['flag']==1]
                    next_month_tag_df = next_month_fe_df[next_month_fe_df['tag']==1]
                    mask = next_month_tag_df.model.isin(next_month_fault_disk_df.model)
                    mask &= next_month_tag_df.serial_number.isin(next_month_fault_disk_df.serial_number)
                    train_fe_df = pd.concat([next_month_tag_df[mask],train_fe_df], axis=0)

        if train_on_model_id !=None:
            train_fe_df = train_fe_df[train_fe_df.model==train_on_model_id]
        val_fe_df = fe_df[(fe_df['dt'] >= valid_start_date) & (fe_df['dt']<=valid_end_date)]
        if eval_on_model_id !=None:
            val_fe_df = val_fe_df[val_fe_df.model==eval_on_model_id]     
        train_fe_df.reset_index(drop=True,inplace=True)
        val_fe_df.reset_index(drop=True,inplace=True)  
        del fe_df
        gc.collect()

        if cate_cols and not cont_cols:
            train_x = train_fe_df[cate_cols]
            val_x = val_fe_df[cate_cols + index_cols]
        elif not cate_cols and cont_cols:
            if use_log:
                    logger.info("使用log: %s"%use_log)
                    train_fe_df, val_fe_df = log_scale(cont_cols, 
                                               train_fe_df, 
                                               val_fe_df)
            train_x = train_fe_df[cont_cols]
            val_x = val_fe_df[cont_cols]    
        else:
            if use_log:
                    logger.info("使用log: %s"%use_log)
                    train_fe_df, val_fe_df = log_scale(cont_cols, 
                                               train_fe_df, 
                                               val_fe_df)

            train_x = pd.concat([train_fe_df[cate_cols],
                                 train_fe_df[cont_cols]],
                                 axis=1)
            val_x = pd.concat([val_fe_df[cate_cols],
                         val_fe_df[cont_cols],
                               ], 
                               axis=1)

        train_y = train_fe_df[label_cols]
        val_y = val_fe_df[label_cols]

        if use_up_sampling_by_smote:
            train_x, train_y = _up_sampling_by_smote(train_x, 
                                                 train_y, 
                                                 cont_cols,
                                                 cate_cols)

        return  (index_cols, cate_cols, cont_cols, label_cols), train_fe_df, val_fe_df, train_x, train_y, val_x, val_y

def _eval(
          model,
          index_cols,
          cate_cols, 
          cont_cols,
          val_x_index,
          val_x,
          val_y,
          valid_start_date,
          valid_end_date,
          model_name=None):
    
    # eval on valid set 
    results = []
    eval_df = pd.concat([val_x_index,val_x, val_y], axis=1)
    
    eval_df = eval_df.sort_values('dt')
    valid_start_date, valid_end_date = eval_df.iloc[0]['dt'], eval_df.iloc[-1]['dt'] 
    valid_date_range = pd.date_range(valid_start_date, 
                                     valid_end_date, 
                                     freq='D')
    for valid_date in valid_date_range:
        logger.info('验证日期：%s'%valid_date)
        sub_eval_df = eval_df[eval_df.dt==valid_date]
        if model_name =='xgboost':
            sub_eval_df.loc[:,'prob'] = model.predict(data=xgb.DMatrix(sub_eval_df[val_x.columns]))
        elif model_name =='lgb':
            sub_eval_df.loc[:,'prob'] = model.predict(data=sub_eval_df[cate_cols + cont_cols])
        else:
             sub_eval_df.loc[:,'prob'] = model.predict_proba(data=sub_eval_df[cate_cols + cont_cols])
        sub_eval_df.loc[:,'rank'] = sub_eval_df['prob'].rank()
        sub_eval_df.loc[:,'pred'] = (sub_eval_df['rank']>=sub_eval_df.shape[0] * CLS_RANKING).astype(int)
        sub_eval_df = sub_eval_df.loc[sub_eval_df.pred == FAULT_LABEL]
        sub_eval_df = sub_eval_df.sort_values('prob', ascending=False)
        top_sub_eval_df = sub_eval_df.reset_index(drop=True, inplace=False).iloc[:NUM_SUBMISSION]
        logger.info('原始预测为fault disk的个数：%s'%len(sub_eval_df))
        results.append(top_sub_eval_df)
    pred_df = pd.concat(results)
    pred_df.drop_duplicates(['model','serial_number'], inplace=True)
    logger.info('最终预测个数:%s'%len(pred_df))
    eval_df = eval_df.merge(pred_df[index_cols + ['prob','rank','pred']],how='left',on=index_cols)
    eval_df.loc[:,'pred'] = eval_df['pred'].fillna(0)
    acc = metrics.accuracy_score(eval_df[USING_LABEL], eval_df['pred'])
    report = metrics.classification_report(eval_df[USING_LABEL], eval_df['pred'], target_names=CLASS_NAME, digits=4)
    confusion = metrics.confusion_matrix(eval_df[USING_LABEL], eval_df['pred'])
    msg = 'Val Acc: {0:>6.2%}'
    logger.info(msg.format(acc))
    logger.info("Precision, Recall and F1-Score...")
    logger.info(report)
    logger.info("Confusion Matrix...")
    logger.info(confusion)

    # eval on competition scores 
    precision, recall, f1_score = _f1_score(
                                            eval_df,
                                           )
    logger.info("竞赛recall: %s"% recall)
    logger.info("竞赛precision: %s"%precision)
    logger.info("竞赛f1-score: %s"%f1_score)
    
    return eval_df, f1_score, precision, recall

@timer(logger)
def _up_sampling_by_smote(
                          train_x,
                          train_y,
                          cont_cols,
                          cate_cols
):
    result_list = [train_x[cate_cols].fillna(DEFAULT_MISSING_CATE),train_x[cont_cols].fillna(DEFAULT_MISSING_FLOAT)]
    train_x = pd.concat(result_list,axis=1)
    sm = SMOTE(random_state=42)
    ret_x,ret_y = sm.fit_resample(train_x[cate_cols + cont_cols], train_y[[USING_LABEL]])
    return ret_x, ret_y
       

@timer(logger)
def train_pipeline_lgb(fe_df, 
                       model_params,
                       eval_on_model_id,
                       train_on_model_id,
                       train_start_date,
                       train_end_date,
                       is_eval,
                       valid_start_date,
                       valid_end_date,
                       use_log,
                       save_feat_important,
                       next_month_start_date,
                       next_month_end_date,
                       use_next_month_fault_data,
                       use_2017_fault_data,
                       use_random_search,
                       random_search_times,
                       search_params_space,
                       use_up_sampling_by_smote,
                       use_cv,
                       train_date_list,
                       val_date_list,
                       n_fold,
                       model_save_path=None,):
    
    def __random_param_generator(search_params_sapce):
#         scale_pos_weight = np.random.choice(search_params['scale_pos_weight'])
        tmp_dict = {}
        for key in search_params_space.keys():
            if isinstance(search_params_sapce[key],list):
                tmp_dict[key] = np.random.choice(search_params_sapce[key])
            elif isinstance(search_params_sapce[key], tuple):
                 tmp_dict[key] = np.random.uniform(*search_params_sapce[key])
            else:
                raise TypeError('搜索参数区间定义类型需为tuple或者list')
        return  tmp_dict
    
    def __feature_imp_plot(
                          model, 
                          features_name,
                          train_start_time,
                          save_feat_important,
                          max_num_features=30,
                          font_scale=0.7
    ):
        """
        visualize the feature importance for lightgbm classifier
        """
        feat_imp = pd.DataFrame(zip(model.feature_importance(importance_type='gain'), features_name),
                                columns=['Value', 'Feature']).sort_values(by="Value", ascending=False)
        logger.info('特征重要性：%s'%feat_imp)

        # plot importance
        fig, ax = plt.subplots(figsize=(12, 4))
        top_data = feat_imp.iloc[0:max_num_features]
        top_feat_name = top_data['Feature'].values
        sns.barplot(x="Value", y="Feature", data=top_data)
        ax.set_title('lgb top %s features important'% max_num_features)
        ax.set_yticklabels(labels=top_feat_name)
        pic_name = 'lgb_top_%s_feature_importance_%s.png' % (max_num_features, train_start_time)
        if save_feat_important:
            pic_save_path = os.path.join(conf.FIGURE_DIR, pic_name)
            plt.savefig(pic_save_path)
            logger.info('%s 保存至%s'% (pic_name, pic_save_path))
        plt.show()
        return feat_imp
    
    if use_cv:
         cols,  train_x, train_y = _train_valid_split(
                                                       fe_df,
                                                       train_start_date,
                                                       train_end_date,
                                                       train_on_model_id=train_on_model_id,
                                                       use_2017_fault_data=use_2017_fault_data,
                                                       use_up_sampling_by_smote= use_up_sampling_by_smote,
                                                       use_cv=use_cv,
                                                       use_log=use_log
         )
            
    else:    
        cols, train_fe_df, val_fe_df, train_x, train_y, val_x, val_y = _train_valid_split(
                                                                                           fe_df,
                                                                                           train_start_date,
                                                                                           train_end_date,
                                                                                           valid_start_date,
                                                                                           valid_end_date,  
                                                                                           train_on_model_id,
                                                                                           eval_on_model_id,
                                                                                           use_next_month_fault_data,
                                                                                           next_month_start_date,
                                                                                           next_month_end_date,
                                                                                           use_2017_fault_data,
                                                                                           use_up_sampling_by_smote,
                                                                                           use_log
        )
        
        
    index_cols, cate_cols, cont_cols, label_cols = cols
    if not use_cv:
        val_x_index = val_fe_df[index_cols]
    feature_name = train_x[cate_cols + cont_cols].columns
    if is_eval:
        train_start_time = time()
        if use_cv:
            cv_folds_generator = _get_index_for_cv(train_x,
                                         train_y,
                                         train_date_list,
                                         val_date_list,
                                         n_fold)
            resutls = []
            for _ in tqdm(range(random_search_times)):
                tmp_params_dict = __random_param_generator(search_params_space)
                search_params = dict(list(model_params.items()) + list(tmp_params_dict.items()))
                focal_loss_alpha = search_params.get('focal_loss_alpha',None)
                focal_loss_gamma = search_params.get('focal_loss_gamma',None)
                logger.info('eval参数: %s' % (search_params))
                if focal_loss_alpha and focal_loss_gamma:
                    focal_loss = lambda x,y: _focal_loss_lgb(x, y, focal_loss_alpha, focal_loss_gamma)
                    eval_error = lambda x,y: _focal_loss_lgb_eval_error(x, y, focal_loss_alpha, focal_loss_gamma)
                else:
                     focal_loss, eval_error = None, None
                        
                f1_scores = []
                n_estimators = []
                feats_imp = pd.DataFrame(data=zip([0 for _ in range(len(feature_name))],feature_name), columns=['Value', 'Feature'])
                for i in range(n_fold):
                    evals_result = {}
                    tmp_train_x ,tmp_train_y, tmp_val_x, tmp_val_y = next(cv_folds_generator)
                    train_set = lgb.Dataset(data=tmp_train_x[cate_cols + cont_cols], label=tmp_train_y[USING_LABEL])
                    val_set = lgb.Dataset(data=tmp_val_x[cate_cols + cont_cols], label=tmp_val_y[USING_LABEL],                           reference=train_set)
                    model = lgb.train(
                                          params=search_params, 
                                          train_set=train_set, 
                                          valid_sets=[train_set, val_set],
                                          evals_result = evals_result,
                                          fobj=focal_loss,
                                          feval=eval_error,
                                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                    n_estimator = _log_best_round_of_model(
                                                 model,
                                                 evals_result,
                                                 'valid_1',
                                                 'auc')

                    feat_imp= __feature_imp_plot(
                                              model,
                                              feature_name,
                                              train_start_time,
                                              save_feat_important
                                             )

                    _, f1_score = _eval(
                                           model,
                                           index_cols,
                                           cate_cols, 
                                           cont_cols,
                                           tmp_val_x[index_cols],
                                           tmp_val_x[cate_cols + cont_cols],
                                           tmp_val_y[[USING_LABEL,'flag']],
                                           val_date_list[i][0],
                                           val_date_list[i][1],
                                           'lgb')
        
                    del tmp_train_x, tmp_train_y, tmp_val_x, tmp_val_y
                    gc.collect()
        
                    for feat in feat_imp['Feature']:
                        feat_score = feat_imp.loc[feat_imp.Feature==feat,'Value']
                        feats_imp.loc[feats_imp.Feature==feat,'Value'] += feat_score
                    n_estimators += [n_estimator]
                    f1_scores += [f1_score]
        
                search_params['f1_score'] = np.mean(f1_scores)
                search_params['n_estimators'] = n_estimators 
                search_params['feats_imp'] = (feats_imp//n_flod).sort_values('Value')
                results += [search_params]
                
            results.sort(key=lambda x: x['f1_score'], reverse=True)
            best_params = results[0]
            logger.info('最优参数:{}'.format(best_params))
            return results, best_params    
        
        else:
            num_train_pos = len(train_y[train_y[USING_LABEL]==FAULT_LABEL])
            num_train_neg = len(train_y[train_y[USING_LABEL]!=FAULT_LABEL])
            ratio_train_pos_neg = round(num_train_pos/num_train_neg, 5)
            logger.info('训练集正负样本比:%s:%s(i.e. %s)'%(
                                                       num_train_pos,
                                                       num_train_neg,
                                                       ratio_train_pos_neg))                                       

            num_valid_pos = len(val_y[val_y[USING_LABEL]==FAULT_LABEL])
            num_valid_neg = len(val_y[val_y[USING_LABEL]!=FAULT_LABEL])
            ratio_valid_pos_neg = round(num_valid_pos/num_valid_neg, 5)
            logger.info('验证集正负样本比:%s:%s(i.e. %s)'%(
                                                       num_valid_pos,
                                                       num_valid_neg,
                                                       ratio_valid_pos_neg))
            train_set = lgb.Dataset(data=train_x, label=train_y[USING_LABEL])
            val_set = lgb.Dataset(data=val_x, 
                           label=val_y[USING_LABEL],
                           reference=train_set)
            
            # do random search for parameter selection
            if use_random_search:
                    results = []
                    feats_imp = pd.DataFrame(data=zip([0 for _ in range(len(feature_name))],feature_name), columns=['Value', 'Feature'])
                    for _ in tqdm(range(random_search_times)):
                        tmp_params_dict = __random_param_generator(search_params_space)
                        search_params = dict(list(model_params.items()) + list(tmp_params_dict.items()))
                        focal_loss_alpha = search_params.get('focal_loss_alpha',None)
                        focal_loss_gamma = search_params.get('focal_loss_gamma',None)
                        evals_result = {}
                        logger.info('eval参数: %s' % (search_params))
                        if focal_loss_alpha and focal_loss_gamma:
                            focal_loss = lambda x,y: _focal_loss_lgb(x, y, focal_loss_alpha, focal_loss_gamma)
                            eval_error = lambda x,y: _focal_loss_lgb_eval_error(x, y, focal_loss_alpha, focal_loss_gamma)
                        else:
                             focal_loss, eval_error = None, None

                        model = lgb.train(
                                          params=search_params, 
                                          train_set=train_set, 
                                          valid_sets=[train_set, val_set],
                                          evals_result = evals_result,
                                          fobj=focal_loss,
                                          feval=eval_error,
                                          early_stopping_rounds=EARLY_STOPPING_ROUNDS
                        )
            ##                               learning_rates=lambda iter: 0.1 * (0.995 ** iter) if 0.1 * (0.995 ** iter) > 1e-3 else 1e-3)

                        # log best round of lgb
                        n_estimators = _log_best_round_of_model(
                                                 model,
                                                 evals_result,
                                                 'valid_1',
                                                 'auc')

                        train_end_time = time()
                        logger.info('模型训练用时:%s'%get_time_diff(train_start_time,
                                                                   train_end_time))

                        feat_imp = __feature_imp_plot(
                                              model,
                                              feature_name,
                                              train_start_time,
                                              save_feat_important
                                             )

                        eval_df, f1_score,precision, recall = _eval(
                                                                   model,
                                                                   index_cols,
                                                                   cate_cols, 
                                                                   cont_cols,
                                                                   val_x_index,
                                                                   val_x,
                                                                   val_y,
                                                                   valid_start_date,
                                                                   valid_end_date,
                                                                   'lgb')
                        search_params['f1_score'] = f1_score
                        search_params['n_estimators'] = n_estimators 
                        search_params['precision'] = precision
                        search_params['recall'] = recall
                        
                        for feat in feat_imp['Feature']:
                            feat_score = feat_imp.loc[feat_imp.Feature==feat,'Value']
                            feats_imp.loc[feats_imp.Feature==feat,'Value'] += feat_score
                        results += [search_params]
                        
                    results.sort(key=lambda x: x['f1_score'], reverse=True)
                    best_params = results[0]
                    feats_imp['Value'] = feats_imp['Value']/random_search_times
                    feats_imp = feats_imp.sort_values(by='Value',ascending=False)
                    logger.info('最优参数:{}'.format(best_params))
                    return results, best_params, feats_imp

            else:
                evals_result = {}
                focal_loss_alpha = model_params.get('focal_loss_alpha',None)
                focal_loss_gamma = model_params.get('focal_loss_gamma',None)
                logger.info('eval参数: %s' % (model_params))
                if focal_loss_alpha and focal_loss_gamma:
                    focal_loss = lambda x,y: _focal_loss_lgb(x, y, focal_loss_alpha, focal_loss_gamma)
                    eval_error = lambda x,y: _focal_loss_lgb_eval_error(x, y, focal_loss_alpha, focal_loss_gamma)
                else:
                     focal_loss, eval_error = None, None

                model = lgb.train(
                                  params=model_params, 
                                  train_set=train_set, 
                                  valid_sets=[train_set, val_set],
                                  evals_result = evals_result,
                                  fobj=focal_loss,
                                  feval=eval_error,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS
                )

                # log best round of lgb
                _ = _log_best_round_of_model(
                                         model,
                                         evals_result,
                                         'valid_1',
                                         'auc')
                train_end_time = time()
                logger.info('模型训练用时:%s'%get_time_diff(
                                                         train_start_time,
                                                         train_end_time)
                           )
                feat_imp = __feature_imp_plot(     
                                              model,
                                              feature_name,
                                              train_start_time,
                                              save_feat_important
                                             )

                eval_df, f1_score,_,_ = _eval(
                                           model,
                                           index_cols,
                                           cate_cols, 
                                           cont_cols,
                                           val_x_index,
                                           val_x,
                                           val_y,
                                           valid_start_date,
                                           valid_end_date,
                                           'lgb'
                )

                return model, eval_df, feat_imp
            
    # train model 
    else:
        num_train_pos = len(train_y[train_y[USING_LABEL]==FAULT_LABEL])
        num_train_neg = len(train_y[train_y[USING_LABEL]!=FAULT_LABEL])
        ratio_train_pos_neg = round(num_train_pos/num_train_neg, 5)
        logger.info('训练集正负样本比:%s:%s(i.e. %s)'%(
                                                   num_train_pos,
                                                   num_train_neg,
                                                   ratio_train_pos_neg))  
        
        focal_loss_alpha = model_params.get('focal_loss_alpha',None)
        focal_loss_gamma = model_params.get('focal_loss_gamma',None)                                                                     
        if focal_loss_alpha and focal_loss_gamma:
            focal_loss = lambda x,y: _focal_loss_lgb(x, y, focal_loss_alpha, focal_loss_gamma)
            eval_error = lambda x,y: _focal_loss_lgb_eval_error(x, y, focal_loss_alpha, focal_loss_gamma)
        else:
             focal_loss, eval_error = None, None
        
        train_start_time = time()
        train_set = lgb.Dataset(data=train_x, label=train_y[USING_LABEL])
        evals_result = {}
        logger.info('train参数:%s' % model_params)
        model = lgb.train(params=model_params, 
                          train_set=train_set, 
                          valid_sets=[train_set],
                          evals_result = evals_result,
                          fobj=focal_loss,
                          feval=eval_error,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        
        _ = _log_best_round_of_model(model,
                                 evals_result,
                                 'training',
                                 'auc')
        
        train_end_time = time()
        logger.info('模型训练用时:%s'%get_time_diff(train_start_time,
                                                  train_end_time))
        
        _= __feature_imp_plot(
                              model,
                              feature_name,
                              train_start_time,
                              save_feat_important,
                             )
        
        # save the trained model
        if model_save_path is not None:
            save_model(model_save_path, 
                          (index_cols, 
                           cate_cols, 
                           cont_cols,
                           label_cols, 
                           feature_name,
                           model)
                         )
        return model

@timer(logger)
def train_pipeline_xgboost(
                       fe_df, 
                       model_params,
                       eval_on_model_id,
                       train_on_model_id,
                       train_start_date,
                       train_end_date,
                       is_eval,
                       valid_start_date,
                       valid_end_date,
                       use_log,
                       save_feat_important,
                       next_month_start_date,
                       next_month_end_date,
                       use_next_month_fault_data,
                       use_2017_fault_data,
                       model_name=None,
                       model_save_path=None,
                       ):
    
        def __feature_imp_plot_xgboost(model, 
                          features_name,
                          train_start_time,
                          save_feat_important=False,
                          max_num_features=30,
                          font_scale=0.7):
            """
            visualize the feature importance for lightgbm classifier
            """
            feat_imp = pd.DataFrame.from_dict(model.get_score(importance_type='gain'), orient="index", columns=['Value'])
            feat_imp = feat_imp.reset_index().rename(columns = {"index":"Feature"}).sort_values(by="Value", ascending=False)

            logger.info('特征重要性：%s'%feat_imp)

            # plot importance
            fig, ax = plt.subplots(figsize=(12, 4))
            top_data = feat_imp.iloc[0:max_num_features]
            top_feat_name = top_data['Feature'].values
            sns.barplot(x="Value", y="Feature", data=top_data)
            ax.set_title('xgboost top %s features important'% max_num_features)
            ax.set_yticklabels(labels=top_feat_name)
            pic_name = 'xgboost_top_%s_feature_importance_%s.png' % (max_num_features, train_start_time)
            if save_feat_important:
                pic_save_path = os.path.join(conf.FIGURE_DIR, pic_name)
                plt.savefig(pic_save_path)
                logger.info('%s 保存至%s'% (pic_name, pic_save_path))
            plt.show()
            return feat_imp
    
        cols, train_fe_df, val_fe_df, train_x, train_y, val_x, val_y = _train_valid_split(fe_df,
                                                                                       train_start_date,
                                                                                       train_end_date,
                                                                                       valid_start_date,
                                                                                       valid_end_date,  
                                                                                       train_on_model_id,
                                                                                       eval_on_model_id,
                                                                                       use_next_month_fault_data,
                                                                                       next_month_start_date,
                                                                                       next_month_end_date,
                                                                                       use_2017_fault_data,
                                                                                       use_log)
    
        index_cols, cate_cols, cont_cols, label_cols = cols
        val_x_index = val_fe_df[index_cols]
        train_x = pd.get_dummies(train_x, columns=cate_cols)
        val_x = pd.get_dummies(val_x, columns=cate_cols)
        feature_name = train_x.columns

        if is_eval:

            num_train_pos = len(train_fe_df[train_fe_df[USING_LABEL]==FAULT_LABEL])
            num_train_neg = len(train_fe_df[train_fe_df[USING_LABEL]!=FAULT_LABEL])
            ratio_train_pos_neg = round(num_train_pos/num_train_neg, 5)
            logger.info('训练集正负样本比:%s:%s(i.e. %s)'%(
                                                           num_train_pos,
                                                           num_train_neg,
                                                           ratio_train_pos_neg))                                       

            num_valid_pos = len(val_fe_df[val_fe_df[USING_LABEL]==FAULT_LABEL])
            num_valid_neg = len(val_fe_df[val_fe_df[USING_LABEL]!=FAULT_LABEL])
            ratio_valid_pos_neg = round(num_valid_pos/num_valid_neg, 5)
            logger.info('验证集正负样本比:%s:%s(i.e. %s)'%(
                                                           num_valid_pos,
                                                           num_valid_neg,
                                                           ratio_valid_pos_neg))
            train_start_time = time()
            train_set = xgb.DMatrix(data=train_x, label=train_y[USING_LABEL])
            val_set = xgb.DMatrix(data=val_x, label=val_y[USING_LABEL])

            evals_result = {}

            logger.info('eval参数:%s' % model_params)

            model = xgb.train(params = model_params,
                              num_boost_round = model_params["num_boost_round"],
                              dtrain = train_set, 
                              evals = [(train_set, 'train'), (val_set, 'eval')],
                              evals_result = evals_result, 
                              early_stopping_rounds = EARLY_STOPPING_ROUNDS
                              )

                # log best round of xgb
            _log_best_round_of_model(model,
                                     evals_result,
                                     'eval',
                                     'auc')
            train_end_time = time()
            logger.info('模型训练用时:%s'%get_time_diff(train_start_time,
                                                     train_end_time))
            _ = __feature_imp_plot_xgboost(model,
                                  feature_name,
                                  train_start_time,
                                  save_feat_important
                                  )

            eval_df, f1_score = _eval(
                                       model,
                                       val_x_index,
                                       val_x,
                                       val_y,
                                       valid_start_date,
                                       valid_end_date,
                                       model_name = model_name)

            return (model, eval_df, f1_score)

        # train model 
        else:
            num_train_pos = len(train_fe_df[train_fe_df[USING_LABEL]==FAULT_LABEL])
            num_train_neg = len(train_fe_df[train_fe_df[USING_LABEL]!=FAULT_LABEL])
            ratio_train_pos_neg = round(num_train_pos/num_train_neg, 5)
            logger.info('训练集正负样本比:%s:%s(i.e. %s)'%(
                                                       num_train_pos,
                                                       num_train_neg,
                                                       ratio_train_pos_neg))  

            train_start_time = time()
            train_set = xgb.DMatrix(data=train_x, label=train_y[USING_LABEL])

            evals_result = {}

            logger.info('train参数:%s' % model_params)

            model = xgb.train(params = model_params, 
                              num_boost_round = model_params["num_boost_round"],
                              dtrain = train_set, 
                              evals = [(train_set, 'train')],
                              evals_result = evals_result, 
                              early_stopping_rounds = EARLY_STOPPING_ROUNDS
                              )

            _log_best_round_of_model(model,
                                     evals_result,
                                     'training',
                                     'auc')
            train_end_time = time()
            logger.info('模型训练用时:%s'%get_time_diff(train_start_time,
                                                      train_end_time))
            _ = _feature_imp_plot_xgb(model,
                                  feature_name,
                                  train_start_time,
                                  save_feat_important,
                                 )

            # save the trained model
            if model_save_path is not None:
                save_model(model_save_path, 
                              (index_cols, 
                               cate_cols, 
                               cont_cols,
                               label_cols, 
                               feature_name,
                               model)
                             )
            return (model, scaler) if use_standard else (model, '')
    
@timer(logger)
def train(
          fe_filename,
          model_params, 
          model_name,
          is_eval,
          drop_cols = [],
          train_sample_ratio=0.2,
          valid_sample_ratio=None,
          save_sample_data=False,
          save_feat_important=False,
          train_start_date='2018-01-01',
          train_end_date='2018-05-31',
          valid_start_date='2100-12-31',
          valid_end_date='2100-12-31',
          train_on_model_id=None,
          eval_on_model_id=None,
          next_month_start_date='2018-06-01',
          next_month_end_date = '2018-06-30',
          use_next_month_fault_data=True,
          use_2017_fault_data = True,
          use_sampling=False,
          use_sampling_by_month_with_weight=False,
          use_sampling_by_power_on_hours=False,
          use_sampling_by_clustering_label=False,
          use_up_sampling_by_smote=False,
          use_random_search=False,
          random_search_times=20,
          search_params_space=dict(),
          use_log = False,
          use_cv=False,
          train_date_list=[],
          val_date_list=[],
          n_fold=4,
          random_state=1
):
    
    if is_eval:
        logger.info("当前模式:eval, eval on model %s, train on model %s, 使用的数据集:%s, 当前使用模型:%s, 使用cv: %s, use_random_search: %s, 训练集日期:%s - %s, 验证集日期:%s - %s, 分类阈值: %s, 截断个数: %s, 下采样：%s, 上采样：%s, 用的label：%s"%                                                                                                                                                                                           (
            eval_on_model_id, 
            train_on_model_id,
            fe_filename,
            model_name, 
            use_cv,
            use_random_search,
            train_start_date,
            train_end_date,
            valid_start_date,
            valid_end_date,
            CLS_RANKING,
            NUM_SUBMISSION,
            use_sampling,
            use_up_sampling_by_smote,
            USING_LABEL 
            )) 
    else:
        logger.info("当前模式:train, 使用的数据集:%s, 当前使用模型:%s, 训练日期:%s - %s" %(
                                                    model_name, 
                                                    fe_filename,
                                                    train_start_date,
                                                    train_end_date,
                                                          )) 
        
    fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, fe_filename))
    if use_sampling:
        fe_df = _sampling(
                                      fe_df,
                                      is_eval,
                                      use_sampling_by_month_with_weight,
                                      use_sampling_by_power_on_hours,
                                      use_sampling_by_clustering_label,
                                      valid_start_date,
                                      valid_end_date,
                                      train_start_date,
                                      train_end_date,             
                                      train_sample_ratio,
                                      valid_sample_ratio,
                                      next_month_start_date,
                                      next_month_end_date,
                                      use_next_month_fault_data,
                                      use_2017_fault_data,
                                      random_state)
        if  use_sampling_by_month_with_weight:
            save_path = os.path.join(conf.DATA_DIR,'sample_by_month_%s_%s'% (train_sample_ratio,fe_filename))
        elif use_sampling_by_power_on_hours:
            save_path = os.path.join(conf.DATA_DIR,'sample_by_power_on_hours_%s_%s'% (train_sample_ratio,fe_filename))
        elif use_sampling_by_clustering_label:
            save_path = os.path.join(conf.DATA_DIR,'sample_by_clustering_%s_%s'% (train_sample_ratio,fe_filename))  
        else:
            raise NotImplementedError('请选择一种采样方式') 
        if not os.path.exists(save_path) and save_sample_data:
                fe_df.reset_index(drop=True,inplace=True)
                fe_df.to_feather(save_path)
                logger.info('采样文件已保存至: %s'%save_path)
          
    if drop_cols:
        fe_df.drop(columns=drop_cols, inplace=True)
    
#     model_save_path = os.path.join(conf.TRAINED_MODEL_DIR, "%s.model" % model_name) if not is_eval else None
    model_save_path = os.path.join(conf.TRAINED_MODEL_DIR, "%s.model.%s" % (model_name, datetime.now().isoformat())) if not is_eval else       None
    if model_name == 'lgb':
        ret = train_pipeline_lgb(
                           fe_df,
                           model_params,
                           eval_on_model_id,
                           train_on_model_id,
                           train_start_date,
                           train_end_date,
                           is_eval,
                           valid_start_date, 
                           valid_end_date,
                           use_log,
                           save_feat_important,
                           next_month_start_date,
                           next_month_end_date ,
                           use_next_month_fault_data,
                           use_2017_fault_data,
                           use_random_search,
                           random_search_times,
                           search_params_space,
                           use_up_sampling_by_smote,
                           use_cv,
                           train_date_list,
                           val_date_list,
                           n_fold,
                           model_save_path,)   
        
    elif model_name == 'xgboost':
        ret = train_pipeline_xgboost(
                       fe_df, 
                       model_params,
                       eval_on_model_id,
                       train_on_model_id,
                       train_start_date,
                       train_end_date,
                       is_eval,
                       valid_start_date,
                       valid_end_date,
                       use_log,
                       save_feat_important,
                       next_month_start_date,
                       next_month_end_date,
                       use_next_month_fault_data,
                       use_2017_fault_data,
                       model_name,
                       model_save_path              
                       )
    
    elif model_name == 'stacking':
        # TODO: 增加stacking部分train pipeline
        raise NotImplementedError('stacking model has not been implemented')
    elif model_name == 'lgb_lr':
        trained_model_save_path = get_latest_model(conf.TRAINED_MODEL_DIR, 'lgb.model' )
        trained_model =  load_model(trained_model_save_path)[5]
        ret = train_pipeline_lgb_lr(
                                   fe_df, 
                                   model_params,
                                   trained_model,
                                   eval_on_model_id,
                                   train_on_model_id,
                                   train_start_date,
                                   train_end_date,
                                   is_eval,
                                   valid_start_date,
                                   valid_end_date,
                                   use_log,
                                   next_month_start_date,
                                   next_month_end_date,
                                   use_next_month_fault_data,
                                   use_2017_fault_data,
                                   model_save_path=None)
        
    else:
        raise NotImplementedError('%s was not implemented' % model_type)
        
    del fe_df
    gc.collect()
    logger.info("%s模型训练完成!模型保存至:%s" % (model_name, model_save_path)) if not is_eval else logger.info("%s模型训练完成!"% model_name) 
    
    return ret 


if __name__ == '__main__': 
    pass
