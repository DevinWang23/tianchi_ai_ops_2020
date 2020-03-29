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
import gc
from collections import defaultdict

import lightgbm as lgb
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tqdm import tqdm

sys.path.append('../')
from utils import (
    check_columns, 
    check_category_column, 
    save_model, 
    transform_category_column,
    get_time_diff,
    load_model,
    LogManager,
    timer,
    standard_scale,
    log_scale,
    remove_cont_cols_with_small_std,
    correct_column_type
)
import conf
from mlpipeline.feature_engineering import (
    USING_LABEL,
    FAULT_LABEL,
)

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'train.log')
logger = LogManager.get_logger(__name__)

# global varirable
CLASS_NAME = ['无故障','有故障']
EARLY_STOPPING_ROUNDS=10
CLS_RANKING = 0.997  # 0.996,0.994
NUM_SUBMISSION = 30

def _f1_score( eval_df,
          valid_end_date):
    def __precision():
            tmp_df = eval_df[eval_df['pred']==1]
            mask = tmp_df['pred']==tmp_df['tag']
            ntpp = len(tmp_df[mask])
            npp = len(tmp_df)
            return  ntpp / npp
    def __recall():
        #     tmp_df = eval_df[eval_df[USING_LABEL]==FAULT_LABEL] 
        #      npr = len(tmp_df) 
#             fault_time_df = pd.read_csv(os.path.join(conf.DATA_DIR,'disk_sample_fault_tag.csv'),usecols=                                               ['model','serial_number','fault_time'], parse_dates=                   ['fault_time'])
#             tmp_df = tmp_df.merge(fault_time_df, how='left',on=['model','serial_number'])
#             mask = (tmp_df['pred']==tmp_df[USING_LABEL]) & (tmp_df['fault_time']<=valid_end_date)
#             ntpr = len(tmp_df[mask])
            fault_tmp_df = eval_df[eval_df['flag']==1] 
            npr = len(fault_tmp_df)
            pred_tmp_df = eval_df[eval_df['pred']==1]
            mask =  (pred_tmp_df['model'].isin(fault_tmp_df['model'])) & (pred_tmp_df['serial_number'].isin(fault_tmp_df['serial_number']))
            ntpr = len(pred_tmp_df[mask])
            return ntpr / npr
    precision, recall = __precision(), __recall()
    return precision, recall, 2* precision * recall / (precision + recall)
        
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

def _retag(fe_df, 
        num_tag):
    fault_mask = fe_df['tag'] == FAULT_LABEL
    fault_disk_df = fe_df[fault_mask]
    normal_disk_df = fe_df[~fault_mask]
    del fe_df
    gc.collect()
    
    group_cols = ['model','serial_number']
    fault_sub_dfs = dict(tuple(fault_disk_df.groupby(group_cols)))
    results = []
    for x in fault_sub_dfs:
        fault_sub_dfs[x]['30_tag'] = 1
        fault_sub_dfs[x]['tag'].iloc[:-num_tag] = 0
        results += [fault_sub_dfs[x]]
    normal_disk_df['30_tag'] = 0
    results += [normal_disk_df]
    return pd.concat(results,axis=0)

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

def _feature_imp_plot_lgb(model, 
                          features_name,
                          train_start_time,
                          save_feat_important,
                          max_num_features=30,
                          font_scale=0.7):
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
@timer(logger)
def _get_index_for_cv(fe_df,
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
  
    for i in tqdm(range(n_fold)):        
        train_start_date = train_date_list[i][0]
        train_end_date = train_date_list[i][1]
        train_mask = fe_df['dt'] >= train_start_date 
        train_mask &= fe_df['dt'] <= train_end_date
        train_idx = fe_df[train_mask].index
    
        val_start_date = val_date_list[i][0]
        val_end_date = val_date_list[i][1]
        val_mask = fe_df['dt'] >= val_start_date 
        val_mask &= fe_df['dt'] <= val_end_date   
        val_idx = fe_df[val_mask].index
        fold_idx = (train_idx, val_idx)
        folds.append(fold_idx)     
    return iter(folds)        

def __learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

@timer(logger)
def _sampling_by_month(fe_df,
               valid_start_date,
               valid_end_date, 
               ratio=1):
    logger.info('采样比：%s'%ratio)
    
    def __clustering():
        pass
    
#     valid_mask = (fe_df['dt']>=valid_start_date) & (fe_df['dt']<=valid_end_date)
#     valid_fe_df = fe_df[valid_mask]
#     train_mask = (fe_df['dt']>=train_start_date) & (fe_df['dt']<=train_end_date)
#     train_fe_df = fe_df[train_mask]
#     del fe_df
#     gc.collect()
     
    fe_df.loc[:,'year'] = fe_df['dt'].dt.year.astype(np.int16)
    fe_df.loc[:,'month'] = fe_df['dt'].dt.month.astype(np.int8)
    fe_sub_dfs = dict(tuple(fe_df.groupby(['year','month'])))
    del fe_df
    gc.collect()
    
    # just do sampling for normal disks
    sample_dfs = []
    for x in tqdm(fe_sub_dfs):
        tmp_df = fe_sub_dfs[x]
        mask = tmp_df[USING_LABEL]==FAULT_LABEL
        sample_dfs += [tmp_df[mask]]
#         sample_dfs += [tmp_df[~mask].sample(len(tmp_df[mask])*ratio, random_state=1)]
        sample_dfs += [tmp_df[~mask].sample(int(len(tmp_df[~mask])*ratio), random_state=1)]
    del fe_sub_dfs
    gc.collect()
    sample_df = pd.concat(sample_dfs, axis=0)
    logger.info('采样后数据集正负样本数：%s : %s'%(
                                                                                                                                                           len(sample_df[sample_df[USING_LABEL]==FAULT_LABEL]), \
                                                                                                                                                           len(sample_df[sample_df[USING_LABEL]!=FAULT_LABEL])))
    sample_df.drop(columns=['year','month'], inplace=True)
    
    return sample_df
#     return pd.concat([sample_df,valid_fe_df],axis=0)

@timer(logger)
def _sampling_by_month_with_weight(
                   fe_df,
                   is_eval,
                   valid_start_date,
                   valid_end_date,
                   train_start_date,
                   train_end_date,
                   train_total_sample_num,
                   valid_total_sample_num,
                   random_state
                   ):
    logger.info('需采样训练集总数：%s,需采样验证集总数：%s'%(train_total_sample_num,
                                          valid_total_sample_num))
    sample_dfs = []
    if is_eval:
        valid_mask = (fe_df['dt']>=valid_start_date) & (fe_df['dt']<=valid_end_date)
        valid_fe_df = fe_df[valid_mask]
        mask = valid_fe_df[USING_LABEL]==FAULT_LABEL
        sample_dfs += [valid_fe_df[mask]]
        sample_dfs += [valid_fe_df[~mask].sample(int(valid_total_sample_num), random_state=random_state)]
        del valid_fe_df
        gc.collect()
    train_mask = (fe_df['dt']>=train_start_date) & (fe_df['dt']<=train_end_date)
    train_fe_df = fe_df[train_mask]
    del fe_df
    gc.collect()
     
    train_fe_df.loc[:,'year'] = train_fe_df['dt'].dt.year.astype(np.int16)
    train_fe_df.loc[:,'month'] = train_fe_df['dt'].dt.month.astype(np.int8)
    train_fe_sub_dfs = dict(tuple(train_fe_df.groupby(['year','month'])))
  
    # cal the decay ratio by the gap between current date and valid or test date
    valid_year, valid_month = tuple(map(lambda x: int(x),valid_start_date.split('-')[:2]))
    total_month_gap = 0
    sample_weight_dict = defaultdict(float)
    for year,month in train_fe_sub_dfs.keys():
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
    
    # do sampling for normal disks of train data
    for year_and_month in tqdm(train_fe_sub_dfs):
        tmp_df = train_fe_sub_dfs[year_and_month]
        mask = tmp_df[USING_LABEL]==FAULT_LABEL
        sample_dfs += [tmp_df[mask]]
        weight_ratio = sample_weight_dict[year_and_month]
        sample_dfs += [tmp_df[~mask].sample(int(train_total_sample_num*weight_ratio), random_state=random_state)]
    del train_fe_sub_dfs
    gc.collect()
    
    # do sampling for normal
    sample_df = pd.concat(sample_dfs, axis=0)
    logger.info('采样后数据集正负样本数：%s : %s'%(
                                   len(sample_df[sample_df[USING_LABEL]==FAULT_LABEL]), \
                                   len(sample_df[sample_df[USING_LABEL]!=FAULT_LABEL])))
    sample_df.drop(columns=['year','month'], inplace=True)
    return sample_df

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
                       train_date_list,
                       val_date_list,
                       n_fold,
                       use_standard,
                       use_log,
                       use_cv,
                       focal_loss_alpha, 
                       focal_loss_gamma,
                       save_feat_important,
                       model_save_path=None,):
    
    index_cols, cate_cols, cont_cols, label_cols = check_columns(fe_df.dtypes.to_dict())
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
#     cate_transform_dict = check_category_column(fe_df, cate_cols)
#     fe_df = transform_category_column(fe_df, cate_transform_dict)
    logger.info('连续性特征数量: %s' % len(cont_cols))
    logger.info('离散性特征数量: %s' % len(cate_cols))
    
    train_fe_df = fe_df[(fe_df['dt'] >= train_start_date) & (fe_df['dt']<=train_end_date)] 
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
        if use_standard:
                logger.info("使用标准化: %s"%use_standard)
                train_fe_df, val_fe_df = standard_scale(cont_cols, 
                                                train_fe_df, 
                                                val_fe_df)
        if use_log:
                logger.info("使用log: %s"%use_log)
                train_fe_df, val_fe_df = log_scale(cont_cols, 
                                           train_fe_df, 
                                           val_fe_df)
        train_x = train_fe_df[cont_cols]
        val_x = val_fe_df[cont_cols]    
    else:
        if use_standard:
                logger.info("使用标准化: %s"%use_standard)
                train_fe_df, val_fe_df = standard_scale(cont_cols, 
                                                train_fe_df, 
                                                val_fe_df)
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
    val_x_index = val_fe_df[index_cols]
    val_y = val_fe_df[label_cols]
    feature_name = train_x.columns
#     focal_loss = lambda x,y: _focal_loss_lgb(x, y, focal_loss_alpha, focal_loss_gamma)
#     eval_error = lambda x,y: _focal_loss_lgb_eval_error(x, y, focal_loss_alpha, focal_loss_gamma)
    if is_eval:
        if not use_cv:
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
          
            
            start_time = time()
            train_set = lgb.Dataset(data=train_x, label=train_y[USING_LABEL])
            val_set = lgb.Dataset(data=val_x, 
                           label=val_y[USING_LABEL],
                           reference=train_set)
            evals_result = {}
#             logger.info('eval参数:%s' % model_params)
#             model = lgb.train(params=model_params, 
#                               train_set=train_set, 
#                               valid_sets=[train_set, val_set],
#                               evals_result = evals_result,
#                               fobj=focal_loss,
#                               feval=eval_error,
#                               early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            logger.info('eval参数:%s' % model_params)
            model = lgb.train(params=model_params, 
                              train_set=train_set, 
                              valid_sets=[train_set, val_set],
                              evals_result = evals_result,
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS,)
#                               learning_rates=lambda iter: 0.1 * (0.995 ** iter) if 0.1 * (0.995 ** iter) > 1e-3 else 1e-3)
            
            # log best round of lgb
            _log_best_round_of_model(model,
                                     evals_result,
                                     'valid_1',
                                     'auc')
            end_time = time()
            logger.info('模型训练用时:%s'%get_time_diff(start_time,end_time))
            _feature_imp_plot_lgb(model,
                                  feature_name,
                                  start_time,
                                  save_feat_important
                                 )

            # eval on valid set 
            results = []
            eval_df = pd.concat([val_x_index,val_x, val_y], axis=1)
            eval_df = eval_df.sort_values('dt')
            start_date, end_date = eval_df.iloc[0]['dt'], eval_df.iloc[-1]['dt'] 
            date_range = pd.date_range(start_date, end_date, freq='D')
            for date in date_range:
                logger.info('验证日期：%s'%date)
                sub_eval_df = eval_df[eval_df.dt==date]
                sub_eval_df.loc[:,'prob'] = model.predict(data=sub_eval_df[cate_cols + cont_cols])
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

            # eval on competition index by topk 
            precision, recall, f1_score = _f1_score(eval_df,
                                       valid_end_date,
                                                   )
            logger.info("竞赛recall: %s"% recall)
            logger.info("竞赛precision: %s"%precision)
            logger.info("竞赛f1-score: %s"%f1_score)
            return (model, eval_df)
        
        # do cross validation for parameter selection
        else:
            from scipy.stats import randint as sp_randint
            from scipy.stats import uniform as sp_uniform
            start_time = time()
            
#             train_set = lgb.Dataset(data=train_x, label=train_y[USING_LABEL])
            folds = _get_index_for_cv(train_fe_df,
                                      train_date_list,
                                      val_date_list,
                                      n_fold)
            
            n_HP_points_to_test = 10
            fit_params={"early_stopping_rounds":10, 
            "eval_metric" : 'auc', 
#             "eval_set" : [(val_x,val_y[USING_LABEL])],
#             'eval_names': ['valid'],
#             'callbacks': [lgb.reset_parameter(learning_rate=__learning_rate_010_decay_power_0995)],
            'verbose': 5,
            'categorical_feature': 'auto'}
            
            param_test ={'num_leaves': [32,64,128], 
             'min_child_samples': [40,60,80,100], 
             'learning_rate':[0.001,0.005],
             'scale_pos_weight':[25,30,35],
             'subsample': [0.8], 
             'colsample_bytree': [0.4,0.5,0.6,0.8],
             'reg_alpha': [0, 0.5, 1, 2],
             'reg_lambda': [0, 0.5, 1, 2]}
            
            clf = lgb.LGBMClassifier(max_depth=-1, 
                                     random_state=314, 
                                     silent=False, 
                                     metric=None, 
                                     n_jobs=4, 
                                     n_estimators=1000,
                                     importance_type='gain')
            gs = RandomizedSearchCV(
            estimator=clf, 
            param_distributions=param_test, 
            n_iter=n_HP_points_to_test,
            scoring='roc_auc',
            cv=folds,
            refit=False,
            random_state=314,
            verbose=True)
            gs.fit(train_x, train_y[USING_LABEL], **fit_params)
            logger.info('Best score reached: %s with params: %s ' % (gs.best_score_, gs.best_params_))
#             cv_res = lgb.cv(params=model_params,
# #                    num_boost_round=MAX_BOOST_ROUNDS,
#                             early_stopping_rounds=EARLY_STOPPING_ROUNDS,
#                             train_set=train_set,
#                             folds=folds,
#                             seed=0,
#                             verbose_eval=5)
            end_time = time()
            logger.info('模型训练用时:%s'%get_time_diff(start_time,end_time))
            return (None, None)
    
    # train model 
    else:
        num_train_pos = len(train_fe_df[train_fe_df[USING_LABEL]==FAULT_LABEL])
        num_train_neg = len(train_fe_df[train_fe_df[USING_LABEL]!=FAULT_LABEL])
        ratio_train_pos_neg = round(num_train_pos/num_train_neg, 5)
        logger.info('训练集正负样本比:%s:%s(i.e. %s)'%(
                                                   num_train_pos,
                                                   num_train_neg,
                                                   ratio_train_pos_neg))  
        
        start_time = time()
        train_set = lgb.Dataset(data=train_x, label=train_y[USING_LABEL])
        evals_result = {}
#         logger.info('train参数:%s' % model_params)
#         model = lgb.train(params=model_params, 
#                           train_set=train_set, 
#                           valid_sets=[train_set],
#                           evals_result = evals_result,
#                           fobj=focal_loss,
#                           feval=eval_error,
#                           early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        logger.info('train参数:%s' % model_params)
        model = lgb.train(params=model_params, 
                          train_set=train_set, 
                          valid_sets=[train_set],
                          evals_result = evals_result, 
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        
        _log_best_round_of_model(model,
                                 evals_result,
                                 'training',
                                 'auc')
        end_time = time()
        logger.info('模型训练用时:%s'%get_time_diff(start_time,end_time))
        _feature_imp_plot_lgb(model,
                              feature_name,
                              start_time,
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
def train(model_params, 
          model_name,
          is_eval,
          train_start_date,
          train_end_date,
          train_total_sample_num=2.7e6,
          valid_total_sample_num=6e5,
          use_retag=False,
          num_tag=10,
          drop_cols = [],
          use_sampling=False,
          sampling_ratio=1,
          save_sample_data=False,
          focal_loss_alpha=6, 
          focal_loss_gamma=0.9,
          save_feat_important=False,
          train_date_list=[],
          val_date_list=[],
          use_standard=False,
          use_log=False,
          use_cv=False,
          n_fold=3,
          random_state=1,
          train_on_model_id=None,
          eval_on_model_id=None,
          fe_filename='train_fe_df.h5',
          valid_start_date='2100-12-31',
          valid_end_date='2100-12-31'):
    
    if is_eval:
        logger.info("当前模式:eval on model %s, train on model %s, 当前使用模型:%s, 使用cv:%s, 训练集日期:%s - %s, 验证集日期:%s - %s, 分类阈值: %s, 截断个数: %s, 采样：%s, 使用的标签：%s"%                                                                                                                                                                                               (eval_on_model_id, 
            train_on_model_id,
            model_name, 
            use_cv,
            train_start_date,
            train_end_date,
            valid_start_date,
            valid_end_date,
            CLS_RANKING,
            NUM_SUBMISSION,
            use_sampling,
            USING_LABEL )) 
    else:
        logger.info("当前模式:train, 当前使用模型:%s, 训练日期:%s - %s" %(
                                                    model_name, 
                                                    train_start_date,
                                                    train_end_date,
                                                          )) 
        
    fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, fe_filename))
    if use_retag:
        fe_df = _retag(fe_df, 
                  num_tag,
                   )
    if use_sampling:
#             fe_df = _sampling_by_month(fe_df, 
#                                   valid_start_date, 
#                                   valid_end_date,
#                                   train_start_date,
#                                   train_end_date,
#                                   sampling_ratio)
#           if not os.path.exists(os.path.join(conf.DATA_DIR,'sample_%s_%s'% (sampling_ratio,fe_filename))) and save_sample_data:
#                 sample_fe_save_path = os.path.join(conf.DATA_DIR,'sample_%s_%s'% (sampling_ratio, fe_filename))
#                 fe_df.reset_index(drop=True,inplace=True)
#                 fe_df.to_feather(sample_fe_save_path)
#                 logger.info('采样文件已保存至%s'%sample_fe_save_path)
          fe_df = _sampling_by_month_with_weight(fe_df,
                                      is_eval,
                                      valid_start_date,
                                      valid_end_date,
                                      train_start_date,
                                      train_end_date,             
                                      train_total_sample_num,
                                      valid_total_sample_num,
                                      random_state)
          if not os.path.exists(os.path.join(conf.DATA_DIR,'sample_%s_%s'% (train_total_sample_num,fe_filename))) and save_sample_data:
                sample_fe_save_path = os.path.join(conf.DATA_DIR,'sample_%s_%s'% (train_total_sample_num, fe_filename))
                fe_df.reset_index(drop=True,inplace=True)
                fe_df.to_feather(sample_fe_save_path)
                logger.info('采样文件已保存至%s'%sample_fe_save_path)
          
    if drop_cols:
        fe_df.drop(columns=drop_cols, inplace=True)
            
    model_save_path = os.path.join(conf.TRAINED_MODEL_DIR, "%s.model" % model_name) if not is_eval else None
#     model_save_path = os.path.join(conf.TRAINED_MODEL_DIR, "%s.model.%s" % (model_name, datetime.now().isoformat()))
    if model_name == 'lgb':
        ret = train_pipeline_lgb(fe_df,
                           model_params,
                           eval_on_model_id,
                           train_on_model_id,
                           train_start_date,
                           train_end_date,
                           is_eval,
                           valid_start_date, 
                           valid_end_date,
                           train_date_list,
                           val_date_list,
                           n_fold,
                           use_standard,
                           use_log,
                           use_cv,
                           focal_loss_alpha, 
                           focal_loss_gamma,
                           save_feat_important,
                           model_save_path,)                      
    elif model_name == 'stacking':
        # TODO: 增加stacking部分train pipeline
        raise NotImplementedError('stacking model has not been implemented')
    elif model_name == 'lr':
        raise NotImplementedError('stacking model has not been implemented') 
    else:
        raise NotImplementedError('%s was not implemented' % model_type)
        
    del fe_df
    gc.collect()
    logger.info("%s模型训练完成!模型保存至:%s" % (model_name, model_save_path)) if not is_eval else logger.info("%s模型训练完成!"% model_name) 
    
    return ret 


if __name__ == '__main__': 
    pass