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

import lightgbm as lgb
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import derivative
from tqdm import tqdm

from .feature_engineering import (
    USING_LABEL,
    FAULT_LABEL,
)
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
)
import conf

# global setting
# LogManager.created_filename = os.path.join(conf.LOG_DIR, 'mlpipeline.log')
logger = LogManager.get_logger(__name__)

# global varirable
MIN_X = -1000000
MAX_X = 1000000
SMALL_VALUE = 0.01
DEFAULT_MISSING_STRING = 'U'
CLASS_NAME = ['无故障','有故障']
EARLY_STOPPING_ROUNDS=10
CLS_RANKING = 0.994  # 0.996

def _f1_score(eval_df):
    eval_fault_df = eval_df.loc[eval_df.prediction == FAULT_LABEL]
    eval_fault_df = eval_fault_df.sort_values('prob', ascending=False)
    eval_fault_df = eval_fault_df.drop_duplicates(['serial_number', 'model'])
    eval_fault_df.reset_index(drop=True, inplace=True)
    eval_fault_df = eval_fault_df.iloc[:100]
    def __precision():
            mask = eval_fault_df['prediction']==eval_fault_df[USING_LABEL]
            ntpp = len(eval_fault_df[mask])
            npp = len(eval_fault_df)
            return  ntpp / npp
    def __recall():
            tmp_df = eval_df[eval_df['flag']==FAULT_LABEL]
            npr = len(tmp_df) 
            mask = tmp_df['prediction']==tmp_df['flag']
            ntpr = len(tmp_df[mask])
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
#         train_fe_df = fe_df[fe_df['dt'] >= train_start_date] 
#         train_fe_df = train_fe_df[train_fe_df['dt'] <= train_end_date]   
        train_idx = fe_df[train_mask].index
    
        val_start_date = val_date_list[i][0]
        val_end_date = val_date_list[i][1]
        val_mask = fe_df['dt'] >= val_start_date 
        val_mask &= fe_df['dt'] <= val_end_date   
#         val_fe_df = fe_df[fe_df['dt'] >= val_start_date] 
#         val_fe_df = val_fe_df[val_fe_df['dt'] <= val_end_date]   
        val_idx = fe_df[val_mask].index
        fold_idx = (train_idx, val_idx)
        folds.append(fold_idx)     
    return iter(folds)        

def __learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

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
        val_x = val_fe_df[cate_cols]
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
                             train_fe_df[cont_cols]],axis=1)
        val_x = pd.concat([val_fe_df[cate_cols],
                           val_fe_df[cont_cols]], axis=1)
       
    train_y = train_fe_df[label_cols]
    val_x_index = val_fe_df[index_cols]
    val_y = val_fe_df[label_cols]
    
    # train
    feature_name = train_x.columns
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
#             focal_loss = lambda x,y: _focal_loss_lgb(x, y, focal_loss_alpha, focal_loss_gamma)
#             eval_error = lambda x,y: _focal_loss_lgb_eval_error(x, y, focal_loss_alpha, focal_loss_gamma)
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

            # eval on raw prediction
            eval_df = pd.concat([val_y[label_cols], val_x_index], axis=1)
            eval_df.loc[:,'prob'] = model.predict(data=val_x)
#             eval_df = eval_df[eval_df.model==eval_on_model_id]
            eval_df.loc[:,'rank'] = eval_df['prob'].rank()
            eval_df.loc[:,'prediction'] = (eval_df['rank']>=eval_df.shape[0] * CLS_RANKING).astype(int)
            acc = metrics.accuracy_score(eval_df[USING_LABEL], eval_df['prediction'])
            report = metrics.classification_report(eval_df[USING_LABEL], eval_df['prediction'], target_names=CLASS_NAME, digits=4)
            confusion = metrics.confusion_matrix(eval_df[USING_LABEL], eval_df['prediction'])
            msg = 'Val Acc: {0:>6.2%}'
            logger.info("分类阈值: %s"%CLS_RANKING)
            logger.info(msg.format(acc))
            logger.info("Precision, Recall and F1-Score...")
            logger.info(report)
            logger.info("Confusion Matrix...")
            logger.info(confusion)

            # eval on competition index by topk 
            precision, recall, f1_score = _f1_score(eval_df)
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
            "eval_set" : [(val_x,val_y[USING_LABEL])],
            'eval_names': ['valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=__learning_rate_010_decay_power_0995)],
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
        logger.info('train参数:%s' % model_params)
        model = lgb.train(params=model_params, 
                          train_set=train_set, 
                          valid_sets=[train_set],
                          evals_result = evals_result, 
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        
#         print(evals_result)
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
          drop_cols  = [],
          focal_loss_alpha=6, 
          focal_loss_gamma=0.9,
          save_feat_important=False,
          train_date_list=[],
          val_date_list=[],
          use_standard=False,
          use_log=False,
          use_cv=False,
          n_fold=3,
          train_on_model_id=None,
          eval_on_model_id=None,
          train_fe_filename='train_fe_df.h5',
          valid_start_date='2100-12-31',
          valid_end_date='2100-12-31'):
    
    if is_eval:
        logger.info("当前模式:eval on model %s, train on model %s, 当前使用模型:%s, 使用cv:%s, 训练集日期:%s - %s, 验证集日期:%s - %s"%                                                                                          (eval_on_model_id, 
                                                                                        train_on_model_id,
                                                                                        model_name, 
                                                                                        use_cv,
                                                                                        train_start_date,
                                                                                        train_end_date,
                                                                                        valid_start_date,
                                                                                        valid_end_date)) 
    else:
        logger.info("当前模式:train, 当前使用模型:%s, 训练日期:%s - %s" %(
                                                                    model_name, 
                                                                    train_start_date,
                                                                    train_end_date,
                                                                          )) 
        
    train_fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, train_fe_filename))
#     train_fe_df.loc[:,'smart_9raw_in_day_unit'] = train_fe_df['smart_9raw']//24
#     train_fe_df.loc[:,'smart_9raw_weight'] = round(train_fe_df['smart_9raw_in_day_unit']/1500,2)
    if drop_cols:
        train_fe_df.drop(columns=drop_cols, inplace=True)
        
    model_save_path = os.path.join(conf.TRAINED_MODEL_DIR, "%s.model.%s" % (model_name, datetime.now().isoformat())) if not                           is_eval else None

    if model_name == 'lgb':
        ret = train_pipeline_lgb(train_fe_df,
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
    else:
        raise NotImplementedError('%s was not implemented' % model_type)
        
    del train_fe_df
    gc.collect()
    
    logger.info("%s模型训练完成!模型保存至:%s" % (model_name, model_save_path)) if not is_eval else logger.info("%s模型训练完成!"%       model_name) 
    return ret 


if __name__ == '__main__': 
     pass