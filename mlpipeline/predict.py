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
from time import time 
from datetime import timedelta, datetime
from zipfile import ZipFile, ZIP_DEFLATED
import argparse

import lightgbm as lgb
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from utils import (
    check_columns, 
    check_category_column, 
    save_model, 
    transform_category_column,
    get_time_diff,
    load_model,
    LogManager,
    get_latest_model,
    timer,
    log_scale
)
import conf
from mlpipeline.train import (
                    CLS_RANKING,
                    NUM_SUBMISSION)
from mlpipeline.feature_engineering import (
    USING_LABEL,
    FAULT_LABEL,
    feature_engineering
)

# global setting
LogManager.created_filename = os.path.join(conf.LOG_DIR, 'predict.log')
logger = LogManager.get_logger(__name__)

@timer(logger)
def inference_pipeline_ensemble_tree(fe_df, 
                                     use_log,
                                     model_save_path=None,
                                     model_name=None):
    
    index_cols, cate_cols, cont_cols, label_cols, features, model = load_model(model_save_path)
    fe_df = fe_df.sort_values('dt')
    start_date, end_date = fe_df.iloc[0]['dt'], fe_df.iloc[-1]['dt'] 
    date_range = pd.date_range(start_date, end_date, freq='D')
#     lgb.plot_tree(model, dpi=600, show_info=['split_gain','internal_value']
#                  )
#     plt.show()
#     fe_df = transform_category_column(fe_df, cate_transform_dict)
#     print(cont_cols, model.params, model.best_iteration)
#     print(model.feature_importance())
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
    
    submission_df = pd.DataFrame() 
    for date in date_range:
        logger.info('预测日期：%s'%date)
        sub_fe_df = fe_df[fe_df.dt==date]
        if cate_cols and not cont_cols:
            test_features = sub_fe_df[cate_cols]
        elif not cate_cols and cont_cols:
            if use_log:
                    logger.info("使用log: %s"%use_log)
                    sub_fe_df,_ = log_scale(cont_cols, sub_fe_df)
            test_features = sub_fe_df[cont_cols]
        else:
            if use_log:
                    logger.info("使用log: %s"%use_log)
                    sub_fe_df,_ = log_scale(cont_cols, sub_fe_df)
            test_features = pd.concat([sub_fe_df[cate_cols], sub_fe_df[cont_cols]],
                                      axis=1)
        if model_name == "xgboost":
            test_features = pd.get_dummies(test_features, columns=cate_cols)      
            test_x = test_features[features]
            test_x = xgb.DMatrix(data=test_x)  
        else:       
            test_x = test_features[features]
        ret = sub_fe_df[index_cols]
        ret.loc[:,'prob'] = model.predict(test_x)
        ret.loc[:,'rank'] = ret['prob'].rank()
        ret.loc[:,'label'] = (ret['rank']>=ret.shape[0] * CLS_RANKING).astype(int)
        sub_submission_df = ret.loc[ret.label ==  FAULT_LABEL]
        sub_submission_df = sub_submission_df.sort_values('prob', ascending=False)
        top_sub_submission_df = sub_submission_df.reset_index(drop=True, inplace=False).iloc[:NUM_SUBMISSION]
        logger.info('原始预测为fault disk的个数：%s'%len(sub_submission_df))
        
        top_sub_submission_df.loc[:,'manufacturer'] = 'A'
        submission_df = pd.concat([submission_df, top_sub_submission_df[['manufacturer',
                                                                         'model',
                                                                         'serial_number',
                                                                         'dt']]])
    csv_save_filename = 'submission_%s.csv'%datetime.now().isoformat()
    csv_save_path = os.path.join(conf.ROOT_DIR, csv_save_filename)
    submission_df = submission_df.drop_duplicates(['serial_number', 'model'])
    submission_df.to_csv(csv_save_path, index=False, header=False)
    zip_save_path = os.path.join(conf.ROOT_DIR, 'result.zip')
    with ZipFile(zip_save_path,'w') as zf:
              zf.write(csv_save_path, compress_type=ZIP_DEFLATED)
   
    logger.info('最终提交样本个数：%s'%len(submission_df))
    logger.info('csv文件%s已保存至%s，zip文件已保存至%s'%( 
                                                           csv_save_filename,
                                                           csv_save_path,
                                                           zip_save_path
        ))
    
    return submission_df

@timer(logger)
def predict( 
    model_name='lgb', 
    pred_start_date='2018-09-01',
    pred_end_date='2018-09-30',
    scaler='',
    use_log=False,
    num_processes=17,
    is_train=False
):
    logger.info("开始预测, 当前使用模型:%s"% (model_name))
    test_fe_df = feature_engineering(
                                     pred_start_date = pred_start_date,
                                     pred_end_date = pred_end_date,
                                     is_train=False,
                                     num_processes=num_processes)
    logger.info("预测样本数:%s, 分类阈值: %s, 分类每日截断个数：%s" % (test_fe_df.shape[0],
                                                                CLS_RANKING,
                                                                NUM_SUBMISSION))
    model_save_path = get_latest_model(conf.TRAINED_MODEL_DIR, '%s.model' % model_name)
    if model_name == 'lgb':
         submission_df = inference_pipeline_ensemble_tree(test_fe_df, 
                                                              use_log,
                                                              model_save_path=model_save_path,
                                                              model_name=model_name)
    elif model_type == 'stacking':
        # TODO:增加stacking部分inference pipeline
        raise NotImplementedError('%s has not been implemented' % model_name)
    else:
        # TODO:增加神经网络部分
        raise NotImplementedError('%s was not been implemented' % model_name)
   
    logger.info("%s预测完成!" % model_name) 
    return  submission_df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str, help='lgb')
    parser.add_argument('--pred_start_date', required=True, type=str, help='2018-09-01')
    parser.add_argument('--pred_end_date', required=True, type=str, help='2018-09-30')
    parser.add_argument('--use_log', required=False, type=lambda x: (str(x).lower()=='true'), help='using log for normalization or not')
    parser.add_argument('--num_processes', required=True, type=int, help='the num of processes for doing feature engineering')
    parser.add_argument('--is_train', required=True, type=lambda x: (str(x).lower()=='true'), help='flag for identifying train or predict')
    args = parser.parse_args()
    
    predict(is_train=args.is_train,
            model_name=args.model_name,
            pred_start_date=args.pred_start_date,
            pred_end_date=args.pred_end_date,
            use_log=args.use_log,
            num_processes=args.num_processes)
           