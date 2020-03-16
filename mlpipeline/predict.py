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

import lightgbm as lgb
from sklearn import metrics
import pandas as pd

from .train import CLS_RANKING
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
    standard_scale,
    log_scale
)
import conf
from .feature_engineering import (
    USING_LABEL,
    FAULT_LABEL,
)


# global setting
# LogManager.created_filename = os.path.join(conf.LOG_DIR, 'mlpipeline.log')
logger = LogManager.get_logger(__name__)


@timer(logger)
def inference_pipeline_ensemble_tree(fe_df, 
                                     use_standard,
                                     use_log,
                                     scaler,
                                     pred_on_model_id,
                                     model_save_path=None,
                                     logger=None):
    
    index_cols, cate_cols, cont_cols, label_cols, features, model = load_model(model_save_path)
#     fe_df = transform_category_column(fe_df, cate_transform_dict)
#     print(cont_cols, model.params, model.best_iteration)
#     print(model.feature_importance())
    assert cate_cols is not None or cont_cols is not None, 'feature columns are empty' 
    
    if cate_cols and not cont_cols:
        test_features = fe_df[cate_cols]
    elif not cate_cols and cont_cols:
        if use_standard:
                logger.info("使用标准化: %s"%use_standard)
                fe_df = scaler.transform(cont_cols, fe_df)
        if use_log:
                logger.info("使用log: %s"%use_log)
                fe_df,_ = log_scale(cont_cols, fe_df)
        test_features = fe_df[cont_cols]
    else:
        if use_standard:
                logger.info("使用标准化: %s"%use_standard)
                fe_df = scaler.transform(cont_cols, fe_df)
        if use_log:
                logger.info("使用log: %s"%use_log)
                fe_df,_ = log_scale(cont_cols, fe_df)
        test_features = pd.concat([fe_df[cate_cols], fe_df[cont_cols]],
                                  axis=1)
            
    test_x = test_features[features]
    ret = fe_df[index_cols]
    ret.loc[:,'prob'] = model.predict(test_x)
    ret = ret[ret.model==pred_on_model_id]
    ret.loc[:,'rank'] = ret['prob'].rank()
    ret.loc[:,'label'] = (ret['rank']>=ret.shape[0] * CLS_RANKING).astype(int)
    submission_df = ret.loc[ret.label ==  FAULT_LABEL]
    submission_df = submission_df.sort_values('prob', ascending=False)
    submission_df = submission_df.drop_duplicates(['serial_number', 'model'])
    
    submission_filename = 'submission_%s.csv'%datetime.now().isoformat()
    submission_path = os.path.join(conf.SUBMISSION_DIR, submission_filename)
    submission_df.loc[:,'manufacturer'] = 'A'
    submission_df[['manufacturer','model','serial_number','dt']].to_csv(submission_path, index=False, header=False)
    logger.info("分类阈值: %s"%
                                       CLS_RANKING)
    logger.info('提交样本个数：%s'%
                                      len(submission_df))
    logger.info('%s已保存至%s'%( 
                                    submission_filename,
                                    submission_path))
    return ret, submission_df

@timer(logger)
def predict(model_name, 
            test_month,
            use_standard,
            use_log,
            pred_on_model_id,
            scaler,
            test_fe_filename='test_fe_filename',):
 
    logger.info("开始预测, 当前使用模型:%s, 预测月份:%s"% (model_name, test_month))
    test_fe_df = pd.read_feather(os.path.join(conf.DATA_DIR, test_fe_filename))
    test_fe_df.loc[:,'smart_9raw_in_day_unit'] = test_fe_df['smart_9raw']//24
    test_fe_df.loc[:,'smart_9raw_weight'] = round(test_fe_df['smart_9raw_in_day_unit']/1500,2)
    logger.info("预测样本数:%s" % test_fe_df.shape[0])
    model_save_path = get_latest_model(conf.TRAINED_MODEL_DIR, '%s.model' % model_name)
    if model_name == 'lgb':
        ret, submission_df = inference_pipeline_ensemble_tree(test_fe_df, 
                                                              use_standard,
                                                              use_log,
                                                              scaler,
                                                              pred_on_model_id,
                                                              model_save_path=model_save_path,
                                                              logger=logger)
    elif model_type == 'stacking':
        # TODO:增加stacking部分inference pipeline
        raise NotImplementedError('%s has not been implemented' % model_name)
    else:
        raise NotImplementedError('%s was not been implemented' % model_name)
    logger.info("%s预测完成！" %  model_name)
    return ret, submission_df
    
if __name__ == "__main__":
    pass