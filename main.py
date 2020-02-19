# -*- coding: utf-8 -*-
"""
Author:  MengQiu Wang
Email: wangmengqiu@ainnovation.com
Date: 28/09/2019

Description: 
   Main function to run the time series pipeline
    
"""
import argparse
import traceback

from time_series_pipeline.train import generate_period_training_set, offline_train
from time_series_pipeline.predict import predict
from utils.LogManager import LogManager

parser = argparse.ArgumentParser()
parser.add_argument("--train_date", required=True, type=str, help="2100-01-01")
parser.add_argument("--is_train", required=True, default=True, type=lambda x: (str(x).lower() == 'true'), help="True or False")
parser.add_argument("--model_type", required=True, type=str, help="neural, ensemble or stats")
parser.add_argument("--model", required=True, type=str, help="xgb, lgb, lstm, tcn, lstm_attention or prophet")
parser.add_argument("--train_duration", required=True, type=str, help="20190101-20190201")
parser.add_argument("--start_date", required=False, type=str, default='2000-01-01', help="int(%Y%m%d)")
parser.add_argument("--end_date", required=False, type=str, default='2100-01-01', help="int(%Y%m%d)")
parser.add_argument("--business_type", required=False, type=str, default="sample")
parser.add_argument("--data_dir", required=False, type=str, help="input csv storage path", default="./sample_data")
parser.add_argument("--predict_targets", required=False, type=list, default=[1, 2, 7])
parser.add_argument("--predict_date", required=False, type=str, help="2100-01-01")
parser.add_argument("--time_step", required=True, type=int,
                    help="time step for generating time step data for neural network")
args = parser.parse_args()

if __name__ == "__main__":

    # 训练
    if args.is_train:
        logger = LogManager.get_logger('train')
        model_params_, train_params_ = {}, {}
        try:
            # 用于进行特征工程以生成历史训练数据, 如只需调参，可注释掉此步骤
            generate_period_training_set(data_dir=args.data_dir, train_duration=args.train_duration,
                                         predict_targets=args.predict_targets, logger=logger)

            if args.model == 'xgb':
                train_params_ = {'eval_metric': 'rmse',
                                 'subsample': 0.9,
                                 'eta': 0.1,
                                 'n_thread': 4,
                                 'num_boost_round': 1000,
                                 'verbose_eval': 1,
                                 'early_stopping_rounds': 10,
                                 'loss_function': 'test-rmse-mean'}

                # params tuned by random search, bayes optimization or grid search
                model_params_ = {'max_depth': (3, 7),
                                 'gamma': (0, 1),
                                 'colsample_bytree': (0.3, 0.9)}

            elif args.model == 'lgb':
                train_params_ = {'metric': ['rmse'],
                                 'metric_mean': 'rmse-mean',
                                 'verbose_eval': 1,
                                 'early_stopping_rounds': 10,
                                 'application': 'regression',
                                 'num_iterations': 1000,
                                 'lr': 0.05}

                model_params_ = {'max_depth': (15, 20),
                                 'num_leaves': (30, 45),
                                 'lambda_l2': (0, 2),
                                 'min_split_gain': (0.001, 0.1),
                                 'min_child_weight': (5, 50)
                                 }

            elif args.model == 'lstm':
                train_params_ = {'batch_size': 32,
                                 'shuffle': False,
                                 'patience': 5,
                                 'verbose': True,
                                 'delta': 0.01,
                                 'epoch': 20,
                                 'lr': 0.01}
                model_params_ = {'hidden_size': 100,
                                 'num_layers': 1,
                                 }

            elif args.model == 'tcn':
                train_params_ = {'batch_size': 32,
                                 'shuffle': False,
                                 'patience': 5,
                                 'verbose': True,
                                 'delta': 0.01,
                                 'epoch': 20,
                                 'lr': 0.01
                                 }
                model_params_ = {'num_channels': [10, 10],
                                 'kernel_size': 4,
                                 'dropout': 0.2}

            elif args.model == 'lstm_attention':
                train_params_ = {'batch_size': 32,
                                 'shuffle': False,
                                 'patience': 5,
                                 'verbose': True,
                                 'delta': 0.01,
                                 'epoch': 20,
                                 'lr': 0.01
                                 }
                model_params_ = {'hidden_size': 100,
                                 'num_layers': 1,
                                 'dropout': 0.2}
            else:
                raise ValueError('%s model has not been implemented' % args.model)

            # 用于离线训练模型
            offline_train(train_duration=args.train_duration,
                          predict_targets=args.predict_targets,
                          train_date=args.train_date,
                          model=args.model,
                          business_type=args.business_type,
                          logger=logger,
                          model_params=model_params_,
                          train_params=train_params_,
                          model_type=args.model_type,
                          correct_types={'day_of_week': 'category',
                                         'month_of_year': 'category',
                                         'day_of_month': 'category'},
                          time_step=args.time_step)

        except Exception as e:
            logger.error(str(e))
            logger.error(traceback.format_exc())

    # 预测
    else:
        logger = LogManager.get_logger('predict')
        try:
            predict(predict_date=args.predict_date,
                    model=args.model,
                    data_dir=args.data_dir,
                    business_type=args.business_type,
                    logger=logger,
                    model_type=args.model_type,
                    time_step=args.time_step,
                    correct_types={'day_of_week': 'category',
                                   'month_of_year': 'category',
                                   'day_of_month': 'category'},
                    train_duration=args.train_duration)

        except Exception as e:
            logger.error(str(e))
            logger.error(traceback.format_exc())
