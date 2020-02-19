# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 26/12/2019

Description:
    Class for implementing the ensemble tree models
"""

import sys
import os

import lightgbm as lgb
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import numpy as np
from matplotlib.font_manager import FontProperties

from .model import Model

sys.path.append("../../")
from utils.utils import overrides
import conf

FONT = FontProperties(fname=os.path.join(conf.lib_dir, 'simsun.ttc'))


class Lightgbm(Model):
    def __init__(self, X_train, y_train, valid_fold, model_params, train_params):
        self.feature_names = X_train.columns.values.tolist()
        self.dtrain = lgb.Dataset(data=X_train,
                                  label=y_train.values,
                                  feature_name=self.feature_names)
        self.valid_fold = valid_fold
        self.model_params = model_params
        self.train_params = train_params
        self.num_boost_round = []

    @overrides(Model)
    def bayesian_optimization(self, init_points=5, n_iter=5, acq='ei'):
        def __get_cv_result(max_depth, num_leaves, lambda_l2,  min_split_gain, min_child_weight):
            metric_mean = self.train_params['metric_mean']
            params = {'application': self.train_params['application'],
                      'num_iterations': self.train_params['num_iterations'],
                      'learning_rate': self.train_params['lr'],
                      'early_stopping': self.train_params['early_stopping_rounds'],
                      'metric': self.train_params['metric'],
                      'max_depth': int(max_depth),
                      'num_leaves': int(num_leaves),
                      'lambda_l2': max(lambda_l2, 0),
                      'min_split_gain': min_split_gain,
                      'min_child_weight': min_child_weight}

            cv_result = lgb.cv(params,
                               self.dtrain,
                               folds=self.valid_fold,
                               verbose_eval=self.train_params['verbose_eval'],
                               feature_name=self.feature_names)

            self.num_boost_round += [len(cv_result[metric_mean])]

            return -min(cv_result[metric_mean])

        lgb_bo = BayesianOptimization(__get_cv_result, self.model_params, random_state=0)
        lgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq)

        best_params = lgb_bo.max['params']

        # correct some params' types
        try:
            best_params['max_depth'] = int(best_params['max_depth'])
            best_params['num_leaves'] = int(best_params['num_leaves'])
        except KeyError:
            pass

        # get the best params corresponding num boost round
        num_boost_round = self.num_boost_round[lgb_bo.res.index(lgb_bo.max)]

        return best_params, num_boost_round

    @overrides(Model)
    def train(self, params, num_boost_round):
        trained_lgb = lgb.train(params,
                                self.dtrain,
                                num_boost_round,
                                verbose_eval=1,
                                valid_sets=[self.dtrain])

        return trained_lgb

    @overrides(Model)
    def feature_imp_plot(self, model, target, max_num_features=30, font_scale=0.7):
        """
        visualize the feature importance for lightgbm regressor
        """

        feat_imp = pd.DataFrame(sorted(zip(model.feature_importance(), self.feature_names)),
                                columns=['Value', 'Feature'])

        # plot importance
        fig, ax = plt.subplots(figsize=(12, 4))
        top_data = feat_imp.sort_values(by="Value", ascending=False)[0:max_num_features]
        top_feat_name = top_data['Feature'].values
        sns.barplot(x="Value", y="Feature", data=top_data)
        ax.set_title('lgb Features with predict target %s' % target)
        ax.set_yticklabels(labels=top_feat_name, fontproperties=FONT)
        plt.savefig(os.path.join(conf.figs_dir, 'lgb_feature_importance_%s.png' % target))
        plt.show()

    @overrides(Model)
    def grid_search_optimization(self, *args, **kwargs):
        raise NotImplementedError

    @overrides(Model)
    def random_search_optimization(self, *args, **kwargs):
        raise NotImplementedError


class Xgboost(Model):
    def __init__(self, X_train, y_train, valid_fold, model_params, train_params):
        self.dtrain = xgb.DMatrix(X_train.values,
                                  label=y_train.values,
                                  feature_names=X_train.columns)
        self.valid_fold = valid_fold  # the list of indexes of validation data
        self.train_params = train_params
        self.model_params = model_params
        self.num_boost_round = []

    @overrides(Model)
    def bayesian_optimization(self, init_points=5, n_iter=5, acq='ei'):
        def __get_cv_result(max_depth, gamma, colsample_bytree):
            """
            using cross validation for params selection for bo, function params
            are the hyper-parameter required to be tuned, if some params
            are not required to be tuned, put them in model_params
            """
            params = {'eval_metric': self.train_params['eval_metric'],
                      'max_depth': int(max_depth),
                      'subsample': self.train_params['subsample'],
                      'eta': self.train_params['eta'],
                      'gamma': gamma,
                      'colsample_bytree': colsample_bytree,
                      'n_thread': self.train_params['n_thread']}

            cv_result = xgb.cv(params,
                               self.dtrain,
                               num_boost_round=self.train_params['num_boost_round'],
                               folds=self.valid_fold,
                               verbose_eval=self.train_params['verbose_eval'],
                               early_stopping_rounds=self.train_params['early_stopping_rounds'])

            # used for find the best number of tree
            self.num_boost_round += [cv_result.shape[0]]

            return -1 * cv_result[self.train_params['loss_function']].iloc[-1]  # 'test-rmse-mean'

        xgb_bo = BayesianOptimization(__get_cv_result, self.model_params, random_state=0)
        xgb_bo.maximize(init_points=init_points, n_iter=n_iter, acq=acq)
        best_params = xgb_bo.max['params']

        # correct some params' types
        try:
            best_params['max_depth'] = int(best_params['max_depth'])
            best_params['num_leaves'] = int(best_params['num_leaves'])
        except KeyError:
            pass

        # get the best params corresponding num boost round
        num_boost_round = self.num_boost_round[xgb_bo.res.index(xgb_bo.max)]

        return best_params, num_boost_round

    def __mape_ln(self, y):
        """
        user-defined the xgb's evalution metric
        :param y: ndarray - prediction of xgboost
        :param dtrain: DMatrix
        :return: str - name of metric, float - mape value
        """
        c = self.dtrain.get_label()
        # result = np.sum(np.abs((np.expm1(y) - np.expm1(c)) / np.expm1(c))) / len(c)
        result = np.sum(np.abs((y - c) / c)) / len(c)
        return "mape", result

    def __huber_approx_obj(self, preds, h_value=2.7):
        """
        the first-order gradient  and the second-order gradient of mape
        :param preds: ndarray - prediction of xgboost
        :param dtrain: DMatrix
        :return: float - first-order gradient, float - the second order gradient
        """
        d = self.dtrain - preds
        h = h_value
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess

    @overrides(Model)
    def train(self, params, num_boost_round):
        trained_xgb = xgb.train(params,
                                self.dtrain,
                                num_boost_round,
                                verbose_eval=1,
                                evals=[(self.dtrain, 'val')])

        return trained_xgb

    @overrides(Model)
    def feature_imp_plot(self, model, target, max_num_features=30, font_scale=0.7):
        """
        visualize the feature importance for xgboost regressor
        """
        feat_name, feat_imp = zip(*model.get_fscore().items())
        feat_imp = pd.DataFrame({'Feature': feat_name, 'Weight': feat_imp})

        # plot the importance
        fig, ax = plt.subplots(figsize=(12, 4))
        top_data = feat_imp.sort_values(by="Weight", ascending=False)[0:max_num_features]
        top_feat_name = top_data['Feature'].values
        sns.barplot(x="Weight",
                    y="Feature",
                    data=top_data)
        ax.set_yticklabels(labels=top_feat_name, fontproperties=FONT)
        plt.title('xgb Features with predict target %s' % target)
        plt.savefig(os.path.join(conf.figs_dir, 'xgb_feature_importance_%s.png' % target))
        plt.show()

    @overrides(Model)
    def grid_search_optimization(self, *args, **kwargs):
        raise NotImplementedError

    @overrides(Model)
    def random_search_optimization(self, *args, **kwargs):
        raise NotImplementedError
