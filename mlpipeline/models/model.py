# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 29/10/2019

Description:
    Interface for train and validate a model
"""
from abc import abstractmethod


class Model(object):
    """
    Base class for all models training and validation
    """

    _version = 1.0

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def feature_imp_plot(self, *args, **kwargs):
        pass

    @abstractmethod
    def grid_search_optimization(self, *args, **kwargs):
        pass

    @abstractmethod
    def random_search_optimization(self, *args, **kwargs):
        pass

    @abstractmethod
    def bayesian_optimization(self, *args, **kwargs):
        pass
