# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 26/12/2019

Description:
    Class for implementing the Neural Network
"""
import os

import torch
import torch.nn as nn
import numpy as np

from .model import Model
from .lstm import Lstm
from .tcn import TemporalConvNet
from .lstm_attention import LstmAttention
from utils.utils import overrides, get_dataloader
import conf


# class NeuralNetDataset(Dataset):
#     def __init__(self, features, labels, seq_len):
#         self.features = torch.from_numpy(np.asarray(features).astype('float32').reshape((len(features), seq_len, -1)))
#         self.labels = torch.from_numpy(np.asarray(labels).astype('float32').reshape(-1, 1))
#
#     def __getitem__(self, index):
#         return self.features[index], self.labels[index]
#
#     def __len__(self):
#         return self.features.size(0)


class TcnModel(Model):
    def __init__(self, X_train, y_train, valid_fold, train_params, model_params, logger, model):
        """
                :param X_train: nd-array - (total_data, seq_len, input_size)
                :param y_train: nd-array - (total_data, input_size)
                :param valid_fold: tuple - (valid_x, valid_y), each has same shape as train data
        """
        _, self.seq_len, self.input_size = X_train.shape
        self.train_params = train_params
        self.model_params = model_params
        self.train_loader = get_dataloader(X_train.reshape(-1, self.input_size, self.seq_len),
                                           y_train,
                                           batch_size=train_params['batch_size'],
                                           shuffle=train_params['shuffle'])

        val_x, val_y = valid_fold
        self.val_loader = get_dataloader(val_x.reshape(-1, self.input_size, self.seq_len),
                                         val_y,
                                         batch_size=train_params['batch_size'],
                                         shuffle=train_params['shuffle'])
        self.model = model
        self.early_stopping = EarlyStopping(model,
                                            train_params['patience'],
                                            train_params['verbose'],
                                            train_params['delta'],
                                            logger=logger,
                                            )

        self.logger = logger

    @overrides(Model)
    def bayesian_optimization(self, init_points=5, n_iter=5, acq='ei'):
        raise NotImplementedError

    @overrides(Model)
    def train(self):
        model = TemporalConvNet(self.input_size,
                                self.model_params['num_channels'],
                                self.model_params['kernel_size'],
                                self.model_params['dropout'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params['lr'])

        train_losses = []
        valid_losses = []
        n_epochs = self.train_params['epoch']
        for epoch in range(n_epochs):
            model.train()
            for batch, data in enumerate(self.train_loader):
                optimizer.zero_grad()
                features, labels = data
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            for batch, data in enumerate(self.val_loader):
                features, labels = data
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            epoch_len = len(str(self.train_params['epoch']))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            self.logger.info(print_msg)

            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(valid_loss, model)

            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(os.path.join(conf.output_dir, '%s_checkpoint.pt' % self.model)))

        return model

    @overrides(Model)
    def feature_imp_plot(self, model, target, max_num_features=30, font_scale=0.7):
        """
        visualize the feature importance for lightgbm regressor
        """
        raise NotImplementedError

    @overrides(Model)
    def grid_search_optimization(self, *args, **kwargs):
        raise NotImplementedError

    @overrides(Model)
    def random_search_optimization(self, *args, **kwargs):
        raise NotImplementedError


class LstmModel(Model):

    def __init__(self, X_train, y_train, valid_fold, train_params, model_params, logger, model):
        """
        :param X_train: nd-array - (total_data, seq_len, input_size)
        :param y_train: nd-array - (total_data, input_size)
        :param valid_fold: tuple - (valid_x, valid_y), each has same shape as train data
        """
        _, self.seq_len, self.input_size = X_train.shape
        self.train_params = train_params
        self.model_params = model_params
        self.train_loader = get_dataloader(X_train,
                                           y_train,
                                           batch_size=train_params['batch_size'],
                                           shuffle=train_params['shuffle'])

        val_x, val_y = valid_fold
        self.val_loader = get_dataloader(val_x,
                                         val_y,
                                         batch_size=train_params['batch_size'],
                                         shuffle=train_params['shuffle'])

        self.model = model
        self.early_stopping = EarlyStopping(model,
                                            train_params['patience'],
                                            train_params['verbose'],
                                            train_params['delta'],
                                            logger=logger)

        self.logger = logger

    @overrides(Model)
    def bayesian_optimization(self, init_points=5, n_iter=5, acq='ei'):
        raise NotImplementedError

    @overrides(Model)
    def train(self):
        model = Lstm(self.input_size, self.model_params['hidden_size'], self.model_params['num_layers'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params['lr'])

        train_losses = []
        valid_losses = []
        n_epochs = self.train_params['epoch']
        for epoch in range(n_epochs):
            model.train()
            for batch, data in enumerate(self.train_loader):
                optimizer.zero_grad()
                features, labels = data
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            for batch, data in enumerate(self.val_loader):
                features, labels = data
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            epoch_len = len(str(self.train_params['epoch']))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            self.logger.info(print_msg)

            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(valid_loss, model)

            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(os.path.join(conf.output_dir, '%s_checkpoint.pt' % self.model)))

        return model

    @overrides(Model)
    def feature_imp_plot(self, model, target, max_num_features=30, font_scale=0.7):
        """
        visualize the feature importance for lightgbm regressor
        """
        raise NotImplementedError

    @overrides(Model)
    def grid_search_optimization(self, *args, **kwargs):
        raise NotImplementedError

    @overrides(Model)
    def random_search_optimization(self, *args, **kwargs):
        raise NotImplementedError


class LstmAttentionModel(Model):

    def __init__(self, X_train, y_train, valid_fold, train_params, model_params, logger, model):
        """
        :param X_train: nd-array - (total_data, seq_len, input_size)
        :param y_train: nd-array - (total_data, input_size)
        :param valid_fold: tuple - (valid_x, valid_y), each has same shape as train data
        """
        _, self.seq_len, self.input_size = X_train.shape
        self.train_params = train_params
        self.model_params = model_params
        self.train_loader = get_dataloader(X_train,
                                           y_train,
                                           batch_size=train_params['batch_size'],
                                           shuffle=train_params['shuffle'])

        val_x, val_y = valid_fold
        self.val_loader = get_dataloader(val_x,
                                         val_y,
                                         batch_size=train_params['batch_size'],
                                         shuffle=train_params['shuffle'])
        self.model = model
        self.early_stopping = EarlyStopping(model,
                                            train_params['patience'],
                                            train_params['verbose'],
                                            train_params['delta'],
                                            logger=logger)

        self.logger = logger

    @overrides(Model)
    def bayesian_optimization(self, init_points=5, n_iter=5, acq='ei'):
        raise NotImplementedError

    @overrides(Model)
    def train(self):
        model = LstmAttention(self.input_size,
                              self.model_params['hidden_size'],
                              self.model_params['num_layers'],
                              self.model_params['dropout'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_params['lr'])

        train_losses = []
        valid_losses = []
        n_epochs = self.train_params['epoch']
        for epoch in range(n_epochs):
            model.train()
            for batch, data in enumerate(self.train_loader):
                optimizer.zero_grad()
                features, labels = data
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            for batch, data in enumerate(self.val_loader):
                features, labels = data
                outputs = model(features)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            epoch_len = len(str(self.train_params['epoch']))

            print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')

            self.logger.info(print_msg)

            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(valid_loss, model)

            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(os.path.join(conf.output_dir, '%s_checkpoint.pt' % self.model)))

        return model

    @overrides(Model)
    def feature_imp_plot(self, model, target, max_num_features=30, font_scale=0.7):
        """
        visualize the feature importance for lightgbm regressor
        """
        raise NotImplementedError

    @overrides(Model)
    def grid_search_optimization(self, *args, **kwargs):
        raise NotImplementedError

    @overrides(Model)
    def random_search_optimization(self, *args, **kwargs):
        raise NotImplementedError


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience.
       cited from https://github.com/Bjarten/early-stopping-pytorch"""

    def __init__(self, model, patience=7, verbose=False, delta=0, logger=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger
        self.model = model

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(conf.output_dir, '%s_checkpoint.pt' % self.model))
        self.val_loss_min = val_loss
