# -*- coding: utf-8 -*-
"""
Author: MengQiu Wang 
Email: wangmengqiu@ainnovation.com
Date: 03/01/2020

Description:
    Class for implementing the Lstm+Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmAttention(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, device='cpu'):
        super(LstmAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        """
        cal the attention between all hidden state and the last hidden state
        """
        # lstm_hidden - [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        lstm_hidden = lstm_hidden.unsqueeze(1)
        atten_w = self.attention_layer(lstm_hidden)
        m = nn.Tanh()(lstm_out)
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        softmax_w = F.softmax(atten_context, dim=-1)
        context = torch.bmm(softmax_w, lstm_out)
        res = context.squeeze(1)
        return res

    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc(atten_out)
