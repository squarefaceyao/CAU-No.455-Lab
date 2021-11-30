#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-10-07 23:39
@Desc   ：用到的模型
=================================================='''
from torch_geometric.nn import GCNConv
import torch
import numpy as np

class PMESPEncoder(torch.nn.Module):
    def __init__(self, out_channels, num_layers, lstm_hidden):
        super().__init__()
        self.num_layers = num_layers
        self.lstm_hidden = lstm_hidden
        self.lstm = torch.nn.LSTM(588, self.lstm_hidden, self.num_layers, bidirectional=True)
        self.h0 = torch.randn(self.num_layers * 2, 3600, self.lstm_hidden)
        self.c0 = torch.randn(self.num_layers * 2, 3600, self.lstm_hidden)
        self.m = torch.nn.Dropout(p=0.1)
        self.conv1_t = GCNConv(384 + self.lstm_hidden  * 2, 2 * out_channels)
        self.conv2_t = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        pes_feature = x[:,:588]   # ([3600, 588])
        protein_feature = x[:,588:]  #([3600, 343])
        x_p, _ = self.lstm(pes_feature.unsqueeze(dim=0), (self.h0, self.c0))
        x_p = x_p.squeeze(dim=0) /100
        x1_xp = torch.cat([protein_feature,x_p],dim=1)
        x1_xp = self.m(x1_xp)
        x1 = self.conv1_t(x1_xp, edge_index).relu()
        out = self.conv2_t(x1, edge_index)
        return out

class GcnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels * 2)
        self.conv2 = GCNConv(out_channels * 2, out_channels)
    def forward(self, x, edge_index):
        # x = x[:, 588:] # 只用蛋白质序列信息 384
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class proteinEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, out_channels * 2)
        self.conv2 = GCNConv(out_channels * 2, out_channels)
    def forward(self, x, edge_index):
        protein_feature = x[:, 588:]
        x = self.conv1(protein_feature, edge_index).relu()
        return self.conv2(x, edge_index)

class GaeNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(384, out_channels * 2)
        self.conv2 = GCNConv(out_channels * 2, out_channels)

    def encode(self, x, edge_index):
        x = x[:, 588:] # 只用蛋白质序列信息 384
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def decode(self, z, edge_label_index):
        # 可以添加激活函数
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0.6).nonzero(as_tuple=False).t()
