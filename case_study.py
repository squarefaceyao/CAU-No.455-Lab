#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CAU-No.455-Lab -> case_study
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-12-17 09:03
@Desc   ：
=================================================='''
import numpy as np
import os.path as osp
import pandas as pd
import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GAE
from utils import ARAPPI
from utils import PMESPEncoder
from sklearn.model_selection import StratifiedKFold
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_protein(args,test_data,variety):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'PMESP':
        model = GAE(PMESPEncoder(out_channels = args.out_channels,
                                 num_layers = args.lstm_layers,
                                 lstm_hidden = args.lstm_hidden
                                 )).to(device)

    model.load_state_dict(torch.load(f"save_model/196.169951_AUC_0.9684.pt")) # 选择训练好的模型
    model.eval()

    # test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt")
    dataset = ARAPPI(root='./Data/ara-protein', seq_name=args.seq_names)

    protein_mapping = dataset.protein_mapping

    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        return z,auc, ap

    z,auc, ap = test(test_data)
    score = z @ z.t()  # 边链接的评价矩阵
    print(score.shape)
    print(auc)

    test_node = test_data.test_mask.nonzero(as_tuple=False) # 获取test_node的节点
    id2protein = {v: k for k, v in protein_mapping.items()} # 转换protein_mapping. id:protein

    test_score = score[test_node].squeeze(dim=1)  # 保存在测试集上的打分矩阵
    columns = [id2protein[int(i)] for i in test_node]  # 设置测试集节点的名字

    test_node_adj = pd.DataFrame(test_score.detach().numpy().T,
                                  columns=columns,
                                  index=list(protein_mapping.keys())
                                  ) # 保存结果

    path = osp.join('result',f'{variety}_{auc:.4f}_model_test_dataset_score.pt')

    torch.save(test_node_adj.round(4), path)

    try:
        res = test_node_adj.loc[args.protein_name]
        res = res.sort_values(ascending=False) # 从大到小排序
        print("======================================")
        print("与" + args.protein_name + "相互作用Top20蛋白质：")
        print("protein        score")
        print(res[:20])
    except:
        print(f'{args.protein_name}不在数据集里面')

    return path

def main2(score_name, log: bool = True):
    path = osp.join('result',score_name)
    if osp.exists(path):  # pragma: no cover
        if log:
            print('测试集得分矩阵已经计算好，无需计算！')
            test_node_adj = torch.load(path).T
            try:
                res = test_node_adj.loc[args.protein_name]
                res = res.sort_values(ascending=False)  # 从大到小排序
                print("======================================")
                print(f"用的拟南芥品种是{score_name[0:3]}")
                print("与" + args.protein_name + "相互作用Top20蛋白质：")
                print("protein        score")
                print(res[:8])
                res[:200].to_csv(f"./result/{args.protein_name}-{score_name[0:3]}-预测结果.csv")
                return res
            except:
                print(f'{args.protein_name}不在数据集里面')

    else:
        print('正在计算测试集的得分矩阵...')
        predict_protein(args)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model',  type=str, help='chose mode', default='PMESP',
                       choices=['proteinEncoder', # 输入数据只有蛋白序列信息
                                'GcnEncoder' # 输入数据包含蛋白数据和电信号数据
                                ])
    parse.add_argument('--epochs', type=int, default=300)
    parse.add_argument('--lr', type=int, default=0.01)
    parse.add_argument('--folds', type=int, default=10)

    parse.add_argument('--lstm_layers', type=int, default=3) # 性能最优
    parse.add_argument('--lstm_hidden', type=int, default=7) # 性能最优 35
    parse.add_argument('--out_channels', type=int, default=16)  # 性能最优 10

    parse.add_argument('--protein_name', type=str, default="GORK")  # 查找蛋白在测试集上的相互作用蛋白

    parse.add_argument('--seq_names', type=str, help='chose transform squence protein description',
                        default="all-MiniLM-L6-v2",
                        choices=['all-MiniLM-L6-v2', 'roberta-large-nli-stsb-mean-tokens',
                                 'bert-base-nli-mean-tokens', 'one-hot'])
    return parse.parse_args()

def reset_feature():
    '''
    重置电信号数据，删除原有数据。
    '''

    pes_col_path = osp.join("Data", 'Colombia-100mM.csv')

    df = pd.read_csv(pes_col_path, header=None)
    my_array = np.array(df)
    my_tensor = torch.tensor(my_array, dtype=torch.float)
    my_tensor = my_tensor.unsqueeze(0)
    nosalt = my_tensor[:, :588, :]
    salt = my_tensor[:, 588:, :]

    test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt")
    # salt_protein 是节点特征，前面是电信号（588），后面是蛋白特征（384）
    salt_protein = test_data.x[test_data.y == 0][:, 588:]
    pes_protein = test_data.x[test_data.y == 1][:, 588:]

    def cat_pes_protein(salt, salt_protein):
        salt = torch.squeeze(salt, dim=0)  # 去掉一维卷积的一个维度
        salt = salt.T.repeat(200, 1)  # 重复向量
        salt = salt[:salt_protein.shape[0]]  # 电信号维度等于蛋白质维度。这里可以用滑动窗口实现。
        salt_protein2 = torch.cat([salt, salt_protein], dim=-1)  # 和蛋白序列信息合并
        return salt_protein2

    salt_protein2 = cat_pes_protein(salt, salt_protein)
    pes_protein2 = cat_pes_protein(nosalt, pes_protein)
    protein_and_pes = torch.cat([salt_protein2, pes_protein2], dim=0)

    del test_data.x  # 删掉原先的数据

    test_data.x = protein_and_pes  # 设置新的特征
    return test_data


def pes_col():
    """
        使用训练好的模型预测测试集节点连接分数。
        reset_feature() 功能是重置电信号数据
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'PMESP':
        model = GAE(PMESPEncoder(out_channels = args.out_channels,
                                 num_layers = args.lstm_layers,
                                 lstm_hidden = args.lstm_hidden
                                 )).to(device)

    model.load_state_dict(torch.load(f"save_model/196.169951_AUC_0.9684.pt")) # 选择训练好的模型
    model.eval()

    test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt") # 选择模型对应的测试机
    dataset = ARAPPI(root='./Data/ara-protein', seq_name=args.seq_names)

    protein_mapping = dataset.protein_mapping

    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        return z,auc, ap

    aaa = test_data.x[test_data.test_mask]
    z,auc, ap = test(test_data)
    print(auc)

    test_data2 = reset_feature()
    z2,auc2, ap2 = test(test_data2)
    print(auc2)

def diff(args):
#     protein = "ZIP7"
    protein = args.protein_name
    sos_path = osp.join('result',f'{protein}-sos-预测结果.csv')
    col_path = osp.join('result',f'{protein}-col-预测结果.csv')

    sos_res = pd.read_csv(sos_path)
    col_res = pd.read_csv(col_path)
    df1 = sos_res['Unnamed: 0'].to_list()
    df2 = col_res['Unnamed: 0'].to_list()
    c = [x for x in df1 if x in df2]
    d = [y for y in (df1+df2) if y not in c]
    # print("相同的蛋白", c)
    print(f"{protein}在col和sos不同电信号预测结果的差异蛋白的个数为{len(d)},结果保存在result文件夹 \n",)
    path = osp.join('result',f'{protein}-col和sos不同电信号预测结果的差异蛋白.csv')
    pd.DataFrame(d).to_csv(path)
if __name__ == '__main__':
    """
    # 1. 使用训练好的模型计算盐敏感电信号和耐盐性电信号在测试集的链接得分
        print("电信号数据是野生拟南芥")
        test_data2 = reset_feature() # 电信号数据修改为野生型拟南芥
        predict_protein(args,test_data = test_data2,variety="col")
        print("电信号数据是sos1型拟南芥")
        test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt") # 使用训练模型时的数据
        predict_protein(args,test_data = test_data,variety="sos1")
    # 2. 加载不同模型计算测试集链接得分，并计算一个测试集蛋白的链接得分，结果保存在result文件夹
        args = parse_args()
        args.protein_name="GORK"
        main2(score_name = "sos1_0.9631_model_test_dataset_score.pt") # 电信号是sos1
        main2(score_name = "col_0.9621_model_test_dataset_score.pt") # 电信号是col
        # 2.1. 计算不同电信号预测结果的差异蛋白
        diff(args)
    """
    args = parse_args()
    # 1. 使用训练好的模型计算盐敏感电信号和耐盐性电信号在测试集的链接得分
    # print("电信号数据是野生拟南芥")
    # test_data2 = reset_feature() # 电信号数据修改为野生型拟南芥
    # col_score_path = predict_protein(args=args,test_data = test_data2,variety="col")
    # print("电信号数据是sos1型拟南芥")
    # test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt") # 使用训练模型时的数据
    # sos1_score_path = predict_protein(args=args,test_data = test_data,variety="sos1")
    for protein in ['GORK','CNGC10','IMPA-2','GLR2.7','CIB5','ZIP7']:
        args.protein_name=protein # ZIP7 # 输入测试集蛋白计算连接性。用main2函数
         # 2. 加载不同模型计算测试集链接得分，并计算一个测试集蛋白的链接得分，结果保存在result文件夹
        main2(score_name = "sos1_0.9615_model_test_dataset_score.pt") # 电信号是sos1
        main2(score_name = "col_0.9624_model_test_dataset_score.pt") # 电信号是col
        # pes_col()
        diff(args)

