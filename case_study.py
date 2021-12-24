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

def k_fold(data, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=random.randint(1,999))
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(data.x.shape[0]), data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(data.x.shape[0], dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    return train_indices, test_indices, val_indices

def predict_protein(args,test_data):
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

    path = osp.join('result',f'{auc:.4f}_model_test_dataset_score.pt')

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

def main2(score_name, log: bool = True):
    path = osp.join('result',score_name)
    if osp.exists(path):  # pragma: no cover
        if log:
            print('测试集得分矩阵已经计算好，无需计算！')
            test_node_adj = torch.load( path).T
            try:
                res = test_node_adj.loc[args.protein_name]
                res = res.sort_values(ascending=False)  # 从大到小排序
                print("======================================")
                print("与" + args.protein_name + "相互作用Top20蛋白质：")
                print("protein        score")
                print(res[:200])
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

    parse.add_argument('--protein_name', type=str)  # 查找蛋白在测试集上的相互作用蛋白

    parse.add_argument('--seq_names', type=str, help='chose transform squence protein description',
                        default="all-MiniLM-L6-v2",
                        choices=['all-MiniLM-L6-v2', 'roberta-large-nli-stsb-mean-tokens',
                                 'bert-base-nli-mean-tokens', 'one-hot'])
    return parse.parse_args()

def cross_validation_with_val_set(args):
    dataset = ARAPPI(root='./Data/ara-protein', seq_name=args.seq_names)

    protein_mapping = dataset.protein_mapping
    data = dataset[0]
    del data.train_mask, data.val_mask, data.test_mask

    transform = T.Compose([
        # T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    train_losses, val_aucs, test_aucs, durations = [], [], [], []
    folds = args.folds

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(data, folds))):

        print(f"{fold+1} fold train")

        split = {
            'train_idx': np.array(train_idx),
            'val_idx': np.array(val_idx),
            'test_idx': np.array(test_idx)}
        allmask = {}
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            allmask[f'{name}_mask'] = mask

        data.train_mask = allmask['train_mask']
        data.val_mask = allmask['val_mask']
        data.test_mask = allmask['test_mask']

        train_data, val_data, test_data = transform(data)  # Explicitly transform data.

        torch.save(test_data,f"./save_model/{fold+1}_fold_test_data.pt")

def reset_feature():

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'PMESP':
        model = GAE(PMESPEncoder(out_channels = args.out_channels,
                                 num_layers = args.lstm_layers,
                                 lstm_hidden = args.lstm_hidden
                                 )).to(device)

    model.load_state_dict(torch.load(f"save_model/196.169951_AUC_0.9684.pt")) # 选择训练好的模型
    model.eval()

    test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt")
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

def diff(protein):
#     protein = "ZIP7"
    sos_path = osp.join('result',f'{protein}-sos-预测结果.csv')
    col_path = osp.join('result',f'{protein}-col-预测结果.csv')

    sos_res = pd.read_csv(sos_path)
    col_res = pd.read_csv(col_path)
    df1 = sos_res['Unnamed: 0'].to_list()
    df2 = col_res['Unnamed: 0'].to_list()
    c = [x for x in df1 if x in df2]
    d = [y for y in (df1+df2) if y not in c]
    # print("相同的蛋白", c)
    print("差异蛋白", d)
    path = osp.join('result',f'{protein}-col和sos不同电信号预测结果的差异蛋白.csv')
    pd.DataFrame(d).to_csv(path)
if __name__ == '__main__':
    args = parse_args()
    args.protein_name="ZIP7" # ZIP7
    # TPST GORK AT1G15180 CIB5 IMPA-2 FZL SETH5





    # cross_validation_with_val_set(args)
    # 创建的得分矩阵
    # print("电信号数据是野生拟南芥")
    # test_data2 = reset_feature()
    # predict_protein(args,test_data = test_data2)
    # print("电信号数据是sos1型拟南芥")
    # test_data = torch.load(f"save_model/196.169951_8_fold_test_data.pt")
    # predict_protein(args,test_data = test_data)

    # 根据得分矩阵寻找测试集连接关系
    # main2(score_name = "sos1_0.9615_model_test_dataset_score.pt") # 电信号是sos1
    # main2(score_name = "col_0.9624_model_test_dataset_score.pt") # 电信号是col
    # pes_col()
    diff(protein = "ZIP7") # 寻找输入不同电信号对预测结果的影响

