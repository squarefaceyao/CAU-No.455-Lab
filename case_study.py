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

def predict_protein(args):
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

def main2(log: bool = True):
    path = osp.join('result','test_dataset_score.pt')
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
                print(res[:20])
                res[:20].to_csv(f"./result/{args.protein_name}-预测结果.csv")
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


if __name__ == '__main__':
    args = parse_args()
    args.protein_name="SPA1"
    # cross_validation_with_val_set(args)
    predict_protein(args)
