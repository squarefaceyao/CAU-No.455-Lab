#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CAU-No.455-Lab -> summarydatasets
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-12-11 18:37
@Desc   ：Summary of the datasets. LD, AD,  the link density, the average degree
=================================================='''
import networkx as nx
import torch_geometric.transforms as T
import torch
from utils import ARAPPI
import pandas as pd
import numpy as np
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    dataset = ARAPPI(root='./Data/ara-protein',seq_name="all-MiniLM-L6-v2")
    # train_data, val_data, test_data = dataset[0]
    # print(train_data)
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is directed: {data.is_directed()}')
    return dataset

def net():
    re = pd.read_csv('./Data/ara-protein/raw/2_train_cau_edge.csv')

    edgs = pd.DataFrame(re, columns=['protein_source', 'protein_target'])

    # 作图使用的代码，节点的名字是基因
    edgs_gene = []
    for row in edgs.itertuples():
        a = [getattr(row, 'protein_source'), getattr(row, 'protein_target')]
        edgs_gene.append(tuple(a))

    G = nx.Graph(edgs_gene)
    # G = nx.function.to_directed(G) # 转换成有向，默认是无向

    print(f"是有向图吗？:{nx.function.is_directed(G)}")
    print(f"边数是{G.number_of_edges()}")
    print(f"节点数是{G.number_of_nodes()}")
    print(f"(LD) Link density:{nx.function.density(G):.4f}")

    # 参考链接https://www.programcreek.com/python/example/89581/networkx.degree
    avg_deg = np.mean(list(dict(nx.degree(G)).values())) # Average Degree is solely decided by k
    print(f"平均链接密度是{avg_deg}")
    print("\n\n")
    re3 = pd.read_csv('./Data/ara-protein/raw/3_train_cau_node.csv')
    des = re3['description']
    name = re3['name']
    for i in range(7):
        print(f"{name[i]}蛋白语意信息的长度为{len(des[i])}")
    print(G)
    return G

if __name__ == '__main__':
    net()
