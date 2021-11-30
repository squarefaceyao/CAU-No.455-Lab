#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-11-05 07:44
@Desc   ：
=================================================='''
# coding=UTF-8
import argparse
import shutil
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import time
import os
from utils import Initialize,AA,Calculation_AUC,Jaccavrd, RWR,Cn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os.path as osp
import pandas as pd
import torch

def create_adj():
    current_dir = osp.dirname(osp.abspath(__file__)) # 获取当前文件所在路径
    precess_datasets_save_path = osp.join(current_dir, "Data", "ara-protein", "processed") # 边文件保存的路径
    row_datasets_path = osp.join(current_dir, "Data", "ara-protein", "raw", "2_train_cau_node.csv") #原始文件
    df1 = pd.read_csv(row_datasets_path, index_col='name')
    mapping = {index: i for i, index in enumerate(df1.index.unique())}

    row_datasets_path_edg = osp.join(current_dir, "Data", "ara-protein", "raw", "1_train_cau_edge.csv") # 原始文件
    src_index_col = 'protein_source'
    src_mapping = mapping
    dst_index_col = 'protein_target'
    dst_mapping = mapping
    df = pd.read_csv(row_datasets_path_edg)
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    a = np.array(edge_index.T)
    pd.DataFrame(a).to_csv(osp.join(precess_datasets_save_path, 'cau-ara-protein.txt'),sep='\t', header=None, index=False)  # 保存在pressed/文件里

def main(args):

    time_now = time.localtime()
    time_f = time.strftime('%Y/%m/%d %H:%M:%S', time_now)
    with open(f'./result/链接预测机器学习-AUC评价指标.txt', 'a+') as f:
        f.write('\n')
        f.write('\n')
        f.write('实验时间 = {}'.format(time_f) + '\n')
        f.write('保存每次实验的评价指标，并计算均值和标准差' + '\n')
        f.write('AA' + '\t' 'Jaccavrd' + '\t' + 'RWR' + '\t' + 'Cn' + '\n')

    current_dir = osp.dirname(osp.abspath(__file__))  # 当前文件所在路径
    precess_datasets_save_path = osp.join(current_dir, "Data", "ara-protein", "processed")  # 获取链接矩阵坐在文件夹

    adj_path = osp.join(precess_datasets_save_path, "cau-ara-protein.txt")
    if osp.exists(adj_path):
        print("已经存在链接文件")
    else:
        create_adj()
    aas, jacs, rwrs, cns = [], [], [], []
    for _ in range(args.rept):
        print(f"重复实验第{_ + 1}次")
        startTime = time.perf_counter()
        current_dir = osp.dirname(osp.abspath(__file__))  # 当前文件坐在路径
        precess_datasets_save_path = osp.join(current_dir, "Data", "ara-protein", "processed")  # 获取处理后文件夹
        File = osp.join(precess_datasets_save_path, "cau-ara-protein.txt")  # 边文件所载路径
        pathName = 'ara-protein'
        TrainFile_Path = osp.join(current_dir, 'Data', pathName, 'split', 'Train.txt')  # 拆分数据集路径
        # TrainFile_Path = 'Data/'+NetName+'/Train.txt'
        if os.path.exists(TrainFile_Path):  # 训练集是否存在
            Train_File = osp.join(current_dir, 'Data', pathName, 'split', 'Train.txt')
            Test_File = osp.join(current_dir, 'Data', pathName, 'split', 'Test.txt')
            MatrixAdjacency_Train, MatrixAdjacency_Test, MaxNodeNum = Initialize.Init2(Test_File, Train_File)
        else:
            print("正在划分Train, Test")
            MatrixAdjacency_Net, MaxNodeNum = Initialize.Init(File)
            MatrixAdjacency_Train, MatrixAdjacency_Test = Initialize.Divide(File, MatrixAdjacency_Net, MaxNodeNum)
        # 先计算相似性矩阵，在计算测试机的auc
        aa_auc = Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test,
                                 Matrix_similarity=AA(MatrixAdjacency_Train), MaxNodeNum=MaxNodeNum)
        jac_auc = Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test,
                                  Matrix_similarity=Jaccavrd(MatrixAdjacency_Train), MaxNodeNum=MaxNodeNum)
        rwr_auc = Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test,
                                  Matrix_similarity=RWR(MatrixAdjacency_Train), MaxNodeNum=MaxNodeNum)
        cn_auc = Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test,
                                 Matrix_similarity=Cn(MatrixAdjacency_Train), MaxNodeNum=MaxNodeNum)
        with open(f'./result/链接预测机器学习-AUC评价指标.txt', 'a+') as f:
            f.write(f"{aa_auc:.4f}\t{jac_auc:.4f}\t{rwr_auc:.4f}\t{cn_auc:.4f}\n")

        aas.append(aa_auc)
        jacs.append(jac_auc)
        rwrs.append(rwr_auc)
        cns.append(cn_auc)

        print(f"删掉文件夹{osp.join(current_dir, 'Data', pathName, 'split')}")
        # shutil.rmtree(osp.join(current_dir, 'Data', pathName, 'split'))

    with open(f'./result/链接预测机器学习-AUC评价指标.txt', 'a+') as f:
        f.write(f'进行了{args.rept}次实验,其最大值，均值，方差如下：' + '\n')
        f.write(f'最大值：' + '\n')
        f.write(f'均值：' + '\n')
        f.write(f'标准差：' + '\n')
        f.write(f"{np.max(aas):.4f}\t{np.max(jacs):.4f}\t{np.max(rwrs):.4f}\t{np.max(cns):.4f}\n")
        f.write(f"{np.mean(aas):.4f}\t{np.mean(jacs):.4f}\t{np.mean(rwrs):.4f}\t{np.mean(cns):.4f}\n")
        f.write(f"{np.std(aas):.4f}\t{np.std(jacs):.4f}\t{np.std(rwrs):.4f}\t{np.std(cns):.4f}\n")


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--rept', type=int, help='重复次数',default=1)
    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.rept = 1
    main(args)
