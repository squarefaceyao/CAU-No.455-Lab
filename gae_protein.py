#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30/11/2021 22:53
# @Author  : Mr. Y
# @Site    : 只使用蛋白信息
# @File    : gae_protein.py
# @Software: PyCharm

from utils import ARAPPI,plot_history
from torch_geometric.utils import negative_sampling, train_test_split_edges
from utils import GaeNet
import torch_geometric.transforms as T
import torch
import numpy as np
import argparse
import copy
import pandas as pd
from sklearn.metrics import precision_score, recall_score,accuracy_score,f1_score
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve
import time
from sklearn.metrics import auc as auc2

def main(args):
    device = 'cpu'
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    dataset = ARAPPI(root='./Data/ara-protein', pre_transform=transform, seq_name=args.seq_names)
    protein_mapping = dataset.protein_mapping
    train_data, val_data, test_data = dataset[0]

    model = GaeNet(dataset.num_features, hidden_channels=args.hidden_channels,
                                         out_channels=args.out_channels).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)  # 训练优化器
    criterion = torch.nn.BCEWithLogitsLoss()  # 交叉商损失函数
    print(model)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def pytest(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    best_val_auc = final_test_auc = 0
    val_aucs, test_aucs = [], []
    train_loss = []
    n_epochs = args.epochs
    for epoch in range(1, n_epochs + 1):
        loss = train()
        val_auc = pytest(val_data)
        test_auc = pytest(test_data)
        val_aucs.append(val_auc.item())
        train_loss.append(loss.item())
        test_aucs.append(test_auc.item())
        if val_auc > best_val_auc:
            best_val = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')
    plot_history(val_aucs, test_aucs, train_loss)

    def perform():  # Performance evaluation
        result = {}
        model.eval()
        z = model.encode(test_data.x, test_data.edge_index)
        out = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
        y_true2 = test_data.edge_label.cpu().numpy()
        y_proba = out.cpu().detach().numpy()
        fpr, tpr, threshold = roc_curve(y_true2, y_proba)
        # save result
        pre, rec, _ = precision_recall_curve(y_true2, y_proba)
        AUC = auc2(fpr, tpr)
        AUPR = auc2(rec, pre)
        print(f"use {args.model} model")
        print(f'AUC = {AUC:.4f}' )
        print(f"AUPR = {AUPR:.4f} ")
        # 保存结果，用于后续绘图
        result['AUC'] = AUC,
        result['AUPR'] = AUPR,
        result['fpr'] = fpr
        result['tpr'] = tpr,
        result['pre'] = pre,
        result['rec'] = rec,

        out = model.decode(z, test_data.edge_label_index).view(-1).sigmoid()
        y_pred = out.cpu().detach().numpy()
        y_true = test_data.edge_label.cpu().tolist()

        y_pred_roc = copy.copy(y_pred)  # 复制预测得分
        y_pred[y_pred > 0.8] = 1
        y_pred[y_pred <= 0.8] = 0
        print('Precision: %.3f' % precision_score(y_true, y_pred))
        print('Recall: %.3f' % recall_score(y_true, y_pred))
        print('F1: %.3f' % f1_score(y_true, y_pred))
        print('accuracy: %.3f' % accuracy_score(y_true, y_pred))

        print('AUC: %.3f' % roc_auc_score(y_true, y_pred_roc))
        print('AP: %.3f' % average_precision_score(y_true, y_pred_roc))

        pre, rec, _ = precision_recall_curve(y_true, y_pred_roc)
        fpr, tpr, threshold = roc_curve(y_true, y_pred_roc)

        aupr = auc2(rec, pre)
        auc = auc2(fpr, tpr)

        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        ap = average_precision_score(y_true, y_pred_roc)

        with open(f'./result/{args.model}-评价指标-{args.hidden_channels}.txt', 'a+') as f:
            f.write(f"{auc:.4f}\t{aupr:.4f}\t{ap:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{acc:.4f}\n")

        torch.save(result,f"./result/{args.model}-auc-{AUC:.4f}-{args.hidden_channels}.result.pt")
        return auc, aupr, ap, prec, rec, f1, acc

    auc, aupr, ap, prec, rec, f1, acc = perform()
    torch.save(model.state_dict(),f"./result/{args.model}-auc-{auc:.4f}.pt")
    return auc, aupr, ap, prec, rec, f1, acc

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epochs', type=int, help='train epochs',default=10000)
    parse.add_argument('--hidden_channels', type=int, help='hidden_channels',default=50)
    parse.add_argument('--out_channels', type=int, help='out channels',default=20)
    parse.add_argument('--lr', type=int, help='learning rate',default=0.01)
    parse.add_argument('--rept', type=int, help='rept expriment',default=1)
    parse.add_argument('--lstm_layers', type=int, help='early stop wait num',default=14)
    parse.add_argument('--lstm_hidden', type=int, help='early stop wait num',default=30)
    parse.add_argument('--model', type=str, help='chose mode', default="GaeNet")
    parse.add_argument('--seq_names', type=str, help='chose transform squence protein description',
                                  default="all-MiniLM-L6-v2",
                                    choices=['all-MiniLM-L6-v2','roberta-large-nli-stsb-mean-tokens','one-hot'])
    return parse.parse_args()

if __name__ == '__main__':
    import os.path as osp
    import os
    import shutil

    # current_dir = osp.dirname(osp.abspath(__file__))
    # precess_datasets_save_path = osp.join(current_dir, "Data", "ara-protein", "processed")
    # shutil.rmtree(precess_datasets_save_path)

    args = parse_args()
    args.rept = 2
    args.seq_names = "all-MiniLM-L6-v2"
    args.hidden_channels = 128

    args.patience = 200
    args.lstm_layers = 14
    args.lstm_hidden = 27
    args.out_channels = 16
    args.lr = 0.03
    args.epochs = 900

    if args.rept == 1:
        main(args)

    if args.rept != 1:
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 设置图片大小，字体大小
        sns.set_context("talk", font_scale=1)
        sns.set_style("white")
        figure_x = 15
        figure_y = 8.5

        aucs, auprs, aps, precs, recs, f1s, accs = [], [], [], [], [], [], []
        filePath = './result/model'  # 文件夹里存放每次实验的结果
        names = os.listdir(filePath)
        time_now = time.localtime()
        time_f = time.strftime('%Y/%m/%d %H:%M:%S', time_now)
        with open(f'./result/{args.model}-评价指标-{args.hidden_channels}.txt', 'a+') as f:
            f.write('\n')
            f.write('\n')
            f.write('实验时间 = {}'.format(time_f) + '\n')
            f.write(f'实验参数lstm_layers: {args.lstm_layers},lstm_hidden: {args.lstm_hidden },GCN out_channels: {args.out_channels }' + '\n')
            f.write('保存每次实验的评价指标，并计算均值和标准差' + '\n')
            f.write('AUC' + '\t' 'AUPR' + '\t' + 'AP' + '\t' + 'Precision' + '\t' + 'Recall' + '\t' + 'F1' + '\t' + 'Accuracy' + '\n')
    # print(args)
        current_dir = osp.dirname(osp.abspath(__file__))
        precess_datasets_save_path = osp.join(current_dir, "Data", "ara-protein", "processed")
        for _ in range(args.rept):
            # 重复实验
            auc, aupr, ap, prec, rec, f1, acc = main(args)
            aucs.append(auc)
            auprs.append(aupr)
            aps.append(ap)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            accs.append(acc)
            print(f'删掉文件夹{precess_datasets_save_path}')
            shutil.rmtree(precess_datasets_save_path)

        with open(f'./result/{args.model}-评价指标-{args.hidden_channels}.txt', 'a+') as f:
            f.write(f'这{len(aucs)}个模型的平均评价指标是' + '\n')
            f.write(
                'AUC' + '\t' + 'AUPR' + '\t' + 'AP' + '\t' + 'Precision' + '\t' + 'Recall' + '\t' + 'F1' + '\t' + 'Accuracy' + '\n')
            f.write(f'最大值：' + '\n')
            f.write(f'均值：' + '\n')
            f.write(f'标准差：' + '\n')
            f.write(
                f"{np.max(aucs):.4f}\t{np.max(auprs):.4f}\t{np.max(aps):.4f}\t{np.max(precs):.4f}\t{np.max(recs):.4f}\t{np.max(f1s):.4f}\t{np.max(accs):.4f}\n")
            f.write(
                f"{np.mean(aucs):.4f}\t{np.mean(auprs):.4f}\t{np.mean(aps):.4f}\t{np.mean(precs):.4f}\t{np.mean(recs):.4f}\t{np.mean(f1s):.4f}\t{np.mean(accs):.4f}\n")
            f.write(
                f"{np.std(aucs):.4f}\t{np.std(auprs):.4f}\t{np.std(aps):.4f}\t{np.std(precs):.4f}\t{np.std(recs):.4f}\t{np.std(f1s):.4f}\t{np.std(accs):.4f}\n")

        data = {
            "AUC": aucs,
            "AUPR": auprs,
            "Precision": precs,
            "Recall": recs,
            "F1": f1s,
            "Accuracy": accs
        }

        plt.figure(figsize=(figure_x, figure_y))

        df = pd.DataFrame(data)
        df.plot.box(title=" ")
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./Figure/{args.model}_{args.rept}次评价指标', dpi=300)
