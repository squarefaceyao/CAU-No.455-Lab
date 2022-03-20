#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30/11/2021 23:43
# @Author  : Mr. Y
# @Site    : 
# @File    : evaluation.py
# @Software: PyCharm

import matplotlib.pyplot as plt


def plot_roc(fpr, tpr, roc_auc,file_name):
    fig = plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc)
    if file_name == 'AUC curve':
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
    if file_name == 'AUPR curve':
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('precision')
        plt.title('P-R Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    fig.savefig(f'./Figure/{file_name}',dpi=300)
    # plt.close()

def plot_history(val_aucs,test_aucs,train_loss):

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(val_aucs) + 1), val_aucs, label='Validation AUC')
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Validation loss')

    # find position of lowest validation loss
    minposs = train_loss.index(min(train_loss)) + 1
    maxposs = test_aucs.index(max(test_aucs)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('values')
    # plt.ylim(0.7, 1)  # consistent scale
    # plt.xlim(0, len(val_aucs) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('./Figure/history_plot.png', bbox_inches='tight',dpi=300)
