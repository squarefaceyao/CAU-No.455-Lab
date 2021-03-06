import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, VGAE
from utils import ARAPPI
from utils import PMESPEncoder,GcnEncoder,proteinEncoder,cross_validation_with_val_set
import os.path as osp
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import time


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    dataset = ARAPPI(root='./Data/ara-protein', seq_name=args.seq_names)
    data = dataset[0]
    del data.train_mask, data.val_mask, data.test_mask

    if args.model == 'PMESP':
        model = GAE(PMESPEncoder(out_channels = args.out_channels,
                                 num_layers = args.lstm_layers,
                                 lstm_hidden = args.lstm_hidden
                                 )).to(device)
    if args.model == 'GcnEncoder':
        in_channels = dataset.num_features
        model = GAE(GcnEncoder(in_channels = in_channels,
                               out_channels = args.out_channels
                              )).to(device)

    if args.model == 'proteinEncoder':
        in_channels = dataset.num_features - 588
        model = GAE(proteinEncoder(in_channels = in_channels,
                               out_channels = args.out_channels
                              )).to(device)
    print(model)
    transform = T.Compose([
        # T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])

    loss, auc_mean, std = cross_validation_with_val_set(data=data,
                                  model=model,
                                  args=args,
                                  transform=transform)
    return auc_mean

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model',  type=str, help='chose mode', default='PMESP',
                       choices=['proteinEncoder', # ????????????????????????????????????
                                'GcnEncoder' # ????????????????????????????????????????????????
                                ])
    parse.add_argument('--epochs', type=int, default=300)
    parse.add_argument('--lr', type=int, default=0.01)
    parse.add_argument('--folds', type=int, default=10)

    parse.add_argument('--lstm_layers', type=int, default=3) # ????????????
    parse.add_argument('--lstm_hidden', type=int, default=7) # ???????????? 35
    parse.add_argument('--out_channels', type=int, default=16)  # ???????????? 10

    parse.add_argument('--seq_names', type=str, help='chose transform squence protein description',
                        default="all-MiniLM-L6-v2",
                        choices=['all-MiniLM-L6-v2', 'roberta-large-nli-stsb-mean-tokens',
                                 'bert-base-nli-mean-tokens', 'one-hot'])
    return parse.parse_args()

if __name__ == '__main__':
    '''
    # lstm ???AUC????????? 
    args = parse_args()
    # lrs=[0.05,0.04,0.03,0.02,0.01,0.005,0.004,0.003,0.002,0.001]
    import numpy as np
    _x = np.arange(1, 100, 12)
    _y = np.arange(1, 100, 12) + 7
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = [] # save auc value
    # for ep,lr in zip(range(100,1100,100),lrs):
    #     args.lr = lr
    #     args.epochs = ep
    args.epochs = 300
    # args.model = 'GcnEncoder'
    # args.model = 'proteinEncoder'
    # args.seq_names = 'one-hot'
    # args.lr = 0.01
    print(f'????????????{len(x)}??????\n')
    print(f'lstm_layers????????????{x}\n')
    print(f'lstm_hidden????????????{y}\n')

    for layers,hidden in zip(x,y):
        # print(f"epochs:{args.epochs},lr:{args.lr}")
        args.lstm_layers = layers
        args.lstm_hidden = hidden
        print(f"lstm_layers:{args.lstm_layers},lstm_hidden:{args.lstm_hidden}")
        auc_mean = main(args)
        print("\n\n")
        z.append(auc_mean)

    top = np.array(z)

    localtime = time.asctime(time.localtime(time.time()))
    # plt ????????????
    sns.set_context("talk", font_scale=2.5)
    # setup the figure and axes
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(projection='3d')

    bottom = np.zeros_like(top)
    width = depth = 2
    dz = top
    offset = dz + np.abs(dz.min())
    fracs = offset.astype(float) / offset.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    color_values = cm.jet(norm(fracs.tolist()))

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color=color_values, edgecolor="black", )
    ax1.set_title('Shaded')

    colourMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    colourMap.set_array(top)
    colBar = plt.colorbar(colourMap, shrink=0.5, extend='both').set_label('AUC Value')
    plt.savefig(f'{x}lstm_layer-lstm_num???AUC???????????????',dpi=300)
    plt.show()
    '''
    '''epochs ???AUC?????????
    
    args = parse_args()
    # lrs=[0.05,0.04,0.03,0.02,0.01,0.005,0.004,0.003,0.002,0.001]
    z = []
    for ep in range(100,1000,100):
        print(f"epochs:{args.epochs}")

        args.epochs = ep
        auc_mean = main(args)

        print("\n\n")
        z.append(auc_mean)
    print(z)
    print(list(range(100,1000,100)))
    '''
    '''
    # k-fold ???AUC?????????
    args = parse_args()
    ks=[3,5,10]
    z = []
    for lr in ks:
        args.folds = lr
        print(f"k-fold:{args.folds}")
        auc_mean = main(args)
        print("\n\n")
        z.append(auc_mean)
    print(z)
    print(ks)
    '''
    """
    # lstm  ???GCN hidden???AUC?????????
    args = parse_args()
    # lrs=[0.05,0.04,0.03,0.02,0.01,0.005,0.004,0.003,0.002,0.001]
    import numpy as np
    outs = [8, 16, 24, 32]
    lstms = [7, 14, 21, 28]
    _x = np.array(outs)
    _y = np.array(lstms)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    z = [] # save auc value
    # for ep,lr in zip(range(100,1100,100),lrs):
    #     args.lr = lr
    #     args.epochs = ep
    args.epochs = 300
    # args.model = 'GcnEncoder'
    # args.model = 'proteinEncoder'
    # args.seq_names = 'one-hot'
    # args.lr = 0.01
    print(f'????????????{len(x)}??????\n')
    print(f'GCN out????????????{x}\n')
    print(f'lstm_hidden????????????{y}\n')

    for out,lstm in zip(x,y):
        # print(f"epochs:{args.epochs},lr:{args.lr}")
        args.out_channels = out
        args.lstm_hidden = lstm
        print(f"lstm_layers:{args.lstm_layers},lstm_hidden:{args.lstm_hidden}")
        auc_mean = main(args)
        print("\n\n")
        z.append(auc_mean)

    top = np.array(z)

    localtime = time.asctime(time.localtime(time.time()))
    # plt ????????????
    sns.set_context("talk", font_scale=2.5)
    # setup the figure and axes
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(projection='3d')

    bottom = np.zeros_like(top)
    width = depth = 5
    dz = top
    offset = dz + np.abs(dz.min())
    fracs = offset.astype(float) / offset.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    color_values = cm.jet(norm(fracs.tolist()))

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color=color_values, edgecolor="black", )
    ax1.set_title('Shaded')

    colourMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    colourMap.set_array(top)
    colBar = plt.colorbar(colourMap, shrink=0.5, extend='both').set_label('AUC Value')
    plt.savefig(f'{x}lstm_layer-lstm_num???AUC???????????????',dpi=300)
    plt.show()
    print("33333")
    """
    args = parse_args()
    auc_mean = main(args)
    print("\n\n")



