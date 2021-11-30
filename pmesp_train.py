import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, VGAE
from utils import ARAPPI
from utils import PMESPEncoder,GcnEncoder,proteinEncoder,cross_validation_with_val_set
import os.path as osp
import os
import shutil

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    loss, auc, std = cross_validation_with_val_set(data=data,
                                  model=model,
                                  args=args,
                                  transform=transform)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model',  type=str, help='chose mode', default='PMESP',
                       choices=['proteinEncoder', # 输入数据只有蛋白序列信息
                                'GcnEncoder' # 输入数据包含蛋白数据和电信号数据
                                ])
    parse.add_argument('--epochs', type=int, default=900)
    parse.add_argument('--lr', type=int, default=0.01)
    parse.add_argument('--folds', type=int, default=10)

    parse.add_argument('--lstm_layers', type=int, default=3) # 性能最优
    parse.add_argument('--lstm_hidden', type=int, default=7) # 性能最优
    parse.add_argument('--out_channels', type=int, default=16) # 性能最优

    parse.add_argument('--seq_names', type=str, help='chose transform squence protein description',
                        default="all-MiniLM-L6-v2",
                        choices=['all-MiniLM-L6-v2', 'roberta-large-nli-stsb-mean-tokens',
                                 'bert-base-nli-mean-tokens', 'one-hot'])
    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # args.lr = 0.01
    # args.lstm_hidden = 3
    # args.lstm_layers = 1
    # args.epochs = 600
    # args.model = 'GcnEncoder'
    # args.model = 'proteinEncoder'
    # args.seq_names = 'one-hot'
    main(args)