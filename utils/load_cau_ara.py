#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Endeavour -> cau_ara
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-09-28 17:00
@Desc   ：加载数据，
=================================================='''
import torch
from torch import tensor
from torch.utils.data import random_split
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData,Data
import numpy as np
import os.path as osp
import pickle
import random
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        print('encoding...')
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

class SequenceEncoder(object):
    # The 'SequenceEncoder' encodes raw column strings into embeddings.
    def __init__(self, model_name, save_path, device=None,):
        self.device = device
        self.save_path = save_path
        self.model_name = model_name
        if model_name != "one-hot":
            # pass # 测试用，这样代码不会加载模型。如果无法运行把下行代码取消注释
            self.model = SentenceTransformer(model_name, device=device)

        current_dir = osp.dirname(osp.abspath(__file__))
        self.model_save_path = osp.join(current_dir,'..',self.save_path[:-18],f"{model_name}_embedding.pkl")

    @torch.no_grad()
    def __call__(self, df):
        # path = osp.join('../Data/embeddings.pkl')
        if self.model_name == 'one-hot':
            enc = OneHotEncoder()
            df = np.expand_dims(df.values, 1)
            enc.fit(df)
            x = enc.transform(df).toarray()
            return tensor(x).to(torch.float32)
        else:
            path = self.model_save_path
            if osp.exists(path):
                print('已经计算好向量编码，无需计算')
                with open(path, "rb") as fIn:
                    stored_data = pickle.load(fIn)
                    x = stored_data['embeddings']
            else:
                print('正在计算向量编码')
                x = self.model.encode(df.values, show_progress_bar=True,
                                      convert_to_tensor=True, device=self.device)
                # Store sentences & embeddings on disc
                with open(path, "wb") as fOut:
                    pickle.dump({'embeddings': x}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

            return x.to(device)


class GenresEncoder(object):
    # The 'GenreEncoder' splits the raw column strings by 'sep' and converts
    # individual elements to categorical labels.
    def __init__(self, sep=''):
        self.sep = sep

    def __call__(self, df):
        # genres = set(g for col in df.values for g in col.split(self.sep))
        genres=['salt','pes']
        # genres = set(labels)
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            # for genre in col:
            x[i, mapping[col]] = 1
        return x


class IdentityEncoder(object):
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

def pes_feature(pes_path):
    df = pd.read_csv(pes_path, header=None)
    my_array = np.array(df)
    my_tensor = torch.tensor(my_array, dtype=torch.float)
    my_tensor = my_tensor.unsqueeze(0)
    nosalt = my_tensor[:, :588, :]
    salt = my_tensor[:, 588:, :]
    return nosalt.to(device),salt.to(device)

def ara_data(node_path,edge_path,pes_path,seq_names,path):
    # 读取蛋白和蛋白mapping # 'description': SequenceEncoder(),
    protein_x, protein_mapping = load_node_csv(node_path, index_col='name',
                                                encoders={
                                                    'description': SequenceEncoder(model_name=seq_names,save_path = path),
                                                    'label': GenresEncoder()
                                                })
    protein_feature=protein_x[:,:-2]
    protein_label=protein_x[:,-2:]
    protein_label = np.argmax(protein_label, axis=1)
    pes_protein_fea = protein_feature[protein_label == 1]
    salt_protein_fea = protein_feature[protein_label == 0]
    nosalt,salt  =pes_feature(pes_path = pes_path)

    nosalt = torch.squeeze(nosalt, dim=0)
    nosalt = nosalt.T.repeat(120, 1)
    nosalt = nosalt[:pes_protein_fea.shape[0]]

    # nosalt = nosalt.repeat(pes_protein_fea.shape[0], 1)
    nosalt_protein = torch.cat([nosalt, pes_protein_fea], dim=-1)

    salt = torch.squeeze(salt, dim=0) # 去掉一维卷积的一个维度
    # salt = salt.repeat(salt_protein_fea.shape[0], 1) # 重复向量
    salt = salt.T.repeat(50, 1)# 重复向量
    salt = salt[:salt_protein_fea.shape[0]]
    salt_protein = torch.cat([salt, salt_protein_fea], dim=-1) # 和蛋白序列信息合并

    protein_and_pes = torch.cat([salt_protein, nosalt_protein], dim=0)
    edge_index, edge_label = load_edge_csv(
        edge_path,
        src_index_col='protein_source',
        src_mapping=protein_mapping,
        dst_index_col='protein_target',
        dst_mapping=protein_mapping,
        encoders={'score': IdentityEncoder(dtype=torch.long)},
    )

    data = Data(x=protein_and_pes, edge_index=edge_index, y=protein_label,
                num_features = protein_and_pes.shape[1])

    leng_dataset = len(protein_mapping)
    a = range(leng_dataset)
    train_dataset, test_dataset, valid_dataset = random_split(
        dataset=a,
        lengths=[int(leng_dataset * 0.8),
                 int(leng_dataset * 0.1),
                 int(leng_dataset * 0.1), ],
        generator=torch.Generator().manual_seed(random.randint(1,999))
    )
    split = {
        'train_idx': np.array(train_dataset),
        'val_idx': np.array(test_dataset),
        'test_idx': np.array(valid_dataset),
    }

    allmask={}
    for name in ['train', 'val', 'test']:
        idx = split[f'{name}_idx']
        idx = torch.from_numpy(idx).to(torch.long)
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        allmask[f'{name}_mask'] = mask

    data.train_mask = allmask['train_mask']
    data.val_mask = allmask['val_mask']
    data.test_mask = allmask['test_mask']

    return data.to(device),protein_mapping

if __name__ == '__main__':
    pass


