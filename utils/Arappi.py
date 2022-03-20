#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Endeavour -> Arappi
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-09-29 11:56
@Desc   ：
=================================================='''
from typing import Optional, Callable, List
import os
import os.path as osp
import torch
from utils import ara_data
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)


class ARAPPI(InMemoryDataset):
    # 下载数据链接
    url = "http://39.100.142.42:8080/dataset/29/zip?name=ara-protein"


    def __init__(self, root: str,seq_name: str,transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.seq_name = seq_name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.protein_mapping, self.slices = torch.load(self.processed_paths[0]+'.mapping')

    @property
    def raw_file_names(self) -> List[str]:
        '''
         根据列表里的文件名字，检查文件是否存在本地，不在本地就需要下载。
        :return:
        '''
        names = ['3_train_cau_node.csv','2_train_cau_edge.csv']
        # filename = self.url.rpartition('/')[2].split('?')[0]
        # print(filename)
        return [f'{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        '''
        保存处理后的数据名字
        :return:
        '''
        return 'data.pt'

    def download(self):
        '''
        根据url下载数据
        :return:
        '''
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self):
        '''
        主要的处理函数
        :return:
        '''
        print("选择的节点编码方式为：",self.seq_name)
        protein_path = osp.join(self.raw_dir, '3_train_cau_node.csv')
        relation_path = osp.join(self.raw_dir , '2_train_cau_edge.csv')
        pes_path = osp.join(self.raw_dir , '1_Sos1-100mM.csv')

        data,protein_mapping = ara_data(protein_path, relation_path,pes_path, seq_names=self.seq_name,path = self.processed_paths[0])
        # 边预测的时候需要使用，进行正负采样。
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])
        torch.save(self.collate([protein_mapping]), self.processed_paths[0]+'.mapping')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


