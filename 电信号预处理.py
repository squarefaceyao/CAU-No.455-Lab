#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CAU-No.455-Lab -> 电信号预处理
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-12-29 11:38
@Desc   ：
=================================================='''
import torch
from torch import tensor
from torch.utils.data import random_split
import pandas as pd
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt



"""
# 绘制图片
plt.figure(figsize=(15,3))
plt.plot(salt.T.flatten()[588*4:],color='blue',label="wave b")
plt.plot(nosalt.T.flatten()[588*4:],color='g',label="wave a")
plt.savefig("test.png",dpi=600)
plt.legend()
plt.show()
"""

def main(salt):


    def _slide_window(rows, sw_width, sw_steps):
        '''
        函数功能：
        按指定窗口宽度和滑动步长实现单列数据截取
        --------------------------------------------------
        参数说明：
        rows：单个文件中的行数；
        sw_width：滑动窗口的窗口宽度；
        sw_steps：滑动窗口的滑动步长；
        '''
        start = 0
        s_num = (rows - sw_width) // sw_steps  # 计算滑动次数
        new_rows = sw_width + (sw_steps * s_num)  # 完整窗口包含的行数，丢弃少于窗口宽度的采样数据；

        while True:
            if (start + sw_width) > new_rows:  # 如果窗口结束索引超出最大索引，结束截取；
                return
            yield start, start + sw_width
            start += sw_steps

    _test_list = []
    i = 0
    df3 = salt.T.flatten()

    for start, end in _slide_window(df3.shape[0], sw_width=588, sw_steps=4):
        _test_list.append(df3[start:end])

        i += 1

    print(f"循环了{i}")
    print(f"处理后的数据长度是{len(_test_list)}")
    return np.array(_test_list)

if __name__ == '__main__':
    pes_path = osp.join('Data/ara-protein/raw', '1_Sos1-100mM.csv')

    df = pd.read_csv(pes_path, header=None)
    my_tensor = np.array(df)
    # my_tensor = torch.tensor(my_array, dtype=torch.float)
    # my_tensor = my_tensor.unsqueeze(0)
    # nosalt = my_tensor[:588, :]
    # salt = my_tensor[588:, :]
    nosalt = my_tensor[:588, :]
    salt = my_tensor[588:, :]
    new_salt = main(salt=salt)
    new_nosalt = main(salt=nosalt)