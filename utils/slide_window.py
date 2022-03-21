#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CAU-No.455-Lab -> slide_window
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2022-03-20 09:52
@Desc   ：滑动窗口扩充数据集
=================================================='''
import numpy as np
import torch


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


def Slide_expand(salt,sw_steps=4):

    _test_list = []
    i = 0
    df3 = salt.T.flatten()

    for start, end in _slide_window(df3.shape[0], sw_width=588, sw_steps=sw_steps):
        _test_list.append(df3[start:end])

        i += 1

    # print(f"循环了{i}")
    print(f"处理后的数据长度是{len(_test_list)}")

    return torch.from_numpy(np.array(_test_list))
