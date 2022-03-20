#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：CAU-No.455-Lab -> plot_bar
@IDE    ：PyCharm
@Author ：Mr. Y
@Date   ：2021-12-12 10:28
@Desc   ：绘制柱状图
=================================================='''

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import time
localtime = time.asctime( time.localtime(time.time()) )
# plt 图片大小
sns.set_context("talk",font_scale=2)
import numpy as np
import matplotlib.pyplot as plt

x1_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]
pmesp = [0.9558, 0.9583, 0.9616, 0.9602, 0.9662, 0.9664, 0.96, 0.9601, 0.9581]
GCN_ps = [0.5321, 0.5614, 0.6061, 0.6214, 0.6321, 0.6321, 0.6221, 0.619, 0.6021]
GCN_p = [0.9058, 0.9183, 0.9216, 0.9202, 0.9222, 0.9212, 0.9219, 0.9101, 0.911]
x =list(range(len(x1_list)))
plt.figure(figsize=(16,10))
total_width, n = 0.9, 3
width = total_width / n
plt.bar(x, pmesp, width=width,label='PMESP')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, GCN_p, width=width, label='GAE_p',tick_label = x1_list,fc = 'r')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, GCN_ps, width=width, label='GAE_ps',fc = 'g')

plt.xlabel('Train epochs')
plt.ylabel('AUC Value')
plt.ylim((0.5, max(pmesp)))
my_y_ticks = np.arange(0.5, max(pmesp), 0.05)
plt.yticks(my_y_ticks)
plt.grid(b=True,axis="y")
plt.legend()
plt.savefig(f'epoch和AUC之间的关联',dpi=300)
plt.show()