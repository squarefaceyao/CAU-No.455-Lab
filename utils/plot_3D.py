import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import time
import brewer2mpl
from itertools import combinations,permutations

localtime = time.asctime( time.localtime(time.time()) )
# plt 图片大小
sns.set_context("talk",font_scale=2.5)
# setup the figure and axes
fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(projection='3d')


# fake data
_x = np.arange(1,30,3)
_y = np.arange(1,30,3)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
top = x + y
bottom = np.zeros_like(top)
width = depth = 2
dz = top
offset = dz + np.abs(dz.min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
color_values = cm.jet(norm(fracs.tolist()))


ax1.bar3d(x, y, bottom, width, depth, top, shade=True,color=color_values,edgecolor="black",)
ax1.set_title('Shaded')

colourMap = plt.cm.ScalarMappable(cmap=plt.cm.jet)
colourMap.set_array(top)
colBar = plt.colorbar(colourMap,shrink=0.5,extend='both').set_label('AUC value')

plt.show()

print("33333")
