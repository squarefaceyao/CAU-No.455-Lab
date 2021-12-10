import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
import time
from itertools import combinations,permutations

localtime = time.asctime( time.localtime(time.time()) )
# plt 图片大小
sns.set_context("talk",font_scale=2.5)
# setup the figure and axes
fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(projection='3d')
top = np.array([0.95333665, 0.96136744, 0.96361233, 0.96502169, 0.95059894,
       0.9589719 , 0.96232781, 0.96376317, 0.95420918, 0.95786347,
       0.96098537, 0.96000015, 0.94492959, 0.95715988, 0.96252834,
       0.95961287])
x = np.array([ 8, 16, 24, 32,  8, 16, 24, 32,  8, 16, 24, 32,  8, 16, 24, 32])
y = np.array([ 7,  7,  7,  7, 14, 14, 14, 14, 21, 21, 21, 21, 28, 28, 28, 28])

# fake data
# _x = np.arange(1,30,3)
# _y = np.arange(1,30,3)
# _xx, _yy = np.meshgrid(_x, _y)
# x, y = _xx.ravel(), _yy.ravel()
# top = x + y
bottom = np.zeros_like(top)
width = depth = 5
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
plt.savefig(f'{x}lstm-GCN hidden和AUC之间的关联',dpi=300)
plt.show()

print("33333")
