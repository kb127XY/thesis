import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

points = [
    (0,1,1),
    (0,2,2),
    (0,3,2),
    (0,4,2),
    (0,5,1),
    (1,1,2),
    (1,2,2),
    (1,3,2),
    (1,4,2),
    (1,5,2),
    (2,4,2),
    (2,3,1),
]

# 求最大坐标
max_x = max(p[0] for p in points)
max_y = max(p[1] for p in points)

# 建立矩阵，未测试的位置设为 NaN
heatmap = np.full((max_y + 1, max_x + 1), np.nan)

for x, y, count in points:
    heatmap[y, x] = count

# 只设置 0, 1, 2 三种离散颜色
colors = plt.cm.Blues(np.linspace(0.3, 0.9, 3))
cmap = ListedColormap(colors)
cmap.set_bad(color="white")   # NaN 显示为白色

# 设置离散分界：[-0.5,0.5) -> 0, [0.5,1.5) -> 1, [1.5,2.5) -> 2
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

plt.figure(figsize=(6, 5))
sns.heatmap(
    heatmap.T,                 # 转置，实现 x/y 轴交换
    annot=False,               # 不显示格子里的数字
    cmap=cmap,
    norm=norm,
    cbar=True,
    square=True,               # x,y 单位长度相同
    linewidths=1.0,            # 网格线宽度
    linecolor="gray",          # 网格线颜色
    xticklabels=["A", "B", "C", "D", "E", "F"],  # 横轴标签
    yticklabels=["1", "2", "3"],
    cbar_kws={
        "ticks": [0, 1, 2],    # 色阶只显示 0,1,2
        "shrink": 0.6,         # 色阶条缩短
        "aspect": 15           # 色阶条变细一些
    }
)

plt.title("Heatmap of effective grasping point (100_150_kl15)")
plt.xlabel("X Position")   # 原来的 y 变成横轴
plt.ylabel("Y Position")   # 原来的 x 变成纵轴

plt.show()