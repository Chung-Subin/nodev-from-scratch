import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
"""
def sigmoid(x):
Sigmoid 函数: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

# 生成 x 轴数据，范围从 -50 到 50，共 500 个点以保证曲线平滑
x = np.linspace(-50, 50, 500)

# 计算对应的 y 值
#y = sigmoid(x)
y=relu(x)
# 创建图像
plt.figure(figsize=(10, 6))

# 绘制 Sigmoid 曲线
plt.plot(x, y, color='black', linewidth=2)

# 设置标题和轴标签
plt.title('Sigmoid 函数图像 (x ∈ [-50, 50])', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('σ(x)', fontsize=14)

# 设置坐标轴范围，让 y 轴稍留边距
#plt.xlim(-50, 50)
plt.ylim(-0.05, 1.05)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xticks(np.arange(-50, 50.01, 5))
# 添加网格线，便于观察
plt.grid(True, linestyle='--', alpha=0.6)

# 在 y=0.5 处添加水平参考线，表示中心点
plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.8)
# 在 x=0 处添加垂直参考线
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.8)
plt.plot(0, 0.5, 'ko', markersize=8)
# 显示图例
plt.legend()

# 显示图像
plt.show()"""



def relu(x):
    """ReLU 函数: f(x) = max(0, x)"""
    return np.maximum(0, x)

# 生成 x 轴数据，范围从 -10 到 10
x = np.linspace(-10, 10, 500)
y = relu(x)

# 创建图像
plt.figure(figsize=(10, 6))
plt.yticks(np.arange(0, 11.1, 2))
plt.xticks(np.arange(-10, 10.1, 2))
# 绘制 ReLU 曲线
plt.plot(x, y, color='black', linewidth=2)

# 设置标题和轴标签
plt.title('ReLU (Rectified Linear Unit) 函数图像 (x ∈ [-10, 10])', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x) = max(0, x)', fontsize=14)

# 设置坐标轴范围
plt.xlim(-10, 10)
plt.ylim(-1, 11)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 添加参考线
# 在 y=0.5 处添加水平参考线，表示中心点
plt.axhline(y=0.0, color='gray', linestyle='-', alpha=0.8)
# 在 x=0 处添加垂直参考线
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.8)

# 在拐点 (0,0) 处添加圆点
plt.plot(0, 0, 'ko', markersize=8)

# 添加图例
plt.legend()

# 显示图像
plt.show()