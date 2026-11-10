import numpy as np
import struct
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_mnist_images(filename):
    """加载MNIST图像文件"""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(filename):
    """加载MNIST标签文件"""
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def get_nth_image_by_label(images, labels, target_label, n=0):
    """获取指定标签的第n张图像（n=0表示第一张）"""
    indices = np.where(labels == target_label)[0]
    if len(indices) > n:
        return images[indices[n]]
    return None


# 设置路径
data_dir = Path('../data')

# 加载数据
print("加载MNIST数据...")
train_images = load_mnist_images(data_dir / 'train-images.idx3-ubyte')
train_labels = load_mnist_labels(data_dir / 'train-labels.idx1-ubyte')

# 九宫格配置：位置对应的数字（1-9）
grid_config = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 创建九宫格
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle('MNIST 数字九宫格 (1-9)', fontsize=18, fontweight='bold')

for i in range(3):
    for j in range(3):
        digit = grid_config[i][j]
        # 获取该数字的第一张图像
        img = get_nth_image_by_label(train_images, train_labels, digit, n=0)

        ax = axes[i, j]
        if img is not None:
            ax.imshow(img, cmap='gray')
           # ax.set_title(f'数字: {digit}', fontsize=16, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, f'未找到数字 {digit}',
                    ha='center', va='center', fontsize=14)
            ax.set_title(f'数字: {digit}', fontsize=16, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

        # 添加边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('darkblue')
            spine.set_linewidth(2)


plt.tight_layout()

# 保存
save_path = '../mnist_9x9_grid.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"九宫格图像已保存: {save_path}")

plt.show()