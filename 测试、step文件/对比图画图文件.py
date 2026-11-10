import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def parse_log_file(file_path):
    """
    解析日志文件，提取每个epoch的测试准确率和平均损失
    返回: (epochs, accuracies, losses)
    """
    epochs = []
    test_accuracies = []
    test_losses = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 匹配测试结果行
        # 格式: 测试准确率: 82.01%, 平均损失: 1.099302
        pattern = r'Epoch (\d+) 测试结果:\s*测试准确率: ([\d.]+)%, 平均损失: ([\d.]+)'

        matches = re.findall(pattern, content)

        for match in matches:
            epoch = int(match[0])
            accuracy = float(match[1])
            loss = float(match[2])

            epochs.append(epoch)
            test_accuracies.append(accuracy)
            test_losses.append(loss)

        return epochs, test_accuracies, test_losses

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return None, None, None
    except Exception as e:
        print(f"解析文件 '{file_path}' 时发生错误：{e}")
        return None, None, None


def plot_combined_comparison(log_files, labels=None, colors=None):
    """
    在同一张画布上绘制准确率和损失对比图（上下两个子图）

    参数:
        log_files: 日志文件路径列表
        labels: 图例标签列表（可选）
        colors: 颜色列表（可选）
    """
    # 创建图形，上下两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # 默认颜色和线条样式
    default_colors = ['gray', 'gray', 'black']
    default_styles = ['-', 'dashed', '-.']

    data = []

    # 解析所有日志文件
    for i, log_file in enumerate(log_files):
        epochs, accuracies, losses = parse_log_file(log_file)

        if epochs is None:
            continue

        data.append((epochs, accuracies, losses))

        # 确定标签
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = Path(log_file).stem

        # 确定颜色
        color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        linestyle = default_styles[i % len(default_styles)]

        # ===== 上子图：准确率对比 =====
        ax1.plot(epochs, accuracies, color=color, marker='o', linewidth=2,
                 markersize=4, linestyle=linestyle, label=label, alpha=0.8,markevery=5)

        # 在最后一个点添加数值标签
        ax1.annotate(f'{accuracies[-1]:.1f}%',
                     (epochs[-1], accuracies[-1]),
                     xytext=(-20, -12), textcoords='offset points',
                     fontsize=12, color=color, fontweight='bold')

        # ===== 下子图：损失对比 =====
        ax2.plot(epochs, losses, color=color, marker='s', linewidth=2,
                 markersize=4, linestyle=linestyle, label=label, alpha=0.8,markevery=5)

        # 在最后一个点添加数值标签
        ax2.annotate(f'{losses[-1]:.3f}',
                     (epochs[-1], losses[-1]),
                     xytext=(-20, 3), textcoords='offset points',
                     fontsize=12, color=color, fontweight='bold')

    # ===== 设置上子图（准确率）属性 =====
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('测试准确率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('不同实验的准确率对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=12)

    # 设置准确率Y轴范围
    all_acc = [acc for _, acc, _ in data if acc]
    if all_acc:
        min_acc = min([min(acc) for acc in all_acc])
        max_acc = max([max(acc) for acc in all_acc])
        ax1.set_ylim([max(70, min_acc - 5), min(100, max_acc + 2)])

    # ===== 设置下子图（损失）属性 =====
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均损失', fontsize=12, fontweight='bold')
    ax2.set_title('不同实验的损失对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=12)

    # 设置损失Y轴范围
    all_losses = [loss for _, _, loss in data if loss]
    if all_losses:
        max_loss = max([max(loss) for loss in all_losses])
        min_loss = min([min(loss) for loss in all_losses])
        ax2.set_ylim([max(0, min_loss - 0.1), max_loss + 0.2])

    # 设置X轴刻度
    if data:
        max_epoch = max([max(epochs) for epochs, _, _ in data if epochs])
        xticks = range(0, max_epoch + 1, 5)
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)

    # 调整布局，防止标签重叠
    plt.tight_layout()

    # 保存图片
    plt.savefig('combined_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_comparison_summary(log_files, labels=None):
    """
    打印对比总结信息
    """
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)

    results = []

    for i, log_file in enumerate(log_files):
        epochs, accuracies, losses = parse_log_file(log_file)

        if epochs is None:
            continue

        label = labels[i] if labels and i < len(labels) else Path(log_file).stem

        results.append({
            'label': label,
            'final_acc': accuracies[-1],
            'final_loss': losses[-1],
            'best_acc': max(accuracies),
            'best_acc_epoch': epochs[accuracies.index(max(accuracies))],
            'min_loss': min(losses),
            'min_loss_epoch': epochs[losses.index(min(losses))],
            'improvement': accuracies[-1] - accuracies[0]
        })

    # 打印表格
    print(f"{'配置':<20} {'最终准确率':<12} {'最终损失':<12} {'最佳准确率':<12} {'准确率提升':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['label']:<20} {r['final_acc']:>6.2f}%     {r['final_loss']:>8.4f}   "
              f"{r['best_acc']:>6.2f}% (E{r['best_acc_epoch']:<2})  {r['improvement']:>+6.2f}%")

    # 找出最佳模型
    if results:
        best_acc_model = max(results, key=lambda x: x['final_acc'])
        best_loss_model = min(results, key=lambda x: x['final_loss'])

        print("\n" + "=" * 80)
        print("最佳模型：")
        print(f"  最高准确率: {best_acc_model['label']} ({best_acc_model['final_acc']:.2f}%)")
        print(f"  最低损失: {best_loss_model['label']} ({best_loss_model['final_loss']:.4f})")
        print("=" * 80)


def main():
    # ===== 在这里配置你的3个日志文件路径 =====
    log_files = [
        './实验1/log_test_2.txt',  # 实验1的日志文件
        './实验2/lr=1e-3.txt',  # 实验2的日志文件
        './实验3/lr=1e-3.txt'  # 实验3的日志文件
    ]

    # 可选：自定义图例标签（如果不设置，将使用文件名）
    labels = [
        '实验1',  # 第一个模型的标签
        '实验2',  # 第二个模型的标签
        '实验3'  # 第三个模型的标签
    ]

    # 可选：自定义颜色
    #colors = ['blue', 'red', 'green']

    # 打印对比总结
    print_comparison_summary(log_files, labels)

    # 绘制组合对比图（同一画布，上下两个子图）
    print("\n正在绘制组合对比图...")
    plot_combined_comparison(log_files, labels)

    print("\n图表已保存为: combined_comparison.png")


if __name__ == "__main__":
    main()
