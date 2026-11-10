import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def parse_log_file(file_path):
    """
    解析日志文件，提取每个epoch的测试集准确率和平均损失，
    以及训练过程中每个batch的准确率和损失
    """
    epochs = []
    test_accuracies = []
    test_losses = []

    # 训练集数据：每个batch
    train_batches = []
    train_losses = []
    train_accuracies = []
    train_epochs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = None

    for line in lines:
        # 匹配训练batch数据
        # 格式: 1	0	2.353154	10.00%		11:48:21
        batch_match = re.match(r'(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%', line)
        if batch_match:
            epoch = int(batch_match.group(1))
            batch = int(batch_match.group(2))
            loss = float(batch_match.group(3))
            acc = float(batch_match.group(4))

            train_epochs.append(epoch)
            train_batches.append(batch)
            train_losses.append(loss)
            train_accuracies.append(acc)
            continue

        # 匹配测试结果行
        # 格式: 测试准确率: 82.01%, 平均损失: 1.099302
        test_match = re.search(r'Epoch (\d+) 测试结果:', line)
        if test_match:
            epoch = int(test_match.group(1))
            # 读取下一行获取准确率和损失
            next_line = lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else ""
            acc_match = re.search(r'测试准确率:\s+([\d.]+)%', next_line)
            loss_match = re.search(r'平均损失:\s+([\d.]+)', next_line)

            if acc_match and loss_match:
                epochs.append(epoch)
                test_accuracies.append(float(acc_match.group(1)))
                test_losses.append(float(loss_match.group(1)))

    return (train_epochs, train_batches, train_accuracies, train_losses,
            epochs, test_accuracies, test_losses)


def aggregate_train_by_epoch(train_epochs, train_accuracies, train_losses):
    """
    将训练集的batch数据按epoch聚合，计算每个epoch的平均准确率和平均损失
    """
    epoch_train_acc = {}
    epoch_train_loss = {}

    for epoch, acc, loss in zip(train_epochs, train_accuracies, train_losses):
        if epoch not in epoch_train_acc:
            epoch_train_acc[epoch] = []
            epoch_train_loss[epoch] = []
        epoch_train_acc[epoch].append(acc)
        epoch_train_loss[epoch].append(loss)

    # 计算每个epoch的平均值
    epochs = sorted(epoch_train_acc.keys())
    avg_accuracies = [sum(epoch_train_acc[e]) / len(epoch_train_acc[e]) for e in epochs]
    avg_losses = [sum(epoch_train_loss[e]) / len(epoch_train_loss[e]) for e in epochs]

    return epochs, avg_accuracies, avg_losses


def plot_side_by_side(train_epochs, train_accuracies, train_losses,
                      test_epochs, test_accuracies, test_losses):
    """
    绘制两个横向排列的图像：
    左图：训练集的loss和准确率
    右图：测试集的loss和准确率
    """
    # 创建图形，两个子图横向排列
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ========== 左图：训练集 ==========
    # 设置X轴刻度（每2个epoch显示一个）
    x_ticks_train = np.arange(0, max(train_epochs) + 1, 5)
    ax1.tick_params(axis='both', labelsize=16)
    # 绘制准确率曲线
    color1='gray'
    color2 = 'black'
    ax1.plot(train_epochs, train_accuracies, color='gray',marker='o', linewidth=2,
             markersize=4, linestyle='-', label='训练准确率')
    ax1.set_xlabel('轮次', fontsize=16)
    ax1.set_ylabel('准确率(%)',  fontsize=16)
    ax1.tick_params(axis='y')
    #ax1.set_ylim([75, 100])  # 设置Y轴范围
    ax1.set_xlim([0, max(train_epochs) + 1])  # 设置X轴范围

    # 绘制损失曲线（使用不同的Y轴）
    ax1_right = ax1.twinx()

    ax1_right.plot(train_epochs, test_accuracies, color=color2, marker='p', linewidth=2,
                   markersize=4, linestyle='--', label='测试准确率')


    ax1_right.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
    # 设置X轴刻度
    ax1.set_xticks(x_ticks_train)

    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=16)

    # 显示数值标签（每隔5个epoch显示一次）
    """    for i, (epoch, acc, loss) in enumerate(zip(train_epochs, train_accuracies, train_losses)):
        if epoch % 5 == 0 or epoch == train_epochs[-1]:
            ax1.annotate(f'{acc:.1f}%', (epoch, acc),
                         xytext=(0, 8), textcoords='offset points',
                         fontsize=14, color=color1, ha='center')
            ax1_right.annotate(f'{loss:.3f}', (epoch, loss),
                               xytext=(0, -12), textcoords='offset points',
                               fontsize=14, color=color2, ha='center')"""

    # ========== 右图：测试集 ==========
    ax2.tick_params(axis='both', labelsize=16)
    x_ticks2 = np.arange(0, max(train_epochs) + 1, 5)

    # 绘制准确率曲线
    print(train_losses)
    ax2.plot(train_epochs, train_losses, color=color1, marker='o', linewidth=2,
             markersize=4, linestyle='-', label='训练平均损失')
    ax2.set_xlabel('轮次', fontsize=16)
    ax2.set_ylabel('平均损失 (%)', fontsize=16)
    ax2.tick_params(axis='y')
    #ax2.set_ylim([75, 100])  # 设置Y轴范围
    ax2.set_xlim([0, max(test_epochs) + 1])  # 设置X轴范围

    # 绘制损失曲线（使用不同的Y轴）
    ax2_right = ax2.twinx()
    ax2_right.plot(test_epochs, test_losses, color=color2, marker='p', linewidth=2,
                   markersize=4, linestyle='--', label='测试平均损失')
    #ax2_right.tick_params(axis='y', labelcolor=color2)
    ax2_right.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
    # 设置X轴刻度
    ax2.set_xticks(x_ticks2)

    # 添加网格
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='best', fontsize=16)

    """    # 显示数值标签（每隔5个epoch显示一次）
    for i, (epoch, acc, loss) in enumerate(zip(test_epochs, test_accuracies, test_losses)):
        if epoch % 5 == 0 or epoch == test_epochs[-1]:
            ax2.annotate(f'{acc:.1f}%', (epoch, acc),
                         xytext=(0, 8), textcoords='offset points',
                         fontsize=9, color=color1, ha='center')
            ax2_right.annotate(f'{loss:.3f}', (epoch, loss),
                               xytext=(0, -12), textcoords='offset points',
                               fontsize=9, color=color2, ha='center')"""

    # 添加总标题
    plt.suptitle('实验1：PyTorch完成MLP手写数字识别', fontsize=20, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    plt.savefig('training_test_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(train_epochs, train_accuracies, train_losses,
                  test_epochs, test_accuracies, test_losses):
    """
    打印训练总结信息
    """
    print("=" * 70)
    print("训练总结")
    print("=" * 70)

    print(f"\n【训练集】")
    print(f"  总训练轮数: {len(train_epochs)}")
    print(f"  初始准确率 (Epoch 1): {train_accuracies[0]:.2f}%")
    print(f"  最终准确率 (Epoch {train_epochs[-1]}): {train_accuracies[-1]:.2f}%")
    print(f"  准确率提升: +{train_accuracies[-1] - train_accuracies[0]:.2f}%")
    print(f"  初始损失 (Epoch 1): {train_losses[0]:.4f}")
    print(f"  最终损失 (Epoch {train_epochs[-1]}): {train_losses[-1]:.4f}")
    print(f"  损失降低: {train_losses[0] - train_losses[-1]:.4f}")

    best_train_acc_idx = train_accuracies.index(max(train_accuracies))
    print(f"  最佳准确率: {max(train_accuracies):.2f}% (Epoch {train_epochs[best_train_acc_idx]})")
    best_train_loss_idx = train_losses.index(min(train_losses))
    print(f"  最低损失: {min(train_losses):.4f} (Epoch {train_epochs[best_train_loss_idx]})")

    print(f"\n【测试集】")
    print(f"  总测试轮数: {len(test_epochs)}")
    print(f"  初始准确率 (Epoch 1): {test_accuracies[0]:.2f}%")
    print(f"  最终准确率 (Epoch {test_epochs[-1]}): {test_accuracies[-1]:.2f}%")
    print(f"  准确率提升: +{test_accuracies[-1] - test_accuracies[0]:.2f}%")
    print(f"  初始损失 (Epoch 1): {test_losses[0]:.4f}")
    print(f"  最终损失 (Epoch {test_epochs[-1]}): {test_losses[-1]:.4f}")
    print(f"  损失降低: {test_losses[0] - test_losses[-1]:.4f}")

    best_test_acc_idx = test_accuracies.index(max(test_accuracies))
    print(f"  最佳准确率: {max(test_accuracies):.2f}% (Epoch {test_epochs[best_test_acc_idx]})")
    best_test_loss_idx = test_losses.index(min(test_losses))
    print(f"  最低损失: {min(test_losses):.4f} (Epoch {test_epochs[best_test_loss_idx]})")

    print("=" * 70)


def main():
    # 文件路径 - 请根据实际文件位置修改
    file_path = './实验1/log_test_2.txt'  # 使用你提供的文件名

    try:
        # 解析日志文件
        (train_epochs, train_batches, train_accuracies, train_losses,
         test_epochs, test_accuracies, test_losses) = parse_log_file(file_path)

        if not train_epochs or not test_epochs:
            print("未找到有效的训练数据，请检查文件格式")
            print("期望的格式示例：")
            print("  训练batch: '1\t0\t2.353154\t10.00%'")
            print("  测试结果: 'Epoch 1 测试结果:\\n  测试准确率: 82.01%, 平均损失: 1.099302'")
            return

        # 按epoch聚合训练集数据（计算每个epoch的平均值）
        train_epochs_agg, train_accuracies_agg, train_losses_agg = aggregate_train_by_epoch(
            train_epochs, train_accuracies, train_losses
        )

        # 打印总结
        print_summary(train_epochs_agg, train_accuracies_agg, train_losses_agg,
                      test_epochs, test_accuracies, test_losses)

        # 绘制图表（两个横向排列）
        plot_side_by_side(train_epochs_agg, train_accuracies_agg, train_losses_agg,
                          test_epochs, test_accuracies, test_losses)

        # 额外输出训练集和测试集的最终对比
        print(f"\n【最终对比】")
        print(f"  训练集最终准确率: {train_accuracies_agg[-1]:.2f}%")
        print(f"  测试集最终准确率: {test_accuracies[-1]:.2f}%")
        print(f"  泛化差距: {train_accuracies_agg[-1] - test_accuracies[-1]:.2f}%")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        print("请确保文件在当前目录下，或修改 file_path 变量为正确的路径")
    except Exception as e:
        print(f"发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()