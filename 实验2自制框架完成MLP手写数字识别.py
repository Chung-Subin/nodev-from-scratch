from nodev.dataset import MNIST
import os.path
import nodev
from nodev import optimizer, Dataloader
import nodev.operations as F
from nodev.model import MLP
from datetime import datetime
import numpy as np



max_epoch=40
lr=1e-3
batch_size=100
path= []
log_file=[]
train_set=MNIST(train= True)
test_set=MNIST(train=False)
train_loader=Dataloader(train_set,batch_size)
test_loader=Dataloader(test_set,batch_size)
model=MLP((1000,10),activation=F.relu)
optimizer=optimizer.SGD().setup(model)

"""def init_title():
    fold_list_npz=[]
    fold_list_txt=[]
    x1="./实验2/"
    x2=".npz"
    x3=".txt"
    result = "lr="+f"{lr:.0e}".replace('e-0', 'e-')
    fold_list_npz.append(x1)
    fold_list_npz.append(result)
    fold_list_npz.append(x2)
    fold_list_txt.append(x1)
    fold_list_txt.append(result)
    fold_list_txt.append(x3)
    fold_list_npz = ''.join(fold_list_npz)
    fold_list_txt = ''.join(fold_list_txt)
    return fold_list_npz,fold_list_txt"""
def init_title():
    base_dir = "./实验2"   # 目录
    os.makedirs(base_dir, exist_ok=True)  # ✅ 确保目录存在

    result = "lr=" + f"{lr:.0e}".replace('e-0', 'e-')

    npz_path = os.path.join(base_dir, result + ".npz")
    txt_path = os.path.join(base_dir, result + ".txt")

    return npz_path, txt_path

def init_log_file():
    """初始化日志文件，写入头部信息"""
    print(log_file)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MNIST手写数字识别训练日志-实验2\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"最大轮数: {max_epoch}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Epoch\tBatch\tLoss\t\tAccuracy\tTimestamp\n")
        f.write("-" * 80 + "\n")
        #f.write('ceshi')

def write_to_log(epoch, batch, loss, accuracy=None):
    """写入训练信息到日志文件"""
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%H:%M:%S')
        if accuracy is not None:
            # 训练时写入准确率
            f.write(f"{epoch}\t{batch}\t{loss:.6f}\t{accuracy:.2f}%\t\t{timestamp}\n")
        else:
            # 测试时写入（没有准确率参数）
            f.write(f"{epoch}\t{batch}\t{loss:.6f}\t-\t\t{timestamp}\n")


def train(dataloader, model, optimizer, ep):
    size = len(dataloader.dataset)
    sum_loss = 0
    batch_count=0
    for x, t in dataloader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        predictions = np.argmax(y.data, axis=1)
        t_labels = np.argmax(t.data, axis=1) if t.data.ndim > 1 else t.data  # 处理one-hot编码

        batch_correct = np.sum(predictions == t_labels)
        batch_accuracy = 100 * batch_correct / batch_size

        if batch_count % 100 == 0:
            current =  (batch_count+1)  *batch_size

            loss_data=float(loss.data)
            print(f"loss: {loss_data:>7f}, accuracy: {batch_accuracy:>5.2f}%  [{current:>5d}/{size:>5d}]")
            write_to_log(ep + 1, batch_count, loss_data, batch_accuracy)
        batch_count+=1


def test(dataloader, model, ep):
    size = len(dataloader.dataset)  # 总样本数
    test_loss, correct = 0, 0
    total=0
    with nodev.no_grad():
        for x, t in dataloader:
            y = model(x)
            # 累加损失（乘以批次大小，后面再除以总样本数）
            test_loss += F.softmax_cross_entropy(y, t) * len(t)

            # 计算正确预测数
            pred = y.data.argmax(axis=1)
            correct += (pred == t.data).sum()
            total += len(t.data)


    # 计算平均值
    test_loss /= size  # 除以总样本数得到平均损失
    accuracy = correct / total * 100
    loss_data = float(test_loss.data)
    print(f"测试结果 - 准确率: {accuracy:.1f}%, 平均损失: {loss_data:.6f}")

    # 写入日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\nEpoch {ep + 1} 测试结果:\n")
        f.write(f"  测试准确率: {accuracy:.2f}%, 平均损失: {loss_data:.6f}\n")
        f.write("-" * 40 + "\n")

if nodev.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

path,log_file=init_title()
print("模型路径:", path)
print("日志路径:", log_file)
init_log_file()




for epoch in range(max_epoch):
    print("——————现在是第{}轮训练——————".format(epoch + 1))
    train(dataloader=train_loader, model=model, optimizer=optimizer, ep=epoch)
    test(dataloader=test_loader, model=model, ep=epoch)

model.save_weights(path)