import torch
from torch import nn
from torchvision import transforms
#from nodev.dataloaders import Dataloader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
log_file="./实验1/log_test_2.txt"

batch_size=100
lr=0.001
max_epoch=40
model_path= "./实验1/test_1.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
class MNIST(Dataset):
    def __init__(self, train=True,
                 transform=None,
                 target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.prepare()

    def prepare(self):

        if self.train:
            data_path = "data/train-images.idx3-ubyte"
            label_path = "data/train-labels.idx1-ubyte"
        else:
            data_path = "data/t10k-images.idx3-ubyte"
            label_path = "data/t10k-labels.idx1-ubyte"

        self.data = self._load_images(data_path)
        self.label = self._load_labels(label_path)

    def _load_images(self, path):

        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        #data = data.reshape(-1, 784)
        data = data.reshape(-1, 28, 28)
        return data

    def _load_labels(self, path):

        with open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(1000, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def cnn():
    return MLP().to(device)
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 numpy array 转换为 tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])
model=cnn()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
train_set=MNIST(train=True,transform=transform)
test_set=MNIST(train=False,transform=transform)

train_loader=DataLoader(train_set,batch_size=batch_size)
test_loader=DataLoader(test_set,batch_size=batch_size)
def init_log_file():
    """初始化日志文件，写入头部信息"""
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MNIST手写数字识别训练日志\n")
        f.write(f"批次大小: {batch_size}\n")
        f.write(f"学习率: {lr}\n")
        f.write(f"最大轮数: {max_epoch}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Epoch\tBatch\tLoss\t\tAccuracy\tTimestamp\n")
        f.write("-" * 80 + "\n")

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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    # 用于累积整个epoch的损失和准确率
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 计算当前batch的准确率
        predictions = pred.argmax(dim=1)  # 获取预测的类别
        batch_correct = (predictions == y).sum().item()
        batch_size = len(X)
        batch_accuracy = 100 * batch_correct / batch_size

        # 累积统计信息
        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_samples += batch_size

        # 每100个batch打印并记录一次
        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}, accuracy: {batch_accuracy:>5.2f}%  [{current:>5d}/{size:>5d}]")
            write_to_log(epoch + 1, batch, loss.item(), batch_accuracy)

    # 计算整个epoch的平均损失和准确率
    avg_loss = total_loss / total_samples
    avg_accuracy = 100 * total_correct / total_samples

    return avg_loss, avg_accuracy
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"错误率: \n 精度: {(100*correct):>0.1f}%, Avg 损失: {test_loss:>8f} \n")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\nEpoch {epoch + 1} 测试结果:\n")
        f.write(f"  测试准确率: {accuracy:.2f}%, 平均损失: {test_loss:.6f}\n")
        f.write("-" * 40 + "\n")
init_log_file()
for epoch in range(max_epoch):
    print("——————现在是第{}轮训练——————".format(epoch+1))
    train_loss, train_acc=train(train_loader, model, loss_fn, optimizer)
    #write_epoch_summary(ep + 1, train_loss, train_acc)
    test(test_loader,model, loss_fn)
torch.save(model.state_dict(), model_path)
