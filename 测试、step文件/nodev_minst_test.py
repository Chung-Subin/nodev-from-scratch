import numpy as np
from nodev import dataset, Dataloader
import nodev.operations as F
from nodev.model import MLP

# 1️⃣ 加载测试集
test_set = dataset.MNIST(train=False)
test_loader = Dataloader(test_set, batch_size=100)

# 2️⃣ 构建同样的模型
model = MLP((1000, 10), activation=F.relu)

# 3️⃣ 加载你训练的权重
model.load_weights("./minst_test.npz")

# 4️⃣ 测试
correct = 0
total = 0

for x, t in test_loader:
    y = model(x)
    pred = y.data.argmax(axis=1)
    correct += (pred == t.data).sum()
    total += len(t)

print("accuracy:", correct / total)