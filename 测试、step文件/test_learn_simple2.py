import math
import numpy as np
import nodev
from nodev import optimizer, Dataloader
import nodev.operations as F
from nodev.model import MLP

from nodev.dataset import get_spiral
import matplotlib.pyplot as plt
batch_size=10
def plot_decision_boundary(model, x, t):

    h = 0.01   # 网格分辨率

    x_min, x_max = x[:,0].min()-0.5, x[:,0].max()+0.5
    y_min, y_max = x[:,1].min()-0.5, x[:,1].max()+0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # 预测
    y = model(grid)
    y = y.data

    pred = np.argmax(y, axis=1)

    Z = pred.reshape(xx.shape)

    plt.figure(figsize=(6,6))

    # 决策区域
    plt.contourf(xx, yy, Z, alpha=0.3)

    # 原始数据点
    plt.scatter(x[:,0], x[:,1], c=t, cmap='viridis')

    plt.title("Decision Boundary")
    plt.show()

train_set=nodev.dataset.Spiral(train=True)
test_set=nodev.dataset.Spiral(train=False)
trainloader=Dataloader(train_set,batch_size)
testloader=Dataloader(test_set,batch_size,shuffle=False)


"""markers = ['o','x','^']

for i in range(3):
    plt.scatter(x[t==i,0], x[t==i,1], marker=markers[i])"""

"""plt.show()"""
max_epoch=300
batch_size=30
hidden_size=10
lr=1.0
model=MLP((hidden_size,3))
optimizer=optimizer.SGD(lr).setup(model)




for epoch in range(max_epoch):
    sum_loss=0
    sum_acc=0
    for x,t in trainloader:
        y=model(x)
        loss=F.softmax_cross_entropy(y, t)
        acc=F.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)


    print('ep {}'.format(epoch+1))
    print('train loss:{:.4f},accuracy {:4f}'.format(sum_loss/len(train_set),sum_acc/len(train_set)))


