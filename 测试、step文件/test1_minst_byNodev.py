import os.path
import time

import nodev
from nodev import optimizer, Dataloader
import nodev.operations as F
from nodev.model import MLP
from nodev import dataset


path= "../minst_test.npz"

max_epoch=40
batch_size=100

train_set=dataset.MNIST(train= True)
train_loader=Dataloader(train_set,batch_size)
model=MLP((1000,10),activation=F.relu)
optimizer=optimizer.SGD().setup(model)


if nodev.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

if os.path.exists(path):
    model.load_weights(path)

for epoch in range(max_epoch):
    start=time.time()
    sum_loss=0

    for x,t in train_loader:
       # x = x.reshape(1, -1)


        y=model(x)
        loss=F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss+=float(loss.data)*len(t)

    elapsed_time=time.time()-start
    print('ep{}'.format(epoch+1))
    print('train loss:{:.4f},time {:4f}'.format(sum_loss/len(train_set), elapsed_time))
model.save_weight(path)