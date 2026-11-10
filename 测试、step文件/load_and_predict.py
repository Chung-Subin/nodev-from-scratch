import os.path
import time
import numpy as np
import nodev
from nodev import optimizer, Dataloader
import nodev.operations as F
from nodev.model import MLP
from nodev import dataset
import matplotlib.pyplot as plt



path= "../minst_test.npz"

max_epoch=40
batch_size=100

train_set=dataset.MNIST(train=True)
train_loader=Dataloader(train_set,batch_size)
model=MLP((1000,10),activation=F.relu)
optimizer=optimizer.SGD().setup(model)

if os.path.exists(path):
    model.load_weights(path)