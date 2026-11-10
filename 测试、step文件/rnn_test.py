#import numpy as np
import nodev
from nodev import optimizer
import nodev.operations as F
#import nodev.layers as L
from nodev.model import SimpleRNN

max_epoch=150

hidden_size=100
bptt_lenth=30
lr=1.0
model=SimpleRNN(hidden_size,1)
#optimizer=optimizer.SGD(lr).setup(model)
train_set=nodev.dataset.SinCurve(train=True)
seq_len=len(train_set)
optimizer=optimizer.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss,count=0,0
    for x,t in train_set:
        x=x.reshape(1,1)
        y=model(x)
        loss+=F.mean_squared_error(y,t)
        count+=1

        if count%bptt_lenth==0 or count==seq_len:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        avg_loss=float(loss.data)/count
        print("ep %d\tloss%f"%(epoch+1,avg_loss))