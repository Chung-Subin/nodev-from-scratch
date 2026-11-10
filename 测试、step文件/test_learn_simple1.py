from symtable import Class

from jinja2.optimizer import optimize
from tensorflow.python.ops.initializers_ns import variables
from xxhash import xxh128
import nodev.layers as L
from nodev import optimizer
from nodev import Datafield
import numpy as np
from nodev import operations as F
from nodev import Model
from nodev.model import MLP
lr=0.1
iters=100


def sphere(x,y):
    z=x**2+y**2
    return z

def matyas(x,y):
    z=0.26*(x**2+y**2)-0.48*x*y
    return z

def rosenbrock(x,y):
    z=100*(y-x**2)**2+(x-1)**2
    return z

"""np.random.seed(0)
x=np.random.randn(100,1)
y=5+2*x+np.random.rand(100,1)
x,y=Datafield(x),Datafield(y)
w=Datafield(np.zeros((1,1)))
b=Datafield(np.zeros(1))

def predict(x):
    y=F.matmul(x,w)+b
    return y
for i in range(iters):
    y_pred=predict(x)
    loss=F.mean_squared_error(y,y_pred)
    w.clear_grad()
    b.clear_grad()
    loss.backward()
    # 调试信息
    print(f"Iter {i}: w.grad = {w.grad}, b.grad = {b.grad}")
    if w.grad is None or b.grad is None:
        print("Gradient is None!")
        break
    w.data-=lr*w.grad.data
    b.data-=lr*b.grad.data
    print(w,b,loss)
"""
class TwoLayersNet(Model):
    def __init__(self,hidden_size,out_size):
        super().__init__()
        self.l1=L.Linear(hidden_size)
        self.l2=L.Linear(out_size)
    def forward(self,x):
        y=F.sigmoid(self.l1(x))
        y=self.l2(y)
        return y
hidden_size=10
model=MLP((hidden_size,1))
optimizer=optimizer.MomentumSGD(lr)
optimizer.setup(model)
np.random.seed(0)
x=np.random.randn(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

I,H,O=1,10,1
w1=Datafield(0.01 * np.random.randn(I, H))
b1=Datafield(np.zeros(H))
w2=Datafield(0.01 * np.random.randn(H, O))
b2=Datafield(np.zeros(O))

def predict(x):
    y=F.linear(x, w1, b1)
    y=F.sigmoid(y)
    y=F.linear(y, w2, b2)
    return y
lr=0.2
max_iters=10000

for i in range(max_iters):
    y_pred=model(x)
    loss=F.mean_squared_error(y,y_pred)
    model.cleargrads()
    loss.backward()
    optimizer.update()
    if i%1000==0:
        print(loss)

