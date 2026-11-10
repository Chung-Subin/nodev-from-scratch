import numpy as np
from typing import Tuple
from nodev import Datafield
import nodev
from nodev import as_array
from nodev import cuda
from nodev.core import Operation
from nodev import utils
from nodev import as_datafield


class Reshape(Operation):#矩阵变形
    def __init__(self,shape):
        self.shape=shape
    def forward(self,x):
        self.x_shape=x.shape
        y=x.reshape(self.shape)
        return y
    def backward(self,gy):
        return reshape(gy,self.x_shape)
class Sin(Operation):
    def forward(self,x):
        xp=cuda.get_array_module(x)
        y=xp.sin(x)
        return y
    def backward(self,gy):
        x,=self.inputs
        gx=gy*cos(x)
        return gx
class Cos(Operation):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.cos(x)
        return y
    def backward(self,gy):
        x,=self.inputs
        gx=-gy*sin(x)
        return gx
class Tanh(Operation):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.tanh(x)
        return y
    def backward(self,gy):
        y=self.outputs[0]()
        gx=gy*(1-y**2)
        return gx
class Transpose(Operation):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.transpose(x)
        return y
    def backward(self,gy):
        gx=transpose(gy)
        return gx
class Sum(Operation):
    def __init__(self,axis,keepdims):  #axis轴 keepdims=true 保留轴的数量
        self.axis=axis
        self.keepdims=keepdims
    def forward(self,x):
        self.x_shape=x.shape
        y=x.sum(axis=self.axis,keepdims=self.keepdims)
        return y
    def backward(self,gy):
        gy=utils.reshape_sum_backward(gy,self.x_shape,self.axis,self.keepdims)
        gx=broadcast_to(gy,self.x_shape)
        return gx
class BroadcastTo(Operation):
    def __init__(self,shape):
        self.shape=shape

    def forward(self,x):
        xp = cuda.get_array_module(x)
        self.x_shape=x.shape
        y=xp.broadcast_to(x,self.shape)
        return y
    def backward(self,gy):
        gx=sum_to(gy,self.x_shape)
        return gx
class MeanSquaredError(Operation):   #均方误差
    def forward(self,x0,x1):
        diff=x0-x1
        y=(diff**2).sum()/len(diff)
        return y

    def backward(self,gy):
        x0,x1=self.inputs
        diff=x0-x1
        gx0=gy*diff*(2./len(diff))
        gx1=-gx0
        return gx0,gx1
class Log(Operation):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y
    def backward(self,gy):
        x,=self.inputs
        gx = gy / x
        return gx
class Exp(Operation):
    def __init__(self):
        pass
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.exp(x)
        return y
    def backward(self,gy):

        x,=self.inputs
        xp = cuda.get_array_module(x)
        y = xp.exp(x.data)
        gx = gy * y
        return gx
class Clip(Operation):
    def __init__(self,x_min,x_max):
        self.x_min = x_min
        self.x_max = x_max
    def forward(self,x):
        y=np.clip(x,self.x_min,self.x_max)
        return y
    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx
class ReLU(Operation):
    def forward(self,x):
        xp = cuda.get_array_module(x)
        y=xp.maximum(x,0.0)
        return y
    def backward(self,gy):
        x,=self.inputs
        mask=x.data>0
        gx=gy*mask
        return gx

class SumTo(Operation):
    def __init__(self,shape):
        self.shape=shape

    def forward(self,x):
        self.x_shape=x.shape
        y=utils.sum_to(x,self.shape)
        return y
    def backward(self,gy):
        gx=broadcast_to(gy,self.x_shape)
        return gx
class MatMul(Operation):
    def forward(self,x,w):
        y=x.dot(w)
        return y
    def backward(self,gy):
        x,w=self.inputs
        gx=matmul(gy,w.T)
        gW=matmul(x.T,gy)
        return gx,gW
class Min(Operation):
    def __init__(self, axis, keepdims):
        self.keepdims = keepdims
        self.axis = axis

    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()
class Max(Operation):
    def __init__(self,axis,keepdims):
        self.keepdims=keepdims
        self.axis=axis
    def forward(self,x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()

        shape = utils.max_backward_shape(x.shape, self.axis)

        gy = reshape(gy, shape)
        y = reshape(y, shape)

        cond = (x.data == y.data)

        gy = broadcast_to(gy, cond.shape)

        gx = gy * cond

        return gx
class GetItem(Operation):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
class Linear(Operation):
    def forward(
        self, x: np.ndarray, W: np.ndarray, b= None
    ) -> Datafield:
        if W is None or W.data is None:
            print("ERROR: W is None in forward!")
            raise ValueError("W cannot be None")
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Datafield) -> Tuple[Datafield, Datafield, Datafield]:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
class GetItemGrad(Operation):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)

def sum(x,axis=None,keepdims=False):
    return Sum(axis,keepdims)(x)
def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)
def tanh(x):
    return Tanh()(x)

def reshape(x,shape):
    if x.shape==shape:
        return as_datafield(x)
    return Reshape(shape)(x)
def transpose(x):
    return Transpose()(x)
def broadcast_to(x,shape):
    if x.shape==shape:
        return as_datafield(x)
    else:
        return BroadcastTo(shape)(x)

def sum_to(x,shape):
    if x.shape==shape:
        return as_datafield(x)
    else:
        return SumTo(shape)(x)

def softmax_simple(x,axis=1):
    x=as_datafield(x)
    x = x - max(x,axis=axis, keepdims=True)
    y=exp(x)
    sum_y=sum(y,axis=axis,keepdims=True)
    return y/sum_y
"""def softmax_cross_entropy_simple(x,t):
    x,t=as_variable(x),as_variable(t)
    N=x.shape[0]
    p=softmax_simple(x)
    p=clip(p,1e-15,1.0)
    log_p=log(p)
    tlog_p = get_item(log_p, (np.arange(N), t.data))
    #tlog_p=log_p[np.arange(N),t.data]
    y=-1*sum(tlog_p)/N
    return y"""


def softmax_cross_entropy(x, t):
    x = as_datafield(x)
    t = as_datafield(t)

    if t.ndim == 1:
        N = x.shape[0]

        # 数值稳定的 softmax
        # 1. 减去最大值防止 exp 溢出
        max_x = x.max(axis=1, keepdims=True)
        x = x - max_x

        # 2. 计算 exp
        exp_x = exp(x)

        # 3. 计算 softmax
        sum_exp_x = sum(exp_x, axis=1, keepdims=True)
        log_sum_exp_x = log(sum_exp_x)

        # 4. 直接计算 log(softmax) = x - log(sum(exp(x)))
        # 这样可以避免中间结果太大
        log_p = x - log_sum_exp_x

        # 5. 获取目标类别的 log 概率
        batch_indices = np.arange(N)
        class_indices = t.data.flatten()

        # 确保索引正确
        if class_indices.ndim > 1:
            class_indices = class_indices.flatten()

        # 6. 计算交叉熵
        tlog_p = get_item(log_p, (batch_indices, class_indices))
        loss = -sum(tlog_p) / N

        return loss
def relu(x):
    return ReLU()(x)

def clip(x, x_min, x_max):
    x = as_datafield(x)
    return Clip(x_min, x_max)(x)
def exp(x):
    return Exp()(x)
def matmul(x,W):
    return MatMul()(x,W)
def log(x):
    x=as_datafield(x)
    return Log()(x)
def mean_squared_error(x0,x1):
    return MeanSquaredError()(x0,x1)
def linear(x: Datafield, W: Datafield, b= None) -> Datafield:
    return Linear()(x, W, b)
def get_item(x,slices):
    return GetItem(slices)(x)
def sigmoid(x):  #激活函数
    x=as_datafield(x)
    y=1/(1+exp(-x))
    return y
def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)
def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)

def accuracy(y,t):
    y,t=as_datafield(y),as_datafield(t)
    pred=y.data.argmax(axis=1).reshape(t.shape)
    result=(pred==t.data)
    acc=result.mean()
    return Datafield(as_array(acc))
def dropout(x,dropout_ratio=0.5):
    x=as_datafield(x)
    if nodev.Config.train is True:
        xp=cuda.get_array_module(x)
        mask=xp.random.rand(*x.shape)>dropout_ratio
        scale=xp.array(1.0-dropout_ratio).astype(x.dtype)
        y=x*mask/scale
        return y
    else:
        return x


class Argmax(Operation):
    """获取最大值的索引"""

    def __init__(self, axis=None):
        self.axis = axis

    def forward(self, x):
        # 前向传播：直接使用numpy的argmax
        self.x_shape = x.shape
        y = np.argmax(x, axis=self.axis)
        return y

    def backward(self, gy):
        # argmax没有梯度（不可导）
        return None


def argmax(x, axis=None):
    """DeZero风格的argmax函数"""
    return Argmax(axis)(x)

from nodev.operations_conv import conv2d
#from nodev.functions_conv import conv2d
from nodev.operations_conv import pooling
#from nodev.functions_conv import average_pooling