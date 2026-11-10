

import numpy as np
import weakref
from contextlib import contextmanager

import nodev

try:
    import cupy
    array_types=(np.ndarray,cupy.ndarray)
except ImportError:
    array_types=(np.ndarray)
class Config:
    enable_backprop=True
    train=True

class Datafield:
    __array_priority__=200
    def __init__(self,data,name=None):
        if data is not None:
            if not isinstance(data, np.ndarray): #是判断 “x 是否属于类型 T”
                raise  TypeError('{} is not supported'.format(type(data)))
        self.data=data
        self.grad=None
        self.name=name
        self.creator=None
        self.generation=0
    def __sub__(self,other):
        return sub(self,other)
    def __neg__(self):
        return neg(self)
    def __mul__(self,other):
        return mul(self,other)
    def __repr__(self):
        if self.data is None:
            return 'Datafield(None)'
        p=str(self.data).replace('\n','\n'+' '*9)
        return 'Datafield(' + p +')'
    def __truediv__(self, other):
        return div(self,other)
    def __rtruediv__(self, other):
        return rdiv(self,other)
    def __len__(self):  return len(self.data)
    def __add__(self,other):
        return add(self,other)
    def __rmul__(self,other):
        return mul(other,self)

    def __radd__(self,other):
        return add(other, self)
    def __pow__(self, power, modulo=None):
        return pow(self,power)
    def __getitem__(self, slices):
        return nodev.operations.get_item(self, slices)

    def max(self, axis=None, keepdims=False):
        """返回最大值"""
        return nodev.operations.max(self, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        """返回最小值"""
        return nodev.operations.min(self, axis, keepdims)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):
            shape=shape[0]
        return nodev.operations.reshape(self,shape)
    def transpose(self):
        return  nodev.operations.transpose(self)
    @property
    def T(self):
        return nodev.operations.transpose(self)
    @property
    def size(self):
        return self.data.size
    def unchain(self):
        self.creator=None
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def shape(self):  #取出shape
        return self.data.shape
    @property
    def dtype(self):
        return self.data.dtype
    def clear_grad(self):
        self.grad=None
    def to_cpu(self):
        if self.data is not None:
            self.data=nodev.cuda.as_numpy(self.data)
    def to_gpu(self):
        if self.data is not None:
            self.data=nodev.cuda.as_cupy(self.data)
    def set_creator(self,func):
        self.creator=func  #储存当前节点的上一个节点？？
        self.generation=func.generation+1

    def sum(self, axis=None, keepdims=False):
        return nodev.operations.sum(self, axis, keepdims)
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp=nodev.cuda.get_array_module(self.data)
            self.grad=Datafield(xp.ones_like(self.data)) #创建一个全 1 张量，形状和 data 一样
            #self.grad=Datafield(np.ones_like(self.data))
        funcs=[]
        seen_set=set()
        def add_func(f):   #寻找层数最大的因子进行求导
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda  x:x.generation)
        if self.creator is not None:
            add_func(self.creator)
        while funcs:
            f=funcs.pop()  #压出最后一个元素
            gys=[output().grad for output in f.outputs] #汇总导数
            #gys = [output.grad for output in f.outputs]  # 汇总导数
            with using_config('enable_backprop',create_graph):
                gxs=f.backward(*gys)
                if not isinstance(gxs,tuple):
                    gxs=(gxs,)
                for x,gx in zip(f.inputs,gxs):
                    if x.grad is None:
                        x.grad=gx
                    else:
                        x.grad=x.grad+gx
                    if x.creator is not None:
                        #funcs.append(x.creator)
                        add_func(x.creator)
    def unchain_backward(self):
        if self.creator is not None:
            funcs=[self.creator]
            while funcs:
                f=funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        x.unchain()
class Operation:
    def __call__(self,*inputs):
        inputs=[as_datafield(x) for x in inputs]
        xs=[i.data for i in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Datafield(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation=max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs
        self.outputs=[weakref.ref(output) for output in outputs]
        return outputs if len(outputs)>1 else outputs[0]
    def forward(self,*xs):
        raise NotImplementedError()
    def backward(self,*xs):
        raise NotImplementedError()


class Pow(Operation):
    def __init__(self,c):
        self.c=c
    def forward(self,x):
        y=x**self.c
        return y
    def backward(self,gy):
        x, = self.inputs
        c=self.c
        gx=c*x**(c-1)*gy
        return gx
class Div(Operation):
    def forward(self,x0,x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y=x0/x1
        return y
    def backward(self,gy):
        x0,x1=self.inputs

        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        if self.x0_shape!=self.x1_shape:
            gx0=nodev.operations.sum_to(gx0,self.x0_shape)
            gx1 = nodev.operations.sum_to(gx1, self.x1_shape)
        return gx0,gx1


class Add(Operation):
    def forward(self,x0,x1):
        self.x0_shape,self.x1_shape=x0.shape,x1.shape
        y=x0+x1
        return y
    def backward(self,gy):
        gx0,gx1=gy,gy
        if self.x0_shape!=self.x1_shape:
            gx0=nodev.operations.sum_to(gx0,self.x0_shape)
            gx1 = nodev.operations.sum_to(gx1, self.x1_shape)
        return gx0,gx1

class Neg(Operation):
    def forward(self,x):
        return -x
    def backward(self,gy):
        return -gy
class Mul(Operation):
    def forward(self,x0,x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y=x0*x1
        return  y
    def backward(self,gy):
        x0,x1=self.inputs
        gx0,gx1=x1*gy,x0*gy
        if  self.x0_shape!=self.x1_shape:
            gx0=nodev.operations.sum_to(gx0,self.x0_shape)
            gx1 = nodev.operations.sum_to(gx1, self.x1_shape)
        return  gx0,gx1
class Sub(Operation):
    def forward(self,x0,x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y=x0-x1
        return y
    def backward(self,gy):
        gx0,gx1=gy,-gy
        if  self.x0_shape!=self.x1_shape:
            gx0=nodev.operations.sum_to(gx0,self.x0_shape)
            gx1 = nodev.operations.sum_to(gx1, self.x1_shape)
        return gx0,gx1

class Formpara(Datafield):
    pass




@contextmanager
def test_mode():
    old_value = Config.train
    Config.train = False
    try:
        yield
    finally:
        Config.train = old_value

@contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)
def as_datafield(obj): #转换普通数据为variable
    if isinstance(obj, Datafield):
        return obj
    return Datafield(as_array(obj))
def div(x0,x1):
    x1 = as_array(x1, nodev.cuda.get_array_module(x0.data))
    return Div()(x0,x1)
def rdiv(x0,x1):
    x1 = as_array(x1, nodev.cuda.get_array_module(x0.data))
    return Div()(x1,x0)
def sub(x1,x2):
    return Sub()(x1,x2)

def as_array(x,array_module=np):
    """    if x is None:   # ⭐必须加
        return None"""
    if np.isscalar(x):
        return  array_module.array(x)
    return x
def pow(x,c):
    return Pow(c)(x)

def add(x0,x1):
    x1=as_array(x1,nodev.cuda.get_array_module(x0.data))
    return  Add()(x0,x1)
def mul(x0,x1):
    x1 = as_array(x1, nodev.cuda.get_array_module(x0.data))
    return Mul()(x0,x1)
def neg(x):
    return Neg()(x)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)
def no_grad():
    return using_config('enable_backprop',False)

def setup_datafield():
    Datafield.__mul__ = mul
    Datafield.__add__ = add
    Datafield.__rmul__ = mul
    Datafield.__radd__ = add
    Datafield.__neg__ = neg
    Datafield.__pow__ = pow
    Datafield.__truediv__ = div
    Datafield.__rtruediv__ = rdiv
    Datafield.__sub__ = sub
    Datafield.__rsub__ = rsub