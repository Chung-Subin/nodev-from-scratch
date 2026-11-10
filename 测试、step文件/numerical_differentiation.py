
import math
import weakref
import contextlib
import numpy as np

from nodev import as_array


@contextlib.contextmanager
def using_config(name,value):
    old_value=getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

h=0.01
class Config:
    enable_backprop=True
class Variable:
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
        pow(self,power)
    @property
    def size(self):
        return self.data.size
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
    def set_creator(self,func):
        self.creator=func  #储存当前节点的上一个节点？？
        self.generation=func.generation+1
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data) #创建一个全 1 张量，形状和 data 一样
        funcs=[]
        seen_set=set()
        def add_func(f):   #寻找层数最大的因子进行求导
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda  x:x.generation)
        while funcs:
            f=funcs.pop()  #压出最后一个元素
            gys=[output().grad for output in f.outputs] #汇总导数
            #gys = [output.grad for output in f.outputs]  # 汇总导数
            gxs=f.backward(*gys)   #调用了函数f的反向传播
            if not isinstance(gxs,tuple):
                gxs=(gxs,)
            """x,y=f.input,f.output
            x.grad=f.backward(y.grad)"""
            for x,gx in zip(f.inputs,gxs):
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx
                if x.creator is not None:
                    funcs.append(x.creator)

def as_variable(obj): #转换普通数据为variable
    if isinstance(obj,Variable):
        return obj
    return Variable(as_array(obj))


class Function:
    def __call__(self,*inputs):
        inputs=[as_variable(x) for x in inputs]
        xs=[i.data for i in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys,tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]
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





class Pow(Function):
    def __init__(self,c):
        self.c=c
    def forward(self,x):
        y=x**self.c
        return y
    def backward(self,gy):
        x=self.inputs[0].data
        c=self.c
        gx=c*x**(c-1)*gy
class Div(Function):
    def forward(self,x0,x1):
        y=x0/x1
        return y
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        return gx0,gx1
class Square(Function):
    def forward(self,x):
        return x**2
    def backward(self,gy):
        x=self.inputs[0].data
        return 2*x*gy
class Exp(Function):
    def forward(self,x):
        return math.exp(x)
    def backward(self,gy):
        x=self.inputs[0].data2
        return math.exp(x)*gy
class Add(Function):
    def forward(self,x0,x1):
        y=x0+x1
        return y
    def backward(self,gy):
        return gy,gy
class Neg(Function):
    def forward(self,x):
        return -x
    def backward(self,gy):
        return gy
class Mul(Function):
    def forward(self,x0,x1):
        y=x0*x1
        return  y
    def backward(self,gy):
        x0,x1=self.inputs[0].data,self.inputs[1].data
        return  gy*x1,gy*x0
class Sub(Function):
    def forward(self,x0,x1):
        y=x0-x1
        return y
    def backward(self,gy):
        return gy,-gy
def div(x1,x2):
    x1=as_array(x1)
    return Div()(x1,x2)
def rdiv(x1,x2):
    x1 = as_array(x1)
    return Div()(x2,x1)
def sub(x1,x2):
    return Sub()(x1,x2)
def square(x):
    return Square()(x)
def exp(x):
    return Exp()(x)
def as_array(x):
    if np.isscalar(x):
        return  np.array(x)
    return x
def pow(x,c):
    return Pow(c)(x)
def forward(x):
    return square(exp(square(x)))
def add(x0,x1):
    return  Add()(x0,x1)
def mul(x0,x1):
    return Mul()(x0,x1)
def neg(x):
    return Neg()(x)
def rsub(x0,x1):
    x1=as_array(x1)
    return Sub()(x1,x0)
def no_grad():
    return using_config('enable_backprop',False)

Variable.__mul__=mul
Variable.__add__=add
Variable.__rmul__=mul
Variable.__radd__=add
Variable.__neg__=neg
Variable.__pow__=pow
Variable.__truediv__=div
Variable.__rtruediv__=rdiv
Variable.__sub__=sub
Variable.__rsub__=rsub
x0=Variable(np.array(2))
x1=x0**15

print(x1)

"""
A=Square()
B=Exp()
C=Square()
x=Datafield(np.array(0.8))
a=A(x)
b=B(a)
y=C(b)
y.grad=np.array(1.0)
b.grad=C.backward(y.grad)
a.grad=B.backward(b.grad)
x.grad=A.backward(a.grad)
print(x.grad)"""

