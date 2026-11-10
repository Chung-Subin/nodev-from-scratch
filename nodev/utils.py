
import  os
import subprocess
def _dot_func(f):
    txt = ''
    func_name = f.__class__.__name__

    # 函数节点（蓝色）
    txt += '{} [label="{}", color=lightblue, style=filled]\n'.format(
        id(f), func_name
    )

    # 输入 -> 函数
    for x in f.inputs:
        txt += '{} -> {}\n'.format(id(x), id(f))

    # 函数 -> 输出
    for y in f.outputs:
        txt += '{} -> {}\n'.format(id(f), id(y()))

    return txt

def _dot_var(v,verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name=''if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name+=':'
        name+=str(v.shape)+' '+str(v.dtype)
    return dot_var.format(id(v),name)
def max_backward_shape(x_shape, axis):
    if axis is None:
        return (1,) * len(x_shape)

    if not isinstance(axis, tuple):
        axis = (axis,)

    shape = list(x_shape)

    for a in axis:
        shape[a] = 1

    return tuple(shape)
def get_dot_graph(output,verbose=True):
    txt=''
    funcs=[]
    seen_set=set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    add_func(output.creator)
    txt+=_dot_var(output,verbose)
    while funcs:
        func=funcs.pop()
        txt+=_dot_func(func)

        for x in func.inputs:
            txt+=_dot_var(x,verbose)

            if x.creator is not None:
                add_func(x.creator)
    print(txt)
    return 'digraph g {\n' + txt + '}'
def reshape_sum_backward(gy, x_shape, axis, keepdims):
    ndim = len(x_shape)

    if axis is None:
        return gy.reshape((1,) * ndim)

    if not isinstance(axis, tuple):
        axis = (axis,)

    if not keepdims:
        actual_axis = [a if a >= 0 else a + ndim for a in axis]
        shape = list(gy.shape)

        for a in sorted(actual_axis):
            shape.insert(a, 1)

        gy = gy.reshape(shape)

    return gy
def plot_dot_graph(output,verbose=True,to_file='graph.png'):
    dot_graph=get_dot_graph(output,verbose)

    tmp_dir=os.path.join(os.path.expanduser('~'),'.nodev')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path=os.path.join(tmp_dir,'tmp_graph.dot')

    with open(graph_path,'w') as f:
        f.write(dot_graph)

    extension=os.path.splitext(to_file)[1][1:]
    cmd = 'dot -T{} {} -o {}'.format(extension, graph_path, to_file)
    subprocess.run(cmd,shell=True)
    print(cmd)


def sum_to(x, shape):
    """
    将数组 x 沿着广播过的维度求和，使其形状变为 shape
    """
    ndim = len(shape)
    lead = x.ndim - ndim

    # 需要被压缩（求和）的前导轴
    lead_axis = tuple(range(lead))

    # 在 shape 中为 1 的轴，也需要被求和
    axis = tuple(
        i + lead for i, sx in enumerate(shape) if sx == 1
    )

    y = x.sum(lead_axis + axis, keepdims=True)

    # 去掉多余的维度
    if lead > 0:
        y = y.squeeze(lead_axis)

    return y  #补充函数，正文没写


import os
import urllib.request

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def pair(x):
    if isinstance(x,int):
        return (x,x)
    elif isinstance(x,tuple):
        assert len(x)==2
        return x
    else:
        raise ValueError
def get_file(url, file_name=None):
    """Download a file if it is not in the cache."""
    cache_dir = "../data"
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]

    file_path = os.path.join(cache_dir, file_name)

    # 如果缓存目录不存在就创建
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    # 如果文件已经存在直接返回
    if os.path.exists(file_path):
        return file_path

    print("加载: " + file_name)

    try:
        urllib.request.urlretrieve(url, file_path)
    except (Exception, KeyboardInterrupt):
        # 如果下载失败删除不完整文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    print("Done")

    return file_path
def get_conv_outsize(size, k, s, p):
    return (size + 2 * p - k) // s + 1


#print(pair(1),pair((1,4)))