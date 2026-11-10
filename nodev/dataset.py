
import numpy as np

from nodev.transforms import Compose, Flatten, ToFloat, Normalize

class Dataset():
    def __init__(self,train=True,transform=None,target_transform=None):
        self.train=train
        self.transform=transform
        self.target_transform=target_transform
        if self.transform is None:
            self.transform=lambda x:x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data=None
        self.label=None
        self.prepare()
    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]),None
        else:
            return self.transform(self.data[index]),\
                   self.target_transform(self.label[index])
    def prepare(self):
        pass
    def __len__(self):
        return len(self.data)

class Spiral(Dataset):
   """ def __init__(self,train=True):
        super().__init__()"""
   def prepare(self):
        self.data,self.label=get_spiral(self.train)

class Bigdata(Dataset):
    def __getitem__(self,index):
        x=np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x,t
    def __len__(self):
        return 1000000
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed)

    num_data = 100
    num_class = 3
    input_dim = 2

    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            r = rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i

            x[ix] = [r*np.sin(theta), r*np.cos(theta)]
            t[ix] = j

    return x, t


class ImageNet(Dataset):
    def __init__(self):
        super().__init__()
    @staticmethod
    def labels():
        path = 'models/labels'
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels
"""class MNIST(Dataset):

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(),
                                     Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):

        if self.train:
            data_path = "./data/train-images.idx3-ubyte"
            label_path = "./data/train-labels.idx1-ubyte"
        else:
            data_path = "./data/t10k-images.idx3-ubyte"
            label_path = "./data/t10k-labels.idx1-ubyte"
        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
"""


class MNIST(Dataset):

    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(),
                                    Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):

        if self.train:
            data_path = "data/train-images.idx3-ubyte"
            label_path = "data/train-labels.idx1-ubyte"
        else:
            data_path = "data/t10k-images.idx3-ubyte"
            label_path = "data/t10k-labels.idx1-ubyte"

        self.data = self._load_images(data_path)
        self.label = self._load_labels(label_path)

    def _load_images(self, path):

        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)

        #data = data.reshape(-1, 784)
        data = data.reshape(-1, 28, 28)
        return data

    def _load_labels(self, path):

        with open(path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape) #生成均匀分布随机数
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]