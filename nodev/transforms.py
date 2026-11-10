import numpy as np


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, x):
        if not self.transforms:
            return x
        for t in self.transforms:
            x = t(x)
        return x


class Flatten:
    """Flatten ndarray."""

    def __call__(self, x):
        return x.reshape(-1)


class ToFloat:
    """Convert ndarray dtype to float32."""

    def __call__(self, x):
        return x.astype(np.float32)


class Normalize:
    """Normalize ndarray."""

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        mean = self.mean
        std = self.std

        if not np.isscalar(mean):
            mean = np.array(mean, dtype=x.dtype)
        if not np.isscalar(std):
            std = np.array(std, dtype=x.dtype)

        return (x - mean) / std


class AsType:
    """Convert ndarray dtype."""

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, x):
        return x.astype(self.dtype)


class ToInt:
    """Convert ndarray dtype to int."""

    def __call__(self, x):
        return x.astype(np.int32)