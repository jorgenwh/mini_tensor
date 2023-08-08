import numpy as np
import cupy as cp

from .dtypes import SUPPORTED_DTYPES


class Tensor():
    def __init__(self, data):
        if not isinstance(data, (np.ndarray, cp.ndarray)):
            raise ValueError("data must be a numpy or cupy array")
        if data.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype - only {SUPPORTED_DTYPES} are supported")
        self._data = data

    @property
    def device(self):
        return "cpu" if isinstance(self._data, np.ndarray) else "cuda"

    @device.setter
    def device(self, new_device):
        raise ValueError("cannot change device of tensor")

    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, new_dtype):
        if new_dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype - only {SUPPORTED_DTYPES} are supported")
        self._data = self._data.astype(new_dtype)

    @property
    def shape(self):
        return self._data.shape

    @shape.setter
    def shape(self, new_shape):
        if np.prod(new_shape) != self.size:
            raise ValueError(f"cannot reshape array of size {self.size} into shape {new_shape}")
        self._data = self._data.reshape(new_shape)

    @property
    def size(self):
        return self._data.size

    def reshape(self, *new_shape):
        self.shape = new_shape
        return self

    def ravel(self):
        return Tensor(self._data.ravel())

    def to_cpu(self):
        if self.device == "cpu": return self
        return Tensor(cp.asnumpy(self._data))

    def to_cuda(self):
        if self.device == "cuda": return self
        return Tensor(cp.asarray(self._data))

    def __getitem__(self, index):
        index = index._data if isinstance(index, Tensor) else index
        return Tensor(self._data[index])

    def __setitem__(self, index, value):
        index = index._data if isinstance(index, Tensor) else index
        value = value._data if isinstance(value, Tensor) else value
        self._data[index] = value

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"array({self._data.ravel()}, shape={self.shape}, dtype={self.dtype}, device={self.device})"


