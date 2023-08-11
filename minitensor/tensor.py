import numpy as np
import cupy as cp
from numbers import Number

from .dtypes import SUPPORTED_DTYPES


class Tensor():
    def __init__(self, data):
        if not isinstance(data, (np.ndarray, cp.ndarray, np.float32)):
            raise ValueError("data must be numpy- or cupy-data")
        if data.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype - only {SUPPORTED_DTYPES} are supported")
        self._data = data

    @property
    def device(self):
        if isinstance(self._data, tuple(SUPPORTED_DTYPES)): return "cpu"
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

    def as_numpy(self):
        if self.device == "cpu": return self._data.copy()
        else: return cp.asnumpy(self._data)

    def as_cupy(self):
        if self.device == "cpu": return cp.asarray(self._data)
        else: return self._data.copy()

    def __getitem__(self, index):
        if self.size == 1:
            if index != 0:
                raise IndexError(f"index {index} is out of bounds for axis 0 with size 1")
            return self._data if self.device == "cpu" else self._data[0]

        index = index._data if isinstance(index, Tensor) else index
        value = self._data[index]
        if isinstance(value, Number): return value
        return Tensor(value)

    def __setitem__(self, index, value):
        if isinstance(self._data, tuple(SUPPORTED_DTYPES)):
            if index != 0:
                raise IndexError(f"index {index} is out of bounds for axis 0 with size 1")
            if not isinstance(value, tuple(SUPPORTED_DTYPES) + (Number)):
                raise ValueError(f"cannot convert {type(value)} to {self.dtype}")
            self._data = self._data.dtype(value)
            return

        index = index._data if isinstance(index, Tensor) else index
        value = value._data if isinstance(value, Tensor) else value
        self._data[index] = value

    def __eq__(self, other):
        if isinstance(other, Tensor):
            if self.size != other.size: return False
            if self.device != other.device: return False
            if self.shape != other.shape: return False
            if self.device == "cpu" and self.size == 1: return self._data == other._data
            if self.device == "cpu": return np.array_equal(self._data, other._data)
            else: return cp.array_equal(self._data, other._data)
        if isinstance(other, (Number, tuple(SUPPORTED_DTYPES))):
            if self.size != 1: return False
            if self.device == "cpu": return self._data == other
            else: return self._data[0] == other

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"Tensor({self._data.ravel()}, shape={self.shape}, dtype={self.dtype}, device={self.device})"


