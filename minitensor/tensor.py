import numpy as np
import cupy as cp


class Tensor():
    def __init__(self, data):
        if not isinstance(data, (np.ndarray, cp.ndarray)):
            raise ValueError("data must be a numpy or cupy array")
        self._data = data

    @property
    def device(self):
        if isinstance(self._data, np.ndarray):
            return "cpu"
        else:
            return "cuda"

    @device.setter
    def device(self, new_device):
        if new_device == "cpu":
            self._data = cp.asnumpy(self._data)
        elif new_device == "cuda":
            self._data = cp.asarray(self._data)
        else:
            raise ValueError(f"unknown device: {new_device}")

    @property
    def dtype(self):
        return self._data.dtype

    @dtype.setter
    def dtype(self, new_dtype):
        if new_dtype != np.float32:
            raise ValueError("only float32 is currently supported")
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

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"array({self._data.ravel()}, shape={self.shape}, dtype={self.dtype}, device={self.device})"


