import numpy as np
import cupy as cp

from .tensor import Tensor

def zeros(shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype))

def arange(start, stop, step, dtype=np.float32):
    return Tensor(np.arange(start, stop, step, dtype))

def randn(shape, dtype=np.float32):
    return Tensor(np.random.randn(*shape).astype(dtype))

def from_numpy(numpy_array):
    return Tensor(numpy_array.copy())

def from_cupy(cupy_array):
    return Tensor(cupy_array.copy())
