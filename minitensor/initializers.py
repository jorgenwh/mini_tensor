import numpy as np
import cupy as cp

from .tensor import Tensor

def zeros(shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype))

def arange(start, stop, step, dtype=np.float32):
    return Tensor(np.arange(start, stop, step, dtype))

def randn(shape, dtype=np.float32):
    return Tensor(np.random.randn(*shape).astype(dtype))

def from_xp(array):
    if not isinstance(array, (np.ndarray, cp.ndarray)):
        raise ValueError("array must be numpy.ndarray or cupy.ndarray")
    return Tensor(array.copy())
