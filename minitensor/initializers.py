import numpy as np
import cupy as cp

from .tensor import Tensor

def zeros(shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype))

def arange(start, stop, step, dtype=np.float32):
    return Tensor(np.arange(start, stop, step, dtype))
