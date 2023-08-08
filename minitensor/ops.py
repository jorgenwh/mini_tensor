import numpy as np
import cupy as cp

def exp(tensor):
    if tensor.device == "cpu":
        return Tensor(np.exp(tensor._data))
    else:
        return Tensor(cp.exp(tensor._data))

