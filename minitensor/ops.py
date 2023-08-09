import numpy as np
import cupy as cp

from .tensor import Tensor

import minitensor_cpp
import minitensor_cuda
import minitensor_cudnn

cudnn_handle = minitensor_cudnn.CuDNNHandleWrapper()

def exp(tensor):
    if tensor.device == "cpu":
        return Tensor(minitensor_cpp.exp(tensor._data))
    else:
        res = cp.empty_like(tensor._data)
        minitensor_cuda.exp(tensor._data.data.ptr, res.data.ptr, res.size)
        return Tensor(res)

def sum(tensor):
    if tensor.device == "cpu":
        return minitensor_cpp.sum(tensor._data)
    else:
        return minitensor_cudnn.sum(tensor._data.data.ptr, tensor._data.size, cudnn_handle)
