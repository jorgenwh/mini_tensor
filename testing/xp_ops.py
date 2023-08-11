import numpy as np
import cupy as cp

def np_relu(numpy_arr):
    return np.maximum(numpy_arr, 0)

def cp_relu(cupy_arr):
    return cp.maximum(cupy_arr, 0)
