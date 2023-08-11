import pytest
import numpy as np
import cupy as cp
import minitensor as mt

def test_device():
    np_data = np.arange(10, dtype=np.float32)
    cp_data = cp.arange(10, dtype=np.float32)

    cpu_tensor = mt.from_xp(np_data)
    cuda_tensor = mt.from_xp(cp_data)

    assert cpu_tensor.device == "cpu"
    assert cuda_tensor.device == "cuda"

    assert isinstance(cpu_tensor.as_numpy(), np.ndarray)
    assert isinstance(cuda_tensor.as_numpy(), np.ndarray)
    assert isinstance(cpu_tensor.as_cupy(), cp.ndarray)
    assert isinstance(cuda_tensor.as_cupy(), cp.ndarray)

    assert mt.to_cpu(cpu_tensor).device == "cpu"
    assert mt.to_cpu(cuda_tensor).device == "cpu"
    assert mt.to_cuda(cpu_tensor).device == "cuda"
    assert mt.to_cuda(cuda_tensor).device == "cuda"

    assert cpu_tensor.to_cpu().device == "cpu"
    assert cuda_tensor.to_cpu().device == "cpu"
    assert cpu_tensor.to_cuda().device == "cuda"
    assert cuda_tensor.to_cuda().device == "cuda"

    assert cpu_tensor[0].device == "cpu"
    assert cuda_tensor[0].device == "cuda"
    assert cpu_tensor[2:5].device == "cpu"
    assert cuda_tensor[2:5].device == "cuda"
