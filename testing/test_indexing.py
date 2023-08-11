import pytest
import numpy as np
import cupy as cp
import minitensor as mt

def test_indexing_cpu():
    tensor = mt.arange(start=0, stop=12, step=1, dtype=np.float32)

    assert tensor[0] == 0
    assert tensor[-1] == 11
    assert tensor[5] == 5

    np.testing.assert_array_equal(tensor[0:5].as_numpy(), np.arange(0, 5, 1, dtype=np.float32))

    tensor = tensor.reshape(3, 4)

    assert tensor[0,0] == 0
    assert tensor[-1,0] == 8
    assert tensor[2,3] == 11

    np.testing.assert_array_equal(tensor[0].as_numpy(), np.arange(0, 4, 1, dtype=np.float32))
    np.testing.assert_array_equal(tensor[:,1:3].as_numpy(), np.array([1, 2, 5, 6, 9, 10], dtype=np.float32).reshape(3, 2))

def test_indexing_cuda():
    tensor = mt.arange(start=0, stop=12, step=1, dtype=np.float32)
    tensor = tensor.to_cuda()
    print(repr(tensor))

    assert tensor[0] == 0
    assert tensor[-1] == 11
    assert tensor[5] == 5

    cp.testing.assert_array_equal(tensor[0:5].as_cupy(), cp.arange(0, 5, 1, dtype=np.float32))

    tensor = tensor.reshape(3, 4)

    assert tensor[0,0] == 0
    assert tensor[-1,0] == 8
    assert tensor[2,3] == 11

    cp.testing.assert_array_equal(tensor[0].as_cupy(), cp.arange(0, 4, 1, dtype=np.float32))
    cp.testing.assert_array_equal(tensor[:,1:3].as_cupy(), cp.array([1, 2, 5, 6, 9, 10], dtype=np.float32).reshape(3, 2))
