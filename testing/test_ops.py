import numpy as np
import cupy as cp
import minitensor as mt

def test_exp():
    # cpu
    x = np.random.randn(500).astype(np.float32)
    y = mt.from_numpy(x)
    assert np.allclose(np.exp(x), mt.exp(y).to_numpy())

    # cuda 
    x = cp.random.randn(500).astype(np.float32)
    y = mt.from_cupy(x)
    assert cp.allclose(cp.exp(x), mt.exp(y).to_cupy())

def test_sum():
    # cpu
    x = np.random.randn(500).astype(np.float32)
    y = mt.from_numpy(x)
    np_sum = np.sum(x)
    mt_sum = mt.sum(y)
    assert abs(np_sum - mt_sum) < 1e-4, f"np_sum: {np_sum} != mt_sum: {mt_sum}"

    # cuda 
    x = cp.random.randn(500).astype(np.float32)
    y = mt.from_cupy(x)
    cp_sum = np.sum(x)
    mt_sum = mt.sum(y)
    assert abs(cp_sum - mt_sum) < 1e-4, f"cp_sum: {cp_sum} != mt_sum: {mt_sum}"


if __name__ == "__main__":
    test_exp()
    test_sum()
