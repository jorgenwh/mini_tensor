import time

import numpy as np
import cupy as cp
import minitensor as mt

def benchmark_op(op, shape, dtype, n_iter=100, n_warmup=10, device="cpu"):
    if device == "cpu":
        a = mt.tensor(np.random.randn(*shape).astype(dtype))
    elif device == "gpu":
        a = mt.tensor(cp.random.randn(*shape).astype(dtype))
    else:
        raise ValueError("Invalid device.")

    # Warmup
    for _ in range(n_warmup):
        op(a)

    # Benchmark
    start = time.time()
    for _ in range(n_iter):
        op(a)
    end = time.time()

    return (end - start) / n_iterhhh


if __name__ == "__main__":
    pass
