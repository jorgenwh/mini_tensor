import time

import numpy as np
import cupy as cp
import minitensor as mt

from .xp_ops import np_relu, cp_relu

def benchmark_op(op, input_arr, n_iter=100, n_warmup=10):
    # warmup
    for _ in range(n_warmup):
        x = op(input_arr)

    # benchmark
    start = time.time()
    for _ in range(n_iter):
        x = op(input_arr)
    end = time.time()

    elapsed = ((end - start) / n_iter) * 1e4
    return elapsed

def compare_op(name, ag_name, op, ag_op, shape, dtype, n_iter=100, n_warmup=10, device="cpu"):
    a = mt.randn(shape, dtype=dtype)
    ag_a = np.random.randn(*shape).astype(dtype)

    if device == "cuda":
        a = mt.to_cuda(a)
        ag_a = cp.asarray(ag_a)

    # minitensor
    elapsed = benchmark_op(op, a, n_iter=n_iter, n_warmup=n_warmup)
    ag_elapsed = benchmark_op(ag_op, ag_a, n_iter=n_iter, n_warmup=n_warmup)

    print(f"{name}: {elapsed:.4f} ms")
    print(f"{ag_name}: {ag_elapsed:.4f} ms")


if __name__ == "__main__":
    for name, ag_name, op, ag_op, device in [
            ("mt.exp", "np.exp", mt.exp, np.exp, "cpu"),
            ("mt.exp", "cp.exp", mt.exp, cp.exp, "cuda"),
            ("mt.sum", "np.sum", mt.sum, np.sum, "cpu"),
            ("mt.relu", "np_relu", mt.relu, np_relu, "cpu"),
            ("mt.relu", "cp_relu", mt.relu, cp_relu, "cuda")
    ]:
        for shape in [(10, 10), (100, 100), (1000, 1000), (2500, 2500)]:
            for dtype in [np.float32]:
                print(f"comparing \33[1m{name}\33[0m against \33[1m{ag_name}\33[0m for: shape=\33[1m{shape}\33[0m, dtype=\33[1m{dtype}\33[0m, device=\33[1m{device}\33[0m")
                compare_op(name, ag_name, mt.exp, np.exp, shape, dtype, device=device)
