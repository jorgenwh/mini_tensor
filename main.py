import numpy as np
import cupy as cp

import minitensor as mt

if __name__ == "__main__":
    x = mt.zeros((2, 3), dtype=mt.float32)
    print(repr(x))

    x.device = "cuda"
    print(x)

    x.shape = (3, 2)
    print(x)
