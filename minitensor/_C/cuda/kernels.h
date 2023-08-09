#pragma once

#include <cuda_runtime.h>

#define THREAD_BLOCK_SIZE 256

namespace kernels {

__global__ void exp(const float *in, float *out, const int size);

} // namespace kernels
