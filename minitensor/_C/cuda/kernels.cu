#include <cuda_runtime.h>

#include "kernels.h"

namespace kernels {

__global__ void exp(const float *in, float *out, const int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = expf(in[i]);
  }
}

} // namespace kernels
