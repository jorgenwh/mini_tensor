#include <cuda_runtime.h>

#include "kernels.h"

namespace kernels {

__global__ void exp(const float *in, float *out, const int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = expf(in[i]);
  }
}

__global__ void relu(const float *in, float *out, const int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    out[i] = in[i] > 0.0f ? in[i] : 0.0f;
  }
}

} // namespace kernels
