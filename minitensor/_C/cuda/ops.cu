#include <cuda_runtime.h>

#include "helpers.h"
#include "kernels.h"
#include "ops.h"

namespace ops {

void exp(const float *in, float *out, const int size) {
  kernels::exp<<<SDIV(size, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE>>>(in, out, size);
}

void relu(const float *in, float *out, const int size) {
  kernels::relu<<<SDIV(size, THREAD_BLOCK_SIZE), THREAD_BLOCK_SIZE>>>(in, out, size);
}


float sum(const float *in, const int size) {
  return 0.0f;
}

} // namespace ops
