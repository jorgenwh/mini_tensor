#include <cuda_runtime.h>
#include <cudnn.h>

#include "err.h"

namespace ops {

float relu(const float *in, float *out, const int size, const cudnnHandle_t &handle);
float sum(const float *in, const int size, const cudnnHandle_t &handle);

} // namespace ops
