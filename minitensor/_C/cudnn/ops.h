#include <cuda_runtime.h>
#include <cudnn.h>

#include "err.h"

namespace ops {

float sum(const float *in, const int size, cudnnHandle_t handle);

} // namespace ops
