#pragma once

#include <cuda_runtime.h>

#include "helpers.h"

namespace ops {

void exp(const float *in, float *out, const int size);
void relu(const float *in, float *out, const int size);

float sum(const float *in, const int size);

} // namespace ops
