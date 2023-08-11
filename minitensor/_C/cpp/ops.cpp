#include <cmath>
#include "ops.h"

namespace ops {

void exp(const float *in, float *out, const int size) {
  for (int i = 0; i < size; i++) {
    out[i] = std::exp(in[i]);
  }
}

void relu(const float *in, float *out, const int size) {
  for (int i = 0; i < size; i++) {
    out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
  }
}

float sum(const float *in, const int size) {
  float s = 0.0f;
  for (int i = 0; i < size; i++) {
    s += in[i];
  }
  return s;
}

} // namespace ops
