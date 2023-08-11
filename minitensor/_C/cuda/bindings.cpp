#include <pybind11/pybind11.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(minitensor_cuda, m) {
  m.doc() = "minitensor_cuda module";

  m.def("exp", [](const long pin, const long pout, const int size) {
    const float *in = reinterpret_cast<const float *>(pin);
    float *out = reinterpret_cast<float *>(pout);
    ops::exp(in, out, size);
  });

  m.def("relu", [](const long pin, const long pout, const int size) {
    const float *in = reinterpret_cast<const float *>(pin);
    float *out = reinterpret_cast<float *>(pout);
    ops::relu(in, out, size);
  });

  m.def("sum", [](const long pin, const int size) {
    const float *in = reinterpret_cast<const float *>(pin);
    return ops::sum(in, size);
  });
}
