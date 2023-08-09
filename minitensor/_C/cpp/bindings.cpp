#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(minitensor_cpp, m) {
  m.doc() = "minitensor_cpp module";

  m.def("exp", [](py::array_t<float> &in) {
    const int size = in.size();
    const float *in_buffer = in.data();
    auto out = py::array_t<float>(size);
    float *out_buffer = out.mutable_data();
    ops::exp(in_buffer, out_buffer, size);
    return out;
  });

  m.def("sum", [](py::array_t<float> &in) {
    const int size = in.size();
    const float *in_buffer = in.data();
    return ops::sum(in_buffer, size);
  });
}
