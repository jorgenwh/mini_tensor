#include <pybind11/pybind11.h>

#include "cudnn_handle_wrapper.h"
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(minitensor_cudnn, m) {
  m.doc() = "minitensor_cudnn module";

  m.def("relu", [](const long pin, const long pout, const int size, CuDNNHandleWrapper *handle) {
    const float *in = reinterpret_cast<const float *>(pin);
    float *out = reinterpret_cast<float *>(pout);
    return ops::relu(in, out, size, handle->get());
  });

  m.def("sum", [](const long pin, const int size, CuDNNHandleWrapper *handle) {
    const float *in = reinterpret_cast<const float *>(pin);
    return ops::sum(in, size, handle->get());
  });

  py::class_<CuDNNHandleWrapper>(m, "CuDNNHandleWrapper")
    .def(py::init([]() {
        return new CuDNNHandleWrapper();
    }));
}
