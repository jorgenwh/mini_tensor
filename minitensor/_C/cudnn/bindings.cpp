#include <inttypes.h>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(minitensor_cudnn, m) {
  m.doc() = "minitensor_cudnn module";
}
