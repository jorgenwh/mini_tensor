#include <inttypes.h>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(minitensor_cuda, m) {
  m.doc() = "minitensor_cuda module";
}
