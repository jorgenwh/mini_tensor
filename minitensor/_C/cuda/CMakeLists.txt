cmake_minimum_required(VERSION 3.16)
project(minitensor_cuda LANGUAGES CXX CUDA)

set(SOURCES 
  "kernels.cu"
  "ops.cu"
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
