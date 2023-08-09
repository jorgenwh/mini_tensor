cmake_minimum_required(VERSION 3.16)
project(minitensor_cudnn LANGUAGES CXX CUDA)

set(SOURCES 
  "cudnn_handle_wrapper.cu"
  "ops.cu"
)

#include_directories(/usr/local/cuda-11.7/include)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC -lcudnn)
