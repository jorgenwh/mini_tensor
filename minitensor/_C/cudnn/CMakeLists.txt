cmake_minimum_required(VERSION 3.16)
project(minitensor_cudnn LANGUAGES CXX CUDA)

set(SOURCE_DIR ".")
include_directories(${SOURCE_DIR})

set(SOURCES 
  #"${SOURCE_DIR}/kernels.cu"
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
#target_link_libraries(${PROJECT_NAME} PUBLIC -lcudnn)
