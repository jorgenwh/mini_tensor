cmake_minimum_required(VERSION 3.16)
project(minitensor_cpp LANGUAGES CXX)

set(SOURCES 
  "ops.cpp"
)

pybind11_add_module(${PROJECT_NAME} ${SOURCES} "bindings.cpp")
