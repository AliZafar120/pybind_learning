cmake_minimum_required(VERSION 3.4)
project(pybind_test)

# Find Python and pybind11
add_subdirectory(pybind11)
# Create the binding library
pybind11_add_module(example src/test.cpp)
