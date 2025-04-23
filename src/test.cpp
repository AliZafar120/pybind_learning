#include <pybind11/pybind11.h>

namespace py = pybind11;

// A simple C++ function to be exposed to Python
int add(int a, int b) {
    return a + b;
}

// The binding code
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring
    
    // Add the function to the module
    m.def("add", &add, "A function that adds two numbers",
          py::arg("a") = 1, py::arg("b") = 2); // Default arguments
}