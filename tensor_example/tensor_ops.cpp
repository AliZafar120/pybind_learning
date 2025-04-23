#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

// Simple element-wise operation: adds a constant to each element
torch::Tensor tensor_add_scalar(torch::Tensor input, float scalar) {
    // Check if tensor is on CPU
    TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
    
    // Create output tensor with same properties as input
    torch::Tensor output = torch::empty_like(input);
    
    // Get raw pointers to data
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    // Get total number of elements
    int64_t numel = input.numel();
    
    // Perform the operation
    for (int64_t i = 0; i < numel; ++i) {
        output_data[i] = input_data[i] + scalar;
    }
    
    return output;
}

// Element-wise operation between two tensors: a*x + b*y
torch::Tensor tensor_axpby(torch::Tensor x, torch::Tensor y, float a, float b) {
    // Check if tensors are on CPU
    TORCH_CHECK(x.device().is_cpu(), "Tensor x must be on CPU");
    TORCH_CHECK(y.device().is_cpu(), "Tensor y must be on CPU");
    
    // Check if tensors have the same shape
    TORCH_CHECK(x.sizes() == y.sizes(), "Tensors must have the same shape");
    
    // Create output tensor with same properties as x
    torch::Tensor output = torch::empty_like(x);
    
    // Get raw pointers to data
    float* x_data = x.data_ptr<float>();
    float* y_data = y.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    // Get total number of elements
    int64_t numel = x.numel();
    
    // Perform the operation
    for (int64_t i = 0; i < numel; ++i) {
        output_data[i] = a * x_data[i] + b * y_data[i];
    }
    
    return output;
}

// Vector norm calculation
float tensor_norm(torch::Tensor input) {
    // Check if tensor is on CPU
    TORCH_CHECK(input.device().is_cpu(), "Input tensor must be on CPU");
    
    // Get raw pointer to data
    float* input_data = input.data_ptr<float>();
    
    // Get total number of elements
    int64_t numel = input.numel();
    
    // Calculate squared sum
    float sum_squared = 0.0f;
    for (int64_t i = 0; i < numel; ++i) {
        sum_squared += input_data[i] * input_data[i];
    }
    
    // Return square root of sum
    return std::sqrt(sum_squared);
}

// CUDA version of tensor_add_scalar if CUDA is available
#ifdef WITH_CUDA
torch::Tensor tensor_add_scalar_cuda(torch::Tensor input, float scalar);
#endif

// Function that decides whether to use CPU or CUDA implementation
torch::Tensor tensor_add_scalar_dispatch(torch::Tensor input, float scalar) {
#ifdef WITH_CUDA
    if (input.device().is_cuda()) {
        return tensor_add_scalar_cuda(input, scalar);
    }
#endif
    return tensor_add_scalar(input, scalar);
}

// Define the module
PYBIND11_MODULE(tensor_ops, m) {
    m.doc() = "PyBind11 tensor operations example";
    
    // Register our functions
    m.def("add_scalar", &tensor_add_scalar_dispatch, 
          "Add a scalar value to each element of a tensor",
          py::arg("input"), py::arg("scalar"));
    
    m.def("axpby", &tensor_axpby,
          "Compute a*x + b*y element-wise",
          py::arg("x"), py::arg("y"), py::arg("a"), py::arg("b"));
    
    m.def("norm", &tensor_norm,
          "Compute the L2 norm of a tensor",
          py::arg("input"));
}