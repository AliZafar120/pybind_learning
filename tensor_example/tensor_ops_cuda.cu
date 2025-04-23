#include <torch/extension.h>

// CUDA kernel for adding scalar to tensor
__global__ void add_scalar_kernel(float* input, float* output, float scalar, int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        output[idx] = input[idx] + scalar;
    }
}

// CUDA implementation of tensor_add_scalar
torch::Tensor tensor_add_scalar_cuda(torch::Tensor input, float scalar) {
    // Check if tensor is on CUDA
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    
    // Create output tensor with same properties as input
    torch::Tensor output = torch::empty_like(input);
    
    // Get total number of elements
    int64_t numel = input.numel();
    
    // Calculate grid and block dimensions
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    
    // Get raw pointers to data
    float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    // Launch kernel
    add_scalar_kernel<<<blocks, threads>>>(input_data, output_data, scalar, numel);
    
    // Synchronize and check for errors
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel execution failed");
    
    return output;
}