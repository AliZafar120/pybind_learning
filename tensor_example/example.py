#!/usr/bin/env python3
"""
Example demonstrating how to use the tensor_ops C++ extension
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Try to import or build the extension
try:
    from tensor_ops_builder import build_tensor_ops
    tensor_ops = build_tensor_ops(verbose=True)
    print("Successfully loaded tensor_ops extension!")
except ImportError as e:
    print(f"Failed to load tensor_ops extension: {e}")
    exit(1)

def run_basic_example():
    """Demonstrate basic tensor operations"""
    print("\n=== Basic Tensor Operations ===")
    
    # Create sample tensors
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    
    print(f"x = {x}")
    print(f"y = {y}")
    
    # Test add_scalar operation
    scalar = 10.0
    result = tensor_ops.add_scalar(x, scalar)
    expected = x + scalar
    
    print(f"x + {scalar} = {result}")
    print(f"PyTorch result: {expected}")
    print(f"Results match: {torch.allclose(result, expected)}")
    
    # Test axpby operation
    a, b = 2.0, 3.0
    result = tensor_ops.axpby(x, y, a, b)
    expected = a * x + b * y
    
    print(f"{a} * x + {b} * y = {result}")
    print(f"PyTorch result: {expected}")
    print(f"Results match: {torch.allclose(result, expected)}")
    
    # Test norm operation
    norm_result = tensor_ops.norm(x)
    expected_norm = torch.norm(x).item()
    
    print(f"Norm of x = {norm_result}")
    print(f"PyTorch result: {expected_norm}")
    print(f"Results match: {abs(norm_result - expected_norm) < 1e-5}")

def run_cuda_example():
    """Test CUDA implementation if available"""
    if not torch.cuda.is_available():
        print("\n=== CUDA not available, skipping CUDA example ===")
        return
    
    print("\n=== CUDA Tensor Operations ===")
    
    # Create sample tensors on GPU
    x_cuda = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).cuda()
    
    print(f"x_cuda = {x_cuda}")
    
    # Test add_scalar operation on CUDA
    scalar = 10.0
    result = tensor_ops.add_scalar(x_cuda, scalar)
    expected = x_cuda + scalar
    
    print(f"x_cuda + {scalar} = {result}")
    print(f"PyTorch result: {expected}")
    print(f"Results match: {torch.allclose(result, expected)}")
    print(f"Result is on CUDA: {result.is_cuda}")

def benchmark_performance():
    """Compare performance between C++ extension and PyTorch"""
    print("\n=== Performance Benchmark ===")
    
    sizes = [1000, 10000, 100000, 1000000]
    cpp_times = []
    torch_times = []
    
    for size in sizes:
        # Create random tensors
        x = torch.rand(size, dtype=torch.float32)
        y = torch.rand(size, dtype=torch.float32)
        a, b = 2.0, 3.0
        
        # Warm up
        _ = tensor_ops.axpby(x, y, a, b)
        _ = a * x + b * y
        
        # Benchmark C++ extension
        start_time = time.time()
        for _ in range(100):
            _ = tensor_ops.axpby(x, y, a, b)
        cpp_time = (time.time() - start_time) / 100
        cpp_times.append(cpp_time)
        
        # Benchmark PyTorch
        start_time = time.time()
        for _ in range(100):
            _ = a * x + b * y
        torch_time = (time.time() - start_time) / 100
        torch_times.append(torch_time)
        
        print(f"Size: {size}")
        print(f"  C++ extension time: {cpp_time:.6f} seconds")
        print(f"  PyTorch time: {torch_time:.6f} seconds")
        print(f"  Speedup: {torch_time/cpp_time:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpp_times, 'o-', label='C++ Extension')
    plt.plot(sizes, torch_times, 'o-', label='PyTorch')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tensor Size')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: C++ Extension vs. PyTorch')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    print("Performance plot saved as 'performance_comparison.png'")

def main():
    """Main function running all examples"""
    print("=== Tensor Ops C++ Extension Demo ===")
    
    run_basic_example()
    run_cuda_example()
    benchmark_performance()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()