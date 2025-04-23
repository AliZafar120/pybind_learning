# Import the base builder class
from builder import TorchCPUOpBuilder
import os
import torch

class TensorOpsBuilder(TorchCPUOpBuilder):
    """Builder for tensor operations extension"""
    
    # Define environment variable to control building this extension
    BUILD_VAR = "BUILD_TENSOR_OPS"
    NAME = "tensor_ops"
    
    def __init__(self):
        super().__init__(name=self.NAME)
    
    def absolute_name(self):
        """Returns the absolute module name"""
        return f'tensor_ops'
    
    def sources(self):
        """Returns list of source files for the extension"""
        sources = ['tensor_ops.cpp']
        
        # Add CUDA source if we're not building for CPU only
        if not self.build_for_cpu:
            sources.append('tensor_ops_cuda.cu')
        
        return sources
    
    def include_paths(self):
        """Returns list of include paths needed to build the extension"""
        import torch
        
        # Get PyTorch include paths
        torch_dir = os.path.dirname(os.path.dirname(torch.__file__))
        torch_include = os.path.join(torch_dir, 'include')
        
        paths = [
            # Include PyTorch headers
            torch_include,
            os.path.join(torch_include, 'torch', 'csrc', 'api', 'include'),
        ]
        
        # Add CUDA headers if not building for CPU only
        if not self.build_for_cpu and torch.cuda.is_available():
            cuda_home = torch.utils.cpp_extension.CUDA_HOME
            if cuda_home:
                paths.append(os.path.join(cuda_home, 'include'))
        
        return paths
    
    def cxx_args(self):
        """Returns list of CXX args for CPU compilation"""
        import sys
        
        # Start with base class args
        args = super().cxx_args()
        
        # Add CUDA define if not building for CPU only
        if not self.build_for_cpu:
            args.append('-DWITH_CUDA')
        
        # Ensure C++17 is used
        if '-std=c++17' not in args:
            if sys.platform == "win32":
                args.append('/std:c++17')
            else:
                args.append('-std=c++17')
        
        return args
    
    def nvcc_args(self):
        """Returns list of NVCC args for CUDA compilation"""
        args = super().nvcc_args()
        
        # Ensure C++17 is used for CUDA
        if '-std=c++17' not in args:
            args.append('-std=c++17')
        
        return args
    
    def libraries(self):
        """Returns list of libraries to link against"""
        if self.build_for_cpu:
            return []
        else:
            # For CUDA, we need to link against CUDA libraries
            return ['cudart', 'cuda']
    
    def library_paths(self):
        """Returns list of library paths"""
        import os
        import torch
        
        if self.build_for_cpu:
            return []
        
        paths = []
        cuda_home = torch.utils.cpp_extension.CUDA_HOME
        if cuda_home:
            lib64_path = os.path.join(cuda_home, 'lib64')
            if os.path.exists(lib64_path):
                paths.append(lib64_path)
            
            lib_path = os.path.join(cuda_home, 'lib')
            if os.path.exists(lib_path):
                paths.append(lib_path)
        
        return paths

# Function to build and load the extension
def build_tensor_ops(verbose=False):
    """Build the tensor ops extension"""
    builder = TensorOpsBuilder()
    return builder.load(verbose=verbose)

if __name__ == "__main__":
    # Build the extension if this script is run directly
    module = build_tensor_ops(verbose=True)
    if module:
        print(f"Successfully built {TensorOpsBuilder.NAME} extension")
    else:
        print(f"Failed to build {TensorOpsBuilder.NAME} extension")