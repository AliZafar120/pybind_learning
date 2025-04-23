import os
import sys
import torch
import importlib
from abc import ABC, abstractmethod

# Simple implementation inspired by DeepSpeed's OpBuilder system
class OpBuilder(ABC):
    """Base class for all operation builders"""
    
    # Class cache for loaded ops
    _loaded_ops = {}
    
    def __init__(self, name):
        self.name = name
        self.jit_mode = False
        self.build_for_cpu = False
    
    @abstractmethod
    def absolute_name(self):
        """Returns absolute build path for cases where the op is pre-installed"""
        pass
    
    @abstractmethod
    def sources(self):
        """Returns list of source files for your op"""
        pass
    
    def include_paths(self):
        """Returns list of include paths needed to build the extension"""
        return []
    
    def nvcc_args(self):
        """Returns list of NVCC args for CUDA compilation"""
        return []
    
    def cxx_args(self):
        """Returns list of CXX args for CPU compilation"""
        return []
    
    def library_paths(self):
        """Returns list of library paths"""
        return []
    
    def libraries(self):
        """Returns list of libraries to link against"""
        return []
    
    def extra_ldflags(self):
        """Returns list of extra linker flags"""
        return []
    
    def strip_empty_entries(self, args):
        """Drop any empty strings from the list of compile and link flags"""
        return [x for x in args if len(x) > 0]
    
    def load(self, verbose=True):
        """Loads the extension, either from pre-built or JIT compiled"""
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]
        
        try:
            # Try to import the module directly
            op_module = importlib.import_module(self.absolute_name())
            __class__._loaded_ops[self.name] = op_module
            if verbose:
                print(f"Loaded pre-built extension: {self.name}")
            return op_module
        except (ImportError, ModuleNotFoundError):
            # If import fails, build the extension
            return self.jit_load(verbose)
    
    def jit_load(self, verbose=True):
        """Build and load the extension just-in-time"""
        try:
            # Check if ninja is available
            import ninja
        except ImportError:
            raise ImportError("Ninja is required for JIT compilation. Install it with 'pip install ninja'.")
        
        self.jit_mode = True
        from torch.utils.cpp_extension import load
        
        # Set build_for_cpu if CUDA is not available
        if isinstance(self, CUDAOpBuilder):
            self.build_for_cpu = not torch.cuda.is_available()
        
        # Get sources, include paths, and other build info
        sources = [os.path.abspath(path) for path in self.sources()]
        include_paths = [os.path.abspath(path) for path in self.include_paths()]
        
        # Define compiler args
        cxx_args = self.strip_empty_entries(self.cxx_args())
        nvcc_args = self.strip_empty_entries(self.nvcc_args())
        extra_ldflags = self.strip_empty_entries(self.extra_ldflags())
        
        if verbose:
            print(f"JIT compiling {self.name} extension...")
            print(f"Sources: {sources}")
            print(f"Include paths: {include_paths}")
            print(f"C++ args: {cxx_args}")
            print(f"NVCC args: {nvcc_args}")
            print(f"Extra LD flags: {extra_ldflags}")
        
        # Load the module
        op_module = load(
            name=self.name,
            sources=sources,
            extra_include_paths=include_paths,
            extra_cflags=cxx_args,
            extra_cuda_cflags=nvcc_args,
            extra_ldflags=extra_ldflags,
            verbose=verbose,
            with_cuda=(isinstance(self, CUDAOpBuilder) and not self.build_for_cpu)
        )
        
        __class__._loaded_ops[self.name] = op_module
        return op_module
    
    def builder(self):
        """Creates a PyTorch extension builder object"""
        from torch.utils.cpp_extension import CppExtension
        
        # Get include paths and sources
        include_dirs = [os.path.abspath(path) for path in self.include_paths()]
        sources = [os.path.abspath(path) for path in self.sources()]
        
        # Create the extension
        return CppExtension(
            name=self.absolute_name(),
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': self.cxx_args()},
            extra_link_args=self.extra_ldflags()
        )

class CUDAOpBuilder(OpBuilder):
    """Base class for operations that can use CUDA"""
    
    def compute_capability_args(self):
        """Returns NVCC compute capability flags"""
        args = []
        
        if torch.cuda.is_available():
            # For JIT mode, compile for actual device capabilities
            if self.jit_mode:
                for i in range(torch.cuda.device_count()):
                    cc_major, cc_minor = torch.cuda.get_device_capability(i)
                    cc = f"{cc_major}{cc_minor}"
                    args.append(f'-gencode=arch=compute_{cc},code=sm_{cc}')
            
            # Otherwise, use a default set of capabilities
            else:
                # Default compute capabilities
                for cc in ["60", "61", "70", "75", "80", "86"]:
                    args.append(f'-gencode=arch=compute_{cc},code=sm_{cc}')
        
        return args
    
    def nvcc_args(self):
        """Returns list of NVCC args for CUDA compilation"""
        args = ['-O3']
        
        # Standard flags for CUDA compatibility
        args += [
            '-std=c++17',  # Changed from c++14 to c++17
            '--use_fast_math',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__'
        ]
        
        # Add compute capability flags
        args += self.compute_capability_args()
        
        return args
    
    def builder(self):
        """Creates a PyTorch CUDA extension builder object"""
        if self.build_for_cpu:
            from torch.utils.cpp_extension import CppExtension as ExtBuilder
        else:
            from torch.utils.cpp_extension import CUDAExtension as ExtBuilder
        
        # Get include paths and sources
        include_dirs = [os.path.abspath(path) for path in self.include_paths()]
        sources = [os.path.abspath(path) for path in self.sources()]
        
        # Define compile args
        compile_args = {
            'cxx': self.cxx_args()
        }
        
        if not self.build_for_cpu:
            compile_args['nvcc'] = self.nvcc_args()
        
        # Create the extension
        return ExtBuilder(
            name=self.absolute_name(),
            sources=sources,
            include_dirs=include_dirs,
            libraries=self.libraries(),
            library_dirs=self.library_paths(),
            extra_compile_args=compile_args,
            extra_link_args=self.extra_ldflags()
        )

class TorchCPUOpBuilder(CUDAOpBuilder):
    """Builder for operations that can work on both CPU and CUDA"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def cxx_args(self):
        """Returns list of CXX args for compilation"""
        if sys.platform == "win32":
            args = ['-O2', '/std:c++17']  # Changed from c++14 to c++17 for Windows
        else:
            args = ['-O3', '-std=c++17', '-g']  # Changed from c++14 to c++17
        
        # Add CPU-specific optimization flags
        if sys.platform != "win32":
            args.append('-march=native')
            args.append('-fopenmp')
        
        return args
    
    def extra_ldflags(self):
        """Returns list of extra linker flags"""
        if self.build_for_cpu:
            return ['-fopenmp'] if sys.platform != "win32" else []
        
        if sys.platform == "win32":
            return []
        else:
            return ['-fopenmp']