"""Pipeline pool for pure Rust execution."""

import sys
import numpy as np
from numba import cuda
from pathlib import Path

NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"
sys.path.insert(0, str(NATIVE_DIR))

try:
    import cuda_pipeline
except ImportError as e:
    raise ImportError(
        f"Could not import cuda_pipeline from {NATIVE_DIR}. "
        f"Make sure cuda_pipeline.*.so is in the native/ directory. Error: {e}"
    )

from ..buffers import ManagedGPUBuffer

def _get_device_pointer(arr):
    """Extract device pointer from ManagedGPUBuffer."""
    if isinstance(arr, ManagedGPUBuffer):
        return arr.device_ptr
    elif hasattr(arr, '__cuda_array_interface__'):
        return arr.__cuda_array_interface__['data'][0]
    else:
        raise TypeError(f"Cannot extract device pointer from {type(arr)}. Use ManagedGPUBuffer.")

class PipelinePool:
    """Pure Rust pipeline execution pool."""

    def __init__(self):
        self._registered_kernels = set()

    def run(self, kernels, *arrays, output=None, extra_args=None):
        """
        Convenience method: run kernels directly on numpy arrays.

        Automatically creates ManagedGPUBuffer, executes, and returns results.

        Args:
            kernels: Single kernel or list of kernels
            *arrays: Input numpy arrays (uploaded to ManagedGPUBuffer)
            output: Optional output numpy array or ManagedGPUBuffer
            extra_args: Optional extra scalar arguments

        Returns:
            Numpy array with results (or None if output buffer provided)
        """
        # Convert numpy arrays to ManagedGPUBuffer
        input_buffers = []
        input_was_array = []

        for arr in arrays:
            if isinstance(arr, np.ndarray):
                buf = ManagedGPUBuffer(arr=arr)
                input_buffers.append(buf)
                input_was_array.append(True)
            elif isinstance(arr, ManagedGPUBuffer):
                input_buffers.append(arr)
                input_was_array.append(False)
            else:
                raise TypeError(f"Input must be numpy array or ManagedGPUBuffer, got {type(arr)}")

        # Handle output
        if output is None:
            # Create output buffer with same shape/dtype as first input
            first_array = arrays[0]
            if isinstance(first_array, np.ndarray):
                output_buf = ManagedGPUBuffer(length=first_array.size, dtype=first_array.dtype, shape=first_array.shape)
            else:
                output_buf = ManagedGPUBuffer(length=first_array.length, dtype=first_array.dtype, shape=first_array.shape)
            output_buf.zeros()
            should_download = True
            should_free_output = True
        elif isinstance(output, np.ndarray):
            # User provided numpy array - create buffer from it
            output_buf = ManagedGPUBuffer(arr=output)
            should_download = True
            should_free_output = True
        elif isinstance(output, ManagedGPUBuffer):
            # User provided buffer
            output_buf = output
            should_download = False
            should_free_output = False
        else:
            raise TypeError(f"Output must be numpy array or ManagedGPUBuffer, got {type(output)}")

        try:
            # Execute via submit
            self.submit(kernels, input_buffers, output=output_buf, extra_args=extra_args)

            # Download result if needed
            if should_download:
                result = output_buf.download()
            else:
                result = None

        finally:
            # Clean up temporary buffers
            for i, (buf, was_array) in enumerate(zip(input_buffers, input_was_array)):
                if was_array:
                    buf.free()

            if should_free_output:
                output_buf.free()

        return result

    def submit(self, kernels, data, output=None, extra_args=None):
        """
        Submit pipeline for pure Rust execution.

        Args:
            kernels: Numba CUDA kernels (Rust extracts PTX by name)
            data: List of ManagedGPUBuffer
            output: ManagedGPUBuffer output buffer
        """
        if not isinstance(kernels, (list, tuple)):
            kernels = [kernels]

        # Normalize data
        if not isinstance(data, (list, tuple)):
            data_list = [data]
        else:
            data_list = data

        if output is None:
            first = data_list[0]
            output_buf = ManagedGPUBuffer(length=first.length, dtype=first.dtype, shape=first.shape)
            output_buf.zeros()
        else:
            output_buf = output

        # Get device pointers for Rust launch
        input_ptrs = [_get_device_pointer(d) for d in data_list]
        output_ptr = _get_device_pointer(output_buf)
        args = input_ptrs + [output_ptr]

        try:
            for kernel in kernels:
                kernel_name = kernel.py_func.__name__

                cuda_pipeline.launch_kernel(kernel_name, args)
                self._registered_kernels.add(kernel_name)

            cuda.synchronize()

        finally:
            pass

        return None if output else output_buf

    def __call__(self, kernels, data, output=None, extra_args=None):
        return self.submit(kernels, data, output, extra_args)