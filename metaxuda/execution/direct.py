"""Direct Numba GPU pipeline execution."""

import numpy as np
from numba import cuda
from ..buffers import GPUMemoryBuffer
from ..streams import GPUStream

DEFAULT_THREADS_PER_BLOCK = 256
DEFAULT_THREADS_2D = (16, 16)
DEFAULT_THREADS_3D = (8, 8, 8)

def _calculate_launch_config(dims, threads_per_block=None):
    """
    Calculate grid and block dimensions for kernel launch.

    Args:
        dims: Single value for 1D, tuple (W, H) for 2D, tuple (W, H, D) for 3D
        threads_per_block: Block dimensions (optional, uses defaults if None)

    Returns:
        (blocks_per_grid, threads_per_block)
    """
    if isinstance(dims, (list, tuple)):
        num_dims = len(dims)
    else:
        num_dims = 1
        dims = [dims]

    match num_dims:
        case 1:
            threads = threads_per_block or DEFAULT_THREADS_PER_BLOCK
            blocks = (dims[0] + threads - 1) // threads
            return blocks, threads
        case 2:
            threads = threads_per_block or DEFAULT_THREADS_2D
            blocks = (
                (dims[0] + threads[0] - 1) // threads[0],
                (dims[1] + threads[1] - 1) // threads[1]
            )
            return blocks, threads
        case _:
            threads = threads_per_block or DEFAULT_THREADS_3D
            blocks = (
                (dims[0] + threads[0] - 1) // threads[0],
                (dims[1] + threads[1] - 1) // threads[1],
                (dims[2] + threads[2] - 1) // threads[2]
            )
            return blocks, threads

def _prepare_inputs(data):
    """Convert various input types to DeviceNDArray list."""
    if isinstance(data, np.ndarray):
        return [cuda.to_device(data)], data.shape, data.dtype

    if isinstance(data, GPUMemoryBuffer):
        return [data.dev_array], data.shape, data.dtype

    if hasattr(data, "shape") and hasattr(data, "dtype") and hasattr(data, "__cuda_array_interface__"):
        return [data], data.shape, data.dtype

    if isinstance(data, (list, tuple)):
        input_arrays = []
        for x in data:
            if isinstance(x, np.ndarray):
                input_arrays.append(cuda.to_device(x))
            elif isinstance(x, GPUMemoryBuffer):
                input_arrays.append(x.dev_array)
            elif hasattr(x, "__cuda_array_interface__"):
                input_arrays.append(x)
            else:
                input_arrays.append(x)

        first = data[0]
        if isinstance(first, GPUMemoryBuffer):
            shape, dtype = first.shape, first.dtype
        elif isinstance(first, np.ndarray):
            shape, dtype = first.shape, first.dtype
        elif hasattr(first, "shape") and hasattr(first, "dtype"):
            shape, dtype = first.shape, first.dtype
        else:
            raise TypeError(f"Cannot determine shape/dtype from {type(first)}")

        return input_arrays, shape, dtype

    raise TypeError("data must be: ndarray, GPUMemoryBuffer, DeviceNDArray, or list")

def run_single_kernel(kernel, data, output=None, stream=None, blocks_per_grid=None, threads_per_block=None, extra_args=None):
    """Execute a single CUDA kernel via direct Numba."""
    input_arrays, shape, dtype = _prepare_inputs(data)
    n = int(np.prod(shape))

    # Auto-calculate grid/block configuration
    if blocks_per_grid is None or threads_per_block is None:
        if extra_args:
            blocks, threads = _calculate_launch_config(extra_args, threads_per_block)
        else:
            blocks, threads = _calculate_launch_config(n, threads_per_block)
    else:
        blocks, threads = blocks_per_grid, threads_per_block

    numba_stream = stream.numba if isinstance(stream, GPUStream) else None

    if output is None:
        output_buf = cuda.device_array(shape, dtype=dtype)
    elif isinstance(output, GPUMemoryBuffer):
        output_buf = output.dev_array
    elif hasattr(output, "shape"):
        output_buf = output
    else:
        raise TypeError("output must be GPUMemoryBuffer or DeviceNDArray")

    # Build kernel arguments
    kernel_args = list(input_arrays) + [output_buf]
    if extra_args:
        kernel_args.extend(extra_args)

    if numba_stream:
        kernel[blocks, threads, numba_stream](*kernel_args)
        numba_stream.synchronize()
    else:
        kernel[blocks, threads](*kernel_args)
        cuda.synchronize()

    return None if output else GPUMemoryBuffer.from_dev_array(output_buf)

def run_pipeline(kernels, data, output=None, stream=None, blocks_per_grid=None, threads_per_block=None):
    """Execute a pipeline of chained CUDA kernels via direct Numba."""
    input_arrays, shape, dtype = _prepare_inputs(data)
    n = int(np.prod(shape))

    if blocks_per_grid is None or threads_per_block is None:
        blocks, threads = _calculate_launch_config(n, threads_per_block)
    else:
        blocks, threads = blocks_per_grid, threads_per_block

    numba_stream = stream.numba if isinstance(stream, GPUStream) else None

    temp_buf = cuda.device_array(shape, dtype=dtype)
    if numba_stream:
        kernels[0][blocks, threads, numba_stream](*input_arrays, temp_buf)
    else:
        kernels[0][blocks, threads](*input_arrays, temp_buf)

    current = temp_buf
    for i, kernel in enumerate(kernels[1:], start=1):
        is_last = (i == len(kernels) - 1)

        if is_last and output:
            next_buf = output.dev_array if isinstance(output, GPUMemoryBuffer) else output
        else:
            next_buf = cuda.device_array(shape, dtype=dtype)

        if numba_stream:
            kernel[blocks, threads, numba_stream](current, next_buf)
        else:
            kernel[blocks, threads](current, next_buf)

        current = next_buf

    return None if output else GPUMemoryBuffer.from_dev_array(current)

class Pipeline:
    """Wrapper for executing single or chained CUDA kernels."""

    def __init__(self, *kernels):
        """Initialize pipeline with one or more kernels."""
        if not kernels:
            raise ValueError("Pipeline requires at least one kernel")
        self.kernels = list(kernels)

    def __call__(self, data, output=None, stream=None, blocks_per_grid=None, threads_per_block=None, extra_args=None):
        """Execute the pipeline."""
        if len(self.kernels) == 1:
            return run_single_kernel(self.kernels[0], data, output=output, stream=stream,
                                   blocks_per_grid=blocks_per_grid, threads_per_block=threads_per_block,
                                   extra_args=extra_args)
        return run_pipeline(self.kernels, data, output=output, stream=stream,
                          blocks_per_grid=blocks_per_grid, threads_per_block=threads_per_block)