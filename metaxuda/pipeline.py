import numpy as np
from numba import cuda
from .buffer import GPUMemoryBuffer

def _launch_config(shape, threads_per_block=256):
    """
    Choose CUDA launch parameters automatically.
    Uses a 1D grid for any shape by flattening total elements.
    """
    n = int(np.prod(shape))
    tpb = min(threads_per_block, n)
    blocks = (n + tpb - 1) // tpb
    return (blocks,), (tpb,)

def run_pipeline(kernels, data, stream=None):
    """
    Run one or more CUDA kernels in sequence on the given data.
    Supports NumPy arrays, GPUMemoryBuffer objects, or lists/tuples of them.
    """
    if isinstance(data, GPUMemoryBuffer):
        d_in = data.dev_array
        shape = data.shape
    elif isinstance(data, np.ndarray):
        d_in = cuda.to_device(data)
        shape = data.shape
    elif isinstance(data, (list, tuple)):
        d_in = [
            buf.dev_array if isinstance(buf, GPUMemoryBuffer) else cuda.to_device(buf)
            for buf in data
        ]
        shape = data[0].shape if isinstance(data[0], GPUMemoryBuffer) else data[0].shape
    else:
        raise TypeError(
            "data must be np.ndarray, GPUMemoryBuffer, or list/tuple thereof"
        )

    blocks_per_grid, threads_per_block = _launch_config(shape)
    dtype = d_in[0].dtype if isinstance(d_in, list) else d_in.dtype
    n = int(np.prod(shape))

    if not kernels:
        return GPUMemoryBuffer.from_dev_array(
            d_in if not isinstance(d_in, list) else d_in[0]
        )

    if len(kernels) == 1:
        output_buf = cuda.device_array(shape, dtype=dtype)
        kernel = kernels[0]

        if isinstance(d_in, list):
            args = d_in.copy() + [output_buf]
        else:
            args = [d_in, output_buf]

        if kernel.py_func.__code__.co_argcount > len(args):
            args.append(n)

        if stream:
            kernel[blocks_per_grid, threads_per_block, stream.numba](*args)
        else:
            kernel[blocks_per_grid, threads_per_block](*args)

        return GPUMemoryBuffer.from_dev_array(output_buf)

    buf_a = cuda.device_array(shape, dtype=dtype)
    buf_b = cuda.device_array(shape, dtype=dtype)

    kernel = kernels[0]
    if isinstance(d_in, list):
        args = d_in.copy() + [buf_a]
    else:
        args = [d_in, buf_a]

    if kernel.py_func.__code__.co_argcount > len(args):
        args.append(n)

    if stream:
        kernel[blocks_per_grid, threads_per_block, stream.numba](*args)
    else:
        kernel[blocks_per_grid, threads_per_block](*args)

    current_in = buf_a
    current_out = buf_b

    for i, kernel in enumerate(kernels[1:], 1):
        args = [current_in, current_out]

        if kernel.py_func.__code__.co_argcount > len(args):
            args.append(n)

        if stream:
            kernel[blocks_per_grid, threads_per_block, stream.numba](*args)
        else:
            kernel[blocks_per_grid, threads_per_block](*args)

        if i < len(kernels) - 1:
            current_in, current_out = current_out, current_in

    return GPUMemoryBuffer.from_dev_array(current_out)