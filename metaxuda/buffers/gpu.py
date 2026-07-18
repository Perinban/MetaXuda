"""GPU buffer using Numba."""

import numpy as np
from numba import cuda
from .base import BufferBase


class GPUMemoryBuffer(BufferBase):
    """GPU buffer wrapper using Numba DeviceNDArray for standard GPU operations."""

    def __init__(self, arr: np.ndarray = None, length: int = None,
                 dtype=np.float32, shape=None):
        if arr is not None:
            self.dtype = arr.dtype
            self.shape = arr.shape
            self.length = arr.size
            self.size = arr.nbytes
            self.dev_array = cuda.to_device(arr)
        elif length is not None:
            self.dtype = np.dtype(dtype)
            self.shape = shape if shape else (length,)
            self.length = length
            self.size = self.length * self.dtype.itemsize
            self.dev_array = cuda.device_array(self.shape, dtype=self.dtype)
        else:
            raise ValueError("Must provide either arr or length+dtype")

    @classmethod
    def from_dev_array(cls, dev_array):
        """Create GPUMemoryBuffer from existing Numba DeviceNDArray."""
        buf = cls.__new__(cls)
        buf.dev_array = dev_array
        buf.dtype = dev_array.dtype
        buf.shape = dev_array.shape
        buf.length = dev_array.size
        buf.size = dev_array.nbytes
        return buf

    @property
    def device_ptr(self):
        """Get device pointer for kernel launches."""
        return self.dev_array.__cuda_array_interface__['data'][0]

    def upload(self, arr: np.ndarray, stream=None):
        if arr.shape != self.shape:
            raise ValueError(f"Shape mismatch: {arr.shape} vs {self.shape}")

        if stream:
            self.dev_array.copy_to_device(arr, stream=stream.numba)
            stream.sync()
        else:
            self.dev_array.copy_to_device(arr)
            cuda.synchronize()

    def download(self, stream=None) -> np.ndarray:
        if stream:
            result = self.dev_array.copy_to_host(stream=stream.numba)
            stream.sync()
            return result
        else:
            return self.dev_array.copy_to_host()

    def copy_to(self, other, stream=None):
        if self.size != other.size:
            raise ValueError(f"Size mismatch: {self.size} vs {other.size}")

        if hasattr(other, 'dev_array'):
            # Target is GPUMemoryBuffer
            if stream:
                cuda.to_device(self.dev_array, to=other.dev_array, stream=stream.numba)
                stream.sync()
            else:
                cuda.to_device(self.dev_array, to=other.dev_array)
        else:
            # Target is ManagedGPUBuffer, copy via host
            data = self.download(stream=stream)
            other.upload(data, stream=stream)

    def free(self):
        if hasattr(self, "dev_array"):
            del self.dev_array

    def __repr__(self):
        return f"GPUMemoryBuffer(shape={self.shape}, dtype={self.dtype}, size={self.size} bytes)"