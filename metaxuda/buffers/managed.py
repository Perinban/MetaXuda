"""Managed GPU buffer with Rust backend."""

import sys
from pathlib import Path
import numpy as np
from .base import BufferBase

NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"
sys.path.insert(0, str(NATIVE_DIR))
import cuda_pipeline

class ManagedGPUBuffer(BufferBase):
    """GPU buffer using Rust bindings."""

    def __init__(self, arr: np.ndarray = None, length: int = None,
                 dtype=np.float32, shape=None):
        if arr is not None:
            self.dtype = arr.dtype
            self.shape = arr.shape
            self.length = arr.size
            self.size = arr.nbytes
            self.dev_ptr = cuda_pipeline.malloc(self.size)
            cuda_pipeline.copy_to_device(self.dev_ptr, arr.tobytes())
        elif length is not None or shape is not None:
            self.dtype = np.dtype(dtype)
            if shape is not None:
                self.shape = shape
                self.length = np.prod(shape)
            else:
                self.shape = (length,)
                self.length = length
            self.size = self.length * self.dtype.itemsize
            self.dev_ptr = cuda_pipeline.malloc(self.size)
        else:
            raise ValueError("Must provide either arr or length/shape+dtype")

        cuda_pipeline.register_external_buffer(self.dev_ptr.ptr(), self.size)

    @property
    def device_ptr(self):
        return self.dev_ptr.ptr()

    def upload(self, arr: np.ndarray, stream=None):
        if arr.shape != self.shape:
            raise ValueError(f"Shape mismatch: {arr.shape} vs {self.shape}")
        cuda_pipeline.copy_to_device(self.dev_ptr, arr.tobytes())

    def download(self, stream=None) -> np.ndarray:
        data = cuda_pipeline.copy_from_device(self.dev_ptr, self.size)

        if isinstance(data, list):
            data = bytes(data)
        elif not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"Expected bytes or list from copy_from_device, got {type(data)}")

        return np.frombuffer(data, dtype=self.dtype).reshape(self.shape)

    def zeros(self):
        """Initialize buffer to zeros."""
        zeros_data = np.zeros(self.shape, dtype=self.dtype)
        cuda_pipeline.copy_to_device(self.dev_ptr, zeros_data.tobytes())

    def copy_to(self, other, stream=None):
        if self.size != other.size:
            raise ValueError(f"Size mismatch: {self.size} vs {other.size}")
        data = self.download(stream=stream)
        other.upload(data, stream=stream)

    def free(self):
        cuda_pipeline.unregister_external_buffer(self.dev_ptr.ptr())
        cuda_pipeline.free(self.dev_ptr)

    def __del__(self):
        if hasattr(self, 'dev_ptr'):
            try:
                self.free()
            except:
                pass

    def __repr__(self):
        return f"ManagedGPUBuffer(shape={self.shape}, dtype={self.dtype}, size={self.size} bytes)"