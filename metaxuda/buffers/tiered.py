"""Tiered buffer with GPU/RAM/Disk support."""

import ctypes
import numpy as np
from numba import cuda
from ..env import CUDART_PATH
from .base import BufferBase

_cuda = ctypes.CDLL(str(CUDART_PATH))

MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2

_cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cuda.cudaMalloc.restype = ctypes.c_int

_cuda.cudaFree.argtypes = [ctypes.c_void_p]
_cuda.cudaFree.restype = ctypes.c_int

_cuda.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p
]
_cuda.cudaMemcpy.restype = ctypes.c_int


class TieredBuffer(BufferBase):
    """GPU buffer with automatic tiering to RAM and disk."""

    def __init__(self, size_bytes: int):
        self.ptr = ctypes.c_void_p()
        self.size = int(size_bytes)
        rc = _cuda.cudaMalloc(ctypes.byref(self.ptr), self.size)
        if rc != 0:
            raise RuntimeError(f"cudaMalloc failed with code {rc}")

    @property
    def device_ptr(self):
        return self.ptr.value

    def upload(self, arr: np.ndarray, stream=None):
        if arr.nbytes > self.size:
            raise ValueError(f"Array too large: {arr.nbytes} > {self.size}")

        rc = _cuda.cudaMemcpy(
            self.ptr,
            arr.ctypes.data_as(ctypes.c_void_p),
            arr.nbytes,
            MEMCPY_HOST_TO_DEVICE,
            ctypes.c_void_p(0),
        )
        if rc != 0:
            raise RuntimeError(f"Upload failed with code {rc}")

        if stream:
            stream.sync()
        else:
            cuda.synchronize()

    def download(self, shape=None, dtype=np.float32, stream=None) -> np.ndarray:
        if shape is None:
            shape = getattr(self, 'shape', None)
            if shape is None:
                raise ValueError("Shape must be provided")

        host = np.empty(shape, dtype=dtype)

        rc = _cuda.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            self.ptr,
            host.nbytes,
            MEMCPY_DEVICE_TO_HOST,
            ctypes.c_void_p(0),
        )
        if rc != 0:
            raise RuntimeError(f"Download failed with code {rc}")

        if stream:
            stream.sync()
        else:
            cuda.synchronize()

        return host

    def copy_to(self, other, stream=None):
        raise NotImplementedError("TieredBuffer copy_to not yet implemented")

    def free(self):
        if getattr(self, "ptr", None):
            _cuda.cudaFree(self.ptr)
            self.ptr = None

    def __repr__(self):
        return f"TieredBuffer(size={self.size} bytes, ptr=0x{self.ptr.value:x})"