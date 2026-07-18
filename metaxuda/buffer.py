import ctypes
import numpy as np
from numba import cuda
from .env import CUDART_PATH

# ============================================================
# ------------------- Numba-backed buffer --------------------
# ============================================================

class GPUMemoryBuffer:
    """
    GPU buffer wrapper backed by Numba DeviceNDArray.
    """

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
        """
        Create a GPUMemoryBuffer from an existing Numba DeviceNDArray.
        Used internally by run_pipeline().
        """
        buf = cls.__new__(cls)
        buf.dev_array = dev_array
        buf.dtype = dev_array.dtype
        buf.shape = dev_array.shape
        buf.length = dev_array.size
        buf.size = dev_array.nbytes
        return buf

    def upload(self, arr: np.ndarray, stream=None):
        if arr.shape != self.shape:
            raise ValueError("Shape mismatch")
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

    def free(self):
        if hasattr(self, "dev_array"):
            del self.dev_array

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

# ============================================================
# ------------------- Shim-backed buffer ---------------------
# ============================================================

_cuda = ctypes.CDLL(str(CUDART_PATH))

MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2

_cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cuda.cudaMalloc.restype = ctypes.c_int

_cuda.cudaFree.argtypes = [ctypes.c_void_p]
_cuda.cudaFree.restype = ctypes.c_int

_cuda.cudaMemcpy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
]
_cuda.cudaMemcpy.restype = ctypes.c_int

class TieredBuffer:
    """
    Buffer backed by Rust shim with GPU → RAM → Disk tiering.
    """

    def __init__(self, size_bytes: int):
        self.ptr = ctypes.c_void_p()
        self.size = int(size_bytes)
        rc = _cuda.cudaMalloc(ctypes.byref(self.ptr), self.size)
        if rc != 0:
            raise RuntimeError(f"cudaMalloc failed (code {rc})")

    def upload(self, arr: np.ndarray, stream=None):
        if arr.nbytes > self.size:
            raise ValueError("Array too large for buffer")
        stream_handle = stream.numba.handle if stream else None
        rc = _cuda.cudaMemcpy(
            self.ptr,
            arr.ctypes.data_as(ctypes.c_void_p),
            arr.nbytes,
            MEMCPY_HOST_TO_DEVICE,
            stream_handle,
        )
        if rc != 0:
            raise RuntimeError(f"Upload failed (code {rc})")
        if stream:
            stream.sync()
        else:
            cuda.synchronize()

    def download(self, shape, dtype=np.float32, stream=None):
        host = np.empty(shape, dtype=dtype)
        stream_handle = stream.numba.handle if stream else None
        rc = _cuda.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            self.ptr,
            host.nbytes,
            MEMCPY_DEVICE_TO_HOST,
            stream_handle,
        )
        if rc != 0:
            raise RuntimeError(f"Download failed (code {rc})")
        if stream:
            stream.sync()
        else:
            cuda.synchronize()
        return host

    def free(self):
        if getattr(self, "ptr", None):
            _cuda.cudaFree(self.ptr)
            self.ptr = None

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()