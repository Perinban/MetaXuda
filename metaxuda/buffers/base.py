"""Base class for all buffer types."""

import numpy as np

class BufferBase:
    """Base class for all buffer types."""

    @property
    def device_ptr(self):
        """Get device pointer for kernel launches."""
        raise NotImplementedError

    def upload(self, arr: np.ndarray, stream=None):
        """Copy data from host to device."""
        raise NotImplementedError

    def download(self, stream=None) -> np.ndarray:
        """Copy data from device to host."""
        raise NotImplementedError

    def copy_to(self, other, stream=None):
        """Copy this buffer to another buffer (device-to-device)."""
        raise NotImplementedError

    def fill(self, value, stream=None):
        """Fill buffer with a scalar value."""
        arr = np.full(self.shape, value, dtype=self.dtype)
        self.upload(arr, stream=stream)

    def zeros(self, stream=None):
        """Fill buffer with zeros."""
        self.fill(0, stream=stream)

    def ones(self, stream=None):
        """Fill buffer with ones."""
        self.fill(1, stream=stream)

    def free(self):
        """Release GPU memory."""
        raise NotImplementedError

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()