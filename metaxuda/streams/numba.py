"""Numba CUDA stream wrapper."""

from numba import cuda
from .base import StreamBase

class GPUStream(StreamBase):
    """
    CUDA stream wrapper for async GPU operations.

    Provides stream synchronization and lifecycle management.
    Backend scheduling is handled by Rust layer.
    """

    def __init__(self):
        """Create a new CUDA stream."""
        self._numba_stream = cuda.stream()

    @property
    def numba(self):
        """
        Get underlying Numba stream object.

        Returns:
            Numba CUDA stream instance
        """
        return self._numba_stream

    def sync(self):
        """Block until all stream operations complete."""
        if self._numba_stream:
            self._numba_stream.synchronize()

    def close(self):
        """Release stream resources."""
        if self._numba_stream:
            self._numba_stream = None

DEFAULT_STREAM = None