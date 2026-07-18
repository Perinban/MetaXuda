from numba import cuda

class GPUStream:
    """
    GPU stream wrapper using Numba streams.
    Provides synchronization, cleanup, and context manager support.
    """

    def __init__(self):
        self._numba_stream = cuda.stream()

    @property
    def numba(self):
        """Return the underlying Numba stream object."""
        return self._numba_stream

    def sync(self):
        """Synchronize the stream."""
        self._numba_stream.synchronize()

    def close(self):
        """Close the stream. This is a no-op; Numba cleans up automatically."""
        self._numba_stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.sync()
        self.close()

    def __del__(self):
        self.close()

DEFAULT_STREAM = GPUStream()