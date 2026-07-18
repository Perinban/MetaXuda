from .stream import GPUStream

class StreamPool:
    """
    Round-robin pool of GPUStream objects.
    Provides next-stream selection, iteration, bulk access, synchronization, and cleanup.
    """

    def __init__(self, num_streams: int = 8):
        if num_streams <= 0:
            raise ValueError("StreamPool must be initialized with at least one stream")
        self.streams = [GPUStream() for _ in range(num_streams)]
        self._index = 0

    def next(self) -> GPUStream:
        """Return the next stream in round-robin order."""
        stream = self.streams[self._index]
        self._index = (self._index + 1) % len(self.streams)
        return stream

    def all(self):
        """Return all streams for manual management."""
        return list(self.streams)

    def sync_all(self):
        """Synchronize all streams in the pool."""
        for s in self.streams:
            s.sync()

    def close(self):
        """Close all streams."""
        for s in self.streams:
            s.close()
        self.streams = []

    def __iter__(self):
        """Allow direct iteration over the pool."""
        return iter(self.streams)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()