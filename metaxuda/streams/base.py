"""Base stream class."""

class StreamBase:
    """Base class for all stream types."""

    def sync(self):
        """Block until all stream operations complete."""
        raise NotImplementedError

    def close(self):
        """Release stream resources."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sync()
        self.close()

    def __del__(self):
        self.close()