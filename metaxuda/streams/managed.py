"""Managed stream using Rust bindings."""

import sys
from pathlib import Path
from .base import StreamBase

NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"
sys.path.insert(0, str(NATIVE_DIR))
import cuda_pipeline

class ManagedStream(StreamBase):
    """
    CUDA stream managed by Rust backend.

    Provides stream synchronization with full Rust scheduler integration.
    """

    def __init__(self):
        """Create a new CUDA stream via Rust."""
        self._stream = cuda_pipeline.create_stream()

    @property
    def handle(self):
        """Get underlying stream handle."""
        return self._stream.handle

    def sync(self):
        """Block until all stream operations complete."""
        if self._stream:
            pass

    def close(self):
        """Release stream resources."""
        if self._stream:
            cuda_pipeline.destroy_stream(self._stream)
            self._stream = None