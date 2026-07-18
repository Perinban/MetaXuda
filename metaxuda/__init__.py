from .patch import patch_libdevice, get_libcudart_path

# Must run before numba.cuda initialises its driver
patch_libdevice()

from .env import setup_environment

setup_environment()

from .buffer import GPUMemoryBuffer, TieredBuffer
from .stream import GPUStream, DEFAULT_STREAM
from .pipeline import run_pipeline
from .pool import StreamPool

__version__ = "1.0.0"

__all__ = [
    "GPUMemoryBuffer",
    "TieredBuffer",
    "GPUStream",
    "DEFAULT_STREAM",
    "StreamPool",
    "run_pipeline",
    "patch_libdevice",
    "get_libcudart_path",
]