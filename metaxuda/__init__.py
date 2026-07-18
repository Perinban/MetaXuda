from .patch import patch_libdevice, get_libcudart_path

# Must run before numba.cuda initialises its driver
patch_libdevice()

from .env import setup_environment

setup_environment()

from .buffers import GPUMemoryBuffer, TieredBuffer, ManagedGPUBuffer
from .streams import GPUStream, ManagedStream
from .execution import run_pipeline, Pipeline, PipelinePool

__version__ = "2.0.2"

__all__ = [
    "GPUMemoryBuffer",
    "TieredBuffer",
    "ManagedGPUBuffer",
    "GPUStream",
    "ManagedStream",
    "run_pipeline",
    "Pipeline",
    "PipelinePool",
    "patch_libdevice",
    "get_libcudart_path",
]