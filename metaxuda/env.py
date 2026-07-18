"""Environment setup for MetaXuda CUDA shim."""

import os
import sys
import ctypes
from pathlib import Path

from .patch import NATIVE_DIR as _NATIVE_DIR, LIBCUDA, LIBCUDART, LIBNVVM

__ulimit_set = False
__env_patched = False

NATIVE_DIR = Path(_NATIVE_DIR)

# buffers/tiered.py imports this
CUDART_PATH = Path(LIBCUDART)

def _raise_ulimit_once():
    """Raise ulimit -n to 65536 (optional performance optimization)."""
    global __ulimit_set
    if __ulimit_set:
        return
    __ulimit_set = True
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
    except Exception:
        pass

def _preload_library(path):
    """Load a shared library globally, warn if it fails."""
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError as e:
        print(f"[MetaXuda] Warning: could not load {path}: {e}", file=sys.stderr)
        return False

def setup_environment():
    """
    Setup CUDA shim environment.

    The shim reads configuration from environment variables:
    - MX_ENABLE_DATASTORE_COMPRESSION (default: 1)
    - MX_DATASTORE_COMPRESSION_TYPE (default: lz4)
    - MX_DATASTORE_COMPRESSION_LEVEL (default: 3)
    - MX_DISK_PARALLELISM_LEVEL (default: auto)
    - MX_DISK_SPILL_ENABLED (default: 0)
    - MX_TIER3_STRATEGY (default: prefer_external)
    - MX_TIER3_INTERNAL_PATH (default: block_store)
    - MX_TIER3_EXTERNAL_DEVICES (format: "id:path,id:path")
    - MX_DEBUG (options: memory)

    Library resolution for this process is handled in patch.py; dyld has
    already read DYLD_LIBRARY_PATH before Python starts, so exporting it
    here only affects child processes.
    """
    global __env_patched
    if __env_patched:
        return False
    __env_patched = True

    _raise_ulimit_once()

    for lib in (LIBCUDART, LIBCUDA, LIBNVVM):
        _preload_library(lib)

    dyld_path = str(NATIVE_DIR)
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if dyld_path not in existing.split(":"):
        os.environ["DYLD_LIBRARY_PATH"] = f"{dyld_path}:{existing}" if existing else dyld_path

    # cuda_pipeline.so lives in native/
    if dyld_path not in sys.path:
        sys.path.insert(0, dyld_path)

    return False