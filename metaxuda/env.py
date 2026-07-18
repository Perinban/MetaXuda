import os
import sys
import ctypes

from .patch import NATIVE_DIR, LIBCUDA, LIBCUDART, LIBNVVM

__ulimit_set = False
__env_patched = False

# buffer.py imports this
CUDART_PATH = LIBCUDART

def _raise_ulimit_once():
    """Raise ulimit -n to 65536 only once per interpreter session."""
    global __ulimit_set
    if __ulimit_set:
        return
    __ulimit_set = True
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < 65536:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception:
        pass

def _preload_library(path):
    """Try to load a shared library globally, warn if it fails."""
    try:
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError as e:
        print(f"[MetaXuda] Warning: could not load {path}: {e}", file=sys.stderr)
        return False

def setup_environment():
    """
    Force-load the shim dylibs and export DYLD_LIBRARY_PATH for child
    processes. Resolution for this process is handled in patch.py; dyld has
    already read DYLD_LIBRARY_PATH before Python starts.
    """
    global __env_patched
    if __env_patched:
        return False
    __env_patched = True

    _raise_ulimit_once()

    for lib in (LIBCUDART, LIBCUDA, LIBNVVM):
        _preload_library(lib)

    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    if NATIVE_DIR not in existing.split(":"):
        os.environ["DYLD_LIBRARY_PATH"] = (
            f"{NATIVE_DIR}:{existing}" if existing else NATIVE_DIR
        )

    return False