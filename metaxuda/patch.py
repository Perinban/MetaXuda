import os
from collections import namedtuple

import numba.cuda.cuda_paths

NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")

LIBCUDA = os.path.join(NATIVE_DIR, "libcuda.dylib")
LIBCUDART = os.path.join(NATIVE_DIR, "libcudart.dylib")
LIBNVVM = os.path.join(NATIVE_DIR, "libnvvm.dylib")
LIBDEVICE = os.path.join(NATIVE_DIR, "libdevice.bc")

_SHIM_LIBS = {
    "nvvm": LIBNVVM,
    "cudart": LIBCUDART,
    "cuda": LIBCUDA,
}

_env_path_tuple = namedtuple("_env_path_tuple", ["by", "info"])

_patched = False

def setup_native_libs():
    """
    Ensure native libraries exist:
    - libdevice.bc (empty file)
    - libcuda.dylib (symlink -> libcudart.dylib)
    """
    os.makedirs(NATIVE_DIR, exist_ok=True)

    if not os.path.exists(LIBDEVICE):
        with open(LIBDEVICE, "wb"):
            pass

    if not os.path.isfile(LIBCUDART):
        raise FileNotFoundError(
            f"libcudart shim not found at {LIBCUDART}. "
            "Place your built Rust dylib in this location."
        )

    if not os.path.isfile(LIBNVVM):
        raise FileNotFoundError(
            f"libnvvm shim not found at {LIBNVVM}. "
            "Place your built Rust dylib in this location."
        )

    if not os.path.exists(LIBCUDA):
        os.symlink("libcudart.dylib", LIBCUDA)

def _patch_driver_env():
    """
    Point NUMBA_CUDA_DRIVER at the shim. Numba caches this at config import,
    so reload it in case numba was imported first.
    """
    os.environ["NUMBA_CUDA_DRIVER"] = LIBCUDA
    try:
        import numba.core.config as config
        config.reload_config()
    except Exception:
        pass

def _patch_cuda_paths():
    """
    Monkeypatch Numba to resolve nvvm, libdevice and the cudalib dir to the
    shim folder. get_cuda_paths() memoises its result, so drop the cache.
    """
    paths = numba.cuda.cuda_paths
    paths._get_nvvm_path = lambda: _env_path_tuple("METAXUDA", LIBNVVM)
    paths._get_libdevice_paths = lambda: _env_path_tuple("METAXUDA", LIBDEVICE)
    paths._get_cudalib_dir = lambda: _env_path_tuple("METAXUDA", NATIVE_DIR)
    if hasattr(paths.get_cuda_paths, "_cached_result"):
        del paths.get_cuda_paths._cached_result

def _patch_lib_resolution():
    """
    Return absolute shim paths from Numba's resolver. Its probe only matches
    versioned dylib names (libcudart.<ver>.dylib), so the unversioned shims
    are never found and it falls back to a bare name for dyld to resolve.
    """
    from numba.cuda.cudadrv import libs
    if getattr(libs.get_cudalib, "_metaxuda_patched", False):
        return
    original = libs.get_cudalib

    def get_cudalib(lib, static=False):
        path = _SHIM_LIBS.get(lib)
        if path is not None and os.path.isfile(path):
            return path
        return original(lib, static=static)

    get_cudalib._metaxuda_patched = True
    libs.get_cudalib = get_cudalib

def _reset_driver_singleton():
    """
    Clear a driver that already failed to find libcuda. It caches the error
    on first use, so importing numba.cuda before metaxuda would leave it dead.
    """
    from numba.cuda.cudadrv import driver
    obj = getattr(driver, "driver", None)
    if obj is None or getattr(obj, "initialization_error", None) is None:
        return
    obj.__dict__.clear()
    try:
        obj.__init__()
    except Exception:
        pass

def patch_libdevice():
    """
    Point Numba at the shim libraries. Safe to call more than once.
    """
    global _patched
    if _patched:
        return
    setup_native_libs()
    _patch_driver_env()
    _patch_cuda_paths()
    _patch_lib_resolution()
    _reset_driver_singleton()
    _patched = True

def get_libcudart_path():
    """
    Return the absolute path to the libcudart shim.
    """
    setup_native_libs()
    return LIBCUDART