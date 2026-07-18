# MetaXuda

MetaXuda is an **experimental CUDA-compatible runtime shim for Apple Silicon**, written in **Rust**, that runs **Numba CUDA kernels** on Apple GPUs by mapping CUDA runtime calls to **Apple Metal**.

Kernel bodies (`@cuda.jit`) run unchanged. Memory transfers must go through the
buffer API rather than passing NumPy arrays directly - see
[Known Limitations](#-known-limitations).

It is designed as a drop-in replacement for core CUDA runtime libraries, enabling GPU-accelerated Python workflows on macOS **without requiring the NVIDIA CUDA Toolkit or NVIDIA hardware**.

---

## ✨ Features

* Drop-in replacement for `libcudart.dylib` and `libcuda.dylib`
* Run **Numba CUDA kernels** (`@cuda.jit`) directly on Apple Metal
* Metal-backed implementations of core CUDA APIs:

  * `cudaMalloc` / `cudaFree`
  * `cudaMemcpy` / `cudaMemcpyAsync`
  * `cudaLaunchKernel`
* Asynchronous execution with **stream-style overlap** (copy / compute / copy)
* Tier-aware memory management (GPU-first execution)
* Ships with:

  * Stubbed `libdevice.bc` for Numba compatibility
  * Precompiled Metal `.metallib` shaders for fused math operations
  * **`cuda_pipeline.so`**, exposing a low-level execution API that allows Numba and other callers to bypass the CUDA runtime shim and dispatch operations directly
* No CUDA Toolkit, NVIDIA drivers, or NVIDIA GPU required

---

## ⚠️ Project Status

**Alpha / Research Prototype**

MetaXuda is under active development and currently targets:

* Numba CUDA kernels
* Single-GPU execution on Apple Silicon

Not all CUDA APIs are implemented, and behavior may differ from NVIDIA CUDA in edge cases.

---

## 🚫 Known Limitations

### macOS 26 (Tahoe) or later is required

The bundled Metal shaders are compiled against **Metal Shading Language 4.0**, which
is only supported on macOS 26 and above. On earlier releases (Sequoia and older),
kernel launches fail when the compute pipeline is created:

```
thread 'cuda-async-worker' panicked at libcudart/src/api/context.rs:72:13:
Failed to create compute pipeline 'elementwise_dispatch':
"Function elementwise_dispatch is using language version 4.0 which is
 incompatible with this OS."
```

There is no workaround from the Python side: the language version is baked into
the precompiled shaders inside `libcudart.dylib`. Running on macOS 15 or earlier
requires a build targeting Metal 3, which is not currently published.

### Host arrays are not transferred automatically

Numba normally copies a NumPy array to the device for you when it is passed
straight to a kernel. That implicit transfer does not work through the shim:
the kernel runs, but results are **silently** left as zeros and no error is
raised.

```python
out = np.zeros(n, dtype=np.float32)
add[32, 32](a, b, out) # host arrays
print(out[:5]) # [0. 0. 0. 0. 0.] -- wrong, and silent
```

Allocate device memory explicitly instead, with either `GPUMemoryBuffer` or
`cuda.to_device`:

```python
buf_out = GPUMemoryBuffer(length=n, dtype=np.float32)
add[32, 32](buf_a.dev_array, buf_b.dev_array, buf_out.dev_array)
cuda.synchronize()
print(buf_out.download()[:5]) # [0. 2. 4. 6. 8.]
```

Existing Numba CUDA code therefore needs its memory transfers adapted; kernel
bodies run unchanged.

### Scalar arithmetic in kernel arguments

Scalars passed to a kernel are usable in comparisons (for example bounds checks
against `n`), but a scalar used directly in arithmetic is not applied:

```python
@cuda.jit
def scale(a, out, f, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = a[i] * f # f is ignored; out receives a unchanged
```

Pass scalars as single-element device arrays until this is resolved.

---

## ⚙️ Installation

### Requirements

* macOS 26 (Tahoe) or later - see [Known Limitations](#-known-limitations)
* Apple Silicon (M-series)
* Python >= 3.10
* NumPy >= 1.23
* Numba >= 0.61, < 0.67
* psutil >= 5.9

### Install (Editable / Dev)

```bash
# Clone the repository
git clone https://github.com/perinban/MetaXuda.git
cd MetaXuda

# Install in editable mode
pip install -e .
```

The installation places the required shim libraries (`libcudart.dylib`, `libcuda.dylib`, and `libdevice.bc`) inside the package so they can be discovered by Numba at runtime.

---

## 📂 Package Layout

MetaXuda ships demos and helper modules **inside the Python package** so they are available in editable and installed modes:

```
metaxuda/
├── buffers/        # GPU, managed, and tiered buffer abstractions
├── execution/      # Direct and pooled execution backends
├── streams/        # Stream and async execution helpers (Numba-compatible)
├── demos/          # End-to-end demos and debug examples
├── native/         # Native shims and pipelines
│   ├── libcudart.dylib
│   ├── libcuda.dylib
│   ├── libnvvm.dylib
│   ├── libdevice.bc
│   └── cuda_pipeline.so
├── env.py          # Environment detection and setup
├── patch.py        # Numba / runtime patching hooks
└── __init__.py
```

The **`demos/`** directory contains runnable examples covering kernel execution, buffers, streams, disk tiering, and the direct math pipeline.

You can run them directly once the package is installed:

```bash
python -m metaxuda.demos.add
python -m metaxuda.demos.pipeline
```

---

## 🚀 Usage

Import `metaxuda` before using `numba.cuda`, and pass device buffers to your
kernels. Kernel bodies themselves need no changes:

```python
import metaxuda
from metaxuda import GPUMemoryBuffer
from numba import cuda
import numpy as np

@cuda.jit
def add(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]

n = 1024
buf_a = GPUMemoryBuffer(arr=np.arange(n, dtype=np.float32))
buf_b = GPUMemoryBuffer(arr=np.arange(n, dtype=np.float32))
buf_out = GPUMemoryBuffer(length=n, dtype=np.float32)

add[32, 32](buf_a.dev_array, buf_b.dev_array, buf_out.dev_array)
cuda.synchronize()

print(buf_out.download()[:5])   # [0. 2. 4. 6. 8.]
```

`import metaxuda` must come before any kernel launch. It redirects Numba at the
bundled shim libraries; without it, Numba looks for a real CUDA toolkit and
fails with `CudaSupportError: Error at driver init`.

No environment variables are required. `DYLD_LIBRARY_PATH`, `NUMBA_CUDA_DRIVER`
and `CUDA_HOME` do not need to be set.

Execution is transparently dispatched to Metal via the MetaXuda runtime.

---

## 🗜️ Quantization, Compression, and Disk Tiering

MetaXuda supports **quantized and compressed data storage** for non-resident buffers and intermediate results. These behaviors are controlled via environment variables and handled by the runtime initialization logic in `env.py`.

This is primarily used for **Tier‑3 (disk-backed) storage**, allowing large workloads to exceed GPU memory limits while minimizing I/O and storage overhead.

### Environment Configuration

The shim reads the following environment variables at startup:

* `MX_ENABLE_DATASTORE_COMPRESSION` *(default: `1`)*
  Enable or disable compression for spilled data blocks.

* `MX_DATASTORE_COMPRESSION_TYPE` *(default: `lz4`)*
  Compression algorithm to use (e.g. `lz4`).

* `MX_DATASTORE_COMPRESSION_LEVEL` *(default: `3`)*
  Compression level passed to the backend compressor.

* `MX_DISK_PARALLELISM_LEVEL` *(default: `auto`)*
  Controls parallel read/write behavior for disk operations.

* `MX_DISK_SPILL_ENABLED` *(default: `0`)*
  Enable spilling GPU buffers to disk when memory pressure occurs.

* `MX_TIER3_STRATEGY` *(default: `prefer_external`)*
  Strategy for selecting Tier‑3 storage locations.

* `MX_TIER3_INTERNAL_PATH` *(default: `block_store`)*
  Directory used for internal Tier‑3 storage.

* `MX_TIER3_EXTERNAL_DEVICES` *(format: `id:path,id:path`)*
  Comma‑separated list of external devices or paths for Tier‑3 storage.

* `MX_DEBUG` *(options: `memory`)*
  Enable debug logging for specific subsystems.

These settings allow fine‑grained control over **compression, quantization, disk spill behavior, and debugging** without changing application code.

---

## 🧮 Operation Coverage

MetaXuda includes a **precompiled Metal math pipeline** (`cuda_pipeline.so`) implementing a broad set of scalar and elementwise operations that can be invoked directly by Numba or higher-level tooling.

* **230+ operations** covering:

  * Arithmetic, comparison, and logical ops
  * Trigonometric and hyperbolic functions
  * Exponentials, logarithms, and powers
  * Reductions and distance metrics
  * Activation functions (ReLU, GELU, SiLU, Mish, etc.)
  * Probability distributions and loss functions
  * Signal, interpolation, and utility math
* Each operation is mapped to a corresponding **Metal expression**
* Selected ops support **fast-math variants** where numerically safe
* **Full operation list**: See `config/operations.json` for all supported operations and their signatures

This allows many Numba-generated kernels to execute without requiring full PTX → Metal translation, significantly reducing overhead.

---

## 🧠 Architecture Overview

* **Rust-based CUDA shim** implementing core CUDA runtime APIs
* **Metal compute pipelines** for kernel execution
* **Stubbed NVVM / libdevice layer** for Numba compilation compatibility
* Python package acts as a loader and distribution mechanism for native libraries

---

## License

MetaXuda is free for students and personal use. Commercial use requires a license.

- 🎓 **Students**: Free with valid educational email
- 👤 **Personal**: Free for non-commercial projects
- 🏢 **Commercial**: Contact p.perinban@gmail.com

See [LICENSE](LICENSE) for full terms.

---

## 🙏 Disclaimer

MetaXuda is **not affiliated with NVIDIA**. CUDA is a trademark of NVIDIA Corporation. This project is an independent compatibility layer intended for research and development purposes.