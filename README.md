# MetaXuda

MetaXuda is a **CUDA runtime shim for Apple Silicon**, written in Rust, that allows **Numba CUDA kernels** to run unmodified by mapping CUDA calls to Metal.

---

## ✨ Features
- Drop-in replacement for `libcudart.dylib` / `libcuda.dylib`
- Run Numba CUDA kernels (`@cuda.jit`) directly on Apple Metal
- Includes precompiled Metal `.metallib` shaders for fused math ops
- Ships with a stubbed `libdevice.bc` so no CUDA Toolkit is required

---

## ⚙️ Installation

### Requirements
- macOS 13+ with Apple Silicon (M1/M2/M3)
- Python >=3.10
- [NumPy](https://numpy.org/) (>=1.23)
- [Numba](https://numba.pydata.org/) (>=0.59)

### Steps
```bash
# Clone the repo
git clone https://github.com/perinban/MetaXuda.git
cd MetaXuda-

# Back to project root
cd ..
pip install -e .