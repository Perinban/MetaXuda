import numpy as np
from numba import cuda
from metaxuda import GPUMemoryBuffer, run_pipeline

# ------------------------------------------------
# Kernel
# ------------------------------------------------
@cuda.jit
def mul3d_kernel(a, b, out):
    d, h, w = cuda.grid(3)
    D, H, W = out.shape
    if d < D and h < H and w < W:
        out[d, h, w] = a[d, h, w] * b[d, h, w]

# ------------------------------------------------
# Demo
# ------------------------------------------------
def run():
    D, H, W = 4, 4, 4
    a = np.arange(D * H * W, dtype=np.float32).reshape(D, H, W)
    b = np.ones((D, H, W), dtype=np.float32) * 2

    # Only pass inputs, pipeline allocates/manages the output
    buf_a = GPUMemoryBuffer(a)
    buf_b = GPUMemoryBuffer(b)

    out = run_pipeline([mul3d_kernel], [buf_a, buf_b]).download()

    print("Input A:\n", a)
    print("Input B:\n", b)
    print("Output:\n", out)

    buf_a.free()
    buf_b.free()

# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    run()
