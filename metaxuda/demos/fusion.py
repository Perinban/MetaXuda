import numpy as np
import math
from numba import cuda
import metaxuda

# ------------------------------------------------
# Kernels
# ------------------------------------------------

@cuda.jit
def sin_sqrt(a, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = math.sqrt(math.sin(a[i]))

@cuda.jit
def exp_log(a, out):
    i = cuda.grid(1)
    if i < a.size:
        out[i] = math.exp(math.log(a[i]))

# ------------------------------------------------
# Demo
# ------------------------------------------------

def run():
    x = np.linspace(1, 10, 8, dtype=np.float32)

    buf_x = metaxuda.GPUMemoryBuffer(arr=x)
    buf_y = metaxuda.GPUMemoryBuffer(length=x.size, dtype=np.float32)

    threads_per_block = 128
    blocks_per_grid = (x.size + threads_per_block - 1) // threads_per_block

    exp_log[blocks_per_grid, threads_per_block](buf_x.dev_array, buf_y.dev_array)
    cuda.synchronize()

    print("Input: ", x)
    print("Output:", buf_y.download())

    buf_x.free()
    buf_y.free()

# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":
    run()