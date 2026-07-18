import numpy as np
from numba import cuda
from metaxuda import GPUMemoryBuffer, StreamPool

# ------------------------------------------------
# Kernels
# ------------------------------------------------

@cuda.jit
def add_kernel_1d(A, B, C, N):
    idx = cuda.grid(1)
    if idx < N:
        C[idx] = A[idx] + B[idx]

@cuda.jit
def add_kernel_2d(A, B, C, W, H):
    x, y = cuda.grid(2)
    if x < W and y < H:
        idx = y * W + x
        C[idx] = A[idx] + B[idx]

@cuda.jit
def add_kernel_3d(A, B, C, W, H, D):
    x, y, z = cuda.grid(3)
    if x < W and y < H and z < D:
        idx = z * H * W + y * W + x
        C[idx] = A[idx] + B[idx]

# ------------------------------------------------
# Demos
# ------------------------------------------------

def demo_1d_add():
    N = 1024
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(length=A.size, dtype=A.dtype)
    bufA.upload(A)
    bufB = GPUMemoryBuffer(length=B.size, dtype=B.dtype)
    bufB.upload(B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    add_kernel_1d[blocks_per_grid, threads_per_block](bufA.dev_array, bufB.dev_array, bufC.dev_array, N)
    cuda.synchronize()

    print("1D Input A[:5]:", A[:5])
    print("1D Input B[:5]:", B[:5])
    print("1D Output[:5]:", bufC.download()[:5])

    bufA.free(); bufB.free(); bufC.free()

def demo_2d_add():
    W, H = 32, 16
    N = W * H
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(length=A.size, dtype=A.dtype)
    bufA.upload(A)
    bufB = GPUMemoryBuffer(length=B.size, dtype=B.dtype)
    bufB.upload(B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid = ((W + threads_per_block[0] - 1) // threads_per_block[0],
                       (H + threads_per_block[1] - 1) // threads_per_block[1])
    add_kernel_2d[blocks_per_grid, threads_per_block](bufA.dev_array, bufB.dev_array, bufC.dev_array, W, H)
    cuda.synchronize()

    print("2D Input A[:5]:", A[:5])
    print("2D Input B[:5]:", B[:5])
    print("2D Output[:5]:", bufC.download()[:5])

    bufA.free(); bufB.free(); bufC.free()

def demo_3d_add():
    W, H, D = 8, 4, 8  # keep threads <=256
    N = W * H * D
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(length=A.size, dtype=A.dtype)
    bufA.upload(A)
    bufB = GPUMemoryBuffer(length=B.size, dtype=B.dtype)
    bufB.upload(B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)

    threads_per_block = (8, 4, 8)
    blocks_per_grid = ((W + threads_per_block[0] - 1) // threads_per_block[0],
                       (H + threads_per_block[1] - 1) // threads_per_block[1],
                       (D + threads_per_block[2] - 1) // threads_per_block[2])
    add_kernel_3d[blocks_per_grid, threads_per_block](bufA.dev_array, bufB.dev_array, bufC.dev_array, W, H, D)
    cuda.synchronize()

    print("3D Input A[:5]:", A[:5])
    print("3D Input B[:5]:", B[:5])
    print("3D Output[:5]:", bufC.download()[:5])

    bufA.free(); bufB.free(); bufC.free()

def demo_with_streams():
    W, H = 64, 32
    N = W * H
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(length=A.size, dtype=A.dtype)
    bufA.upload(A)
    bufB = GPUMemoryBuffer(length=B.size, dtype=B.dtype)
    bufB.upload(B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)

    pool = StreamPool(num_streams=2)
    s1 = pool.next()

    threads_per_block = (16, 16)
    blocks_per_grid = ((W + threads_per_block[0] - 1) // threads_per_block[0],
                       (H + threads_per_block[1] - 1) // threads_per_block[1])
    add_kernel_2d[blocks_per_grid, threads_per_block, s1.numba](bufA.dev_array, bufB.dev_array, bufC.dev_array, W, H)
    pool.sync_all()

    print("Streamed Input A[:5]:", A[:5])
    print("Streamed Input B[:5]:", B[:5])
    print("Streamed Output[:5]:", bufC.download()[:5])

    bufA.free(); bufB.free(); bufC.free()

# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":
    demo_1d_add()
    demo_2d_add()
    demo_3d_add()
    demo_with_streams()