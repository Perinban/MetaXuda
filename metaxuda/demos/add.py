import numpy as np
from numba import cuda
from metaxuda import GPUMemoryBuffer, Pipeline

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

def demo_1d_add():
    N = 1024
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(arr=A)
    bufB = GPUMemoryBuffer(arr=B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)
    bufC.zeros()

    pipeline = Pipeline(add_kernel_1d)
    pipeline([bufA, bufB], output=bufC, extra_args=[N])

    C_result = bufC.download()
    expected = A + B

    print("=" * 60)
    print("1D Addition Test")
    print("=" * 60)
    print(f"Input A[:5]: {A[:5]}")
    print(f"Input B[:5]: {B[:5]}")
    print(f"Output C[:5]: {C_result[:5]}")
    print(f"Expected[:5]: {expected[:5]}")
    print(f"Verification: {np.allclose(C_result, expected)}")
    print()

    bufA.free()
    bufB.free()
    bufC.free()

def demo_2d_add():
    W, H = 32, 16
    N = W * H
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(arr=A)
    bufB = GPUMemoryBuffer(arr=B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)
    bufC.zeros()

    pipeline = Pipeline(add_kernel_2d)
    pipeline([bufA, bufB], output=bufC, extra_args=[W, H])

    C_result = bufC.download()
    expected = A + B

    print("=" * 60)
    print(f"2D Addition Test ({W}x{H})")
    print("=" * 60)
    print(f"Input A[:5]: {A[:5]}")
    print(f"Input B[:5]: {B[:5]}")
    print(f"Output C[:5]: {C_result[:5]}")
    print(f"Expected[:5]: {expected[:5]}")
    print(f"Verification: {np.allclose(C_result, expected)}")
    print()

    bufA.free()
    bufB.free()
    bufC.free()

def demo_3d_add():
    W, H, D = 8, 4, 8
    N = W * H * D
    A = np.random.rand(N).astype(np.float32)
    B = np.random.rand(N).astype(np.float32)

    bufA = GPUMemoryBuffer(arr=A)
    bufB = GPUMemoryBuffer(arr=B)
    bufC = GPUMemoryBuffer(length=N, dtype=np.float32)
    bufC.zeros()

    pipeline = Pipeline(add_kernel_3d)
    pipeline([bufA, bufB], output=bufC, extra_args=[W, H, D])

    C_result = bufC.download()
    expected = A + B

    print("=" * 60)
    print(f"3D Addition Test ({W}x{H}x{D})")
    print("=" * 60)
    print(f"Input A[:5]: {A[:5]}")
    print(f"Input B[:5]: {B[:5]}")
    print(f"Output C[:5]: {C_result[:5]}")
    print(f"Expected[:5]: {expected[:5]}")
    print(f"Verification: {np.allclose(C_result, expected)}")
    print()

    bufA.free()
    bufB.free()
    bufC.free()

if __name__ == "__main__":
    demo_1d_add()
    demo_2d_add()
    demo_3d_add()