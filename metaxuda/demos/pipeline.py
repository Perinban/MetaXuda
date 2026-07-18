import numpy as np
from numba import cuda
from metaxuda import PipelinePool

@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < a.shape[0]:
        c[i] = a[i] + b[i]

def test_pipeline():
    N = 1024 * 1024
    rng = np.random.default_rng(42)
    a_data = rng.random(N).astype(np.float32) * 2 - 1
    b_data = rng.random(N).astype(np.float32) * 0.5

    print("=" * 60)
    print("PipelinePool.run() Test - Automatic Buffer Management")
    print("=" * 60)
    print(f"Data size: {N:,} elements")
    print()

    pool = PipelinePool()
    result = pool.run(add_kernel, a_data, b_data)

    expected = a_data + b_data
    max_error = np.max(np.abs(result - expected))

    print("add_kernel via pool.run():")
    print(f"  Input A[:5]: {a_data[:5]}")
    print(f"  Input B[:5]: {b_data[:5]}")
    print(f"  Output[:5]: {result[:5]}")
    print(f"  Expected[:5]: {expected[:5]}")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Verification: {np.allclose(result, expected, rtol=1e-5, atol=1e-6)}")

    print()
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()