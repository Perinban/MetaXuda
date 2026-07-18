import numpy as np
import math
from numba import cuda
from metaxuda import GPUMemoryBuffer, Pipeline

@cuda.jit
def sin_sqrt(a, out, N):
    i = cuda.grid(1)
    if i < N:
        out[i] = math.sqrt(math.sin(a[i]))

@cuda.jit
def exp_log(a, out, N):
    i = cuda.grid(1)
    if i < N:
        out[i] = math.exp(math.log(a[i]))

def run():
    N = 1024 * 1024  # 1M elements
    x = np.linspace(1, 10, N, dtype=np.float32)

    buf_x = GPUMemoryBuffer(arr=x)
    buf_y = GPUMemoryBuffer(length=N, dtype=np.float32)
    buf_y.zeros()

    print("=" * 60)
    print("Pipeline Execution Test")
    print("=" * 60)
    print(f"Data size: {N:,} elements")
    print()

    # Test exp_log
    pipeline = Pipeline(exp_log)
    pipeline([buf_x], output=buf_y, extra_args=[N])

    result = buf_y.download()
    expected = np.exp(np.log(x))

    print("exp_log kernel:")
    print(f"  Input[:5]: {x[:5]}")
    print(f"  Output[:5]: {result[:5]}")
    print(f"  Expected[:5]: {expected[:5]}")
    print(f"  Max error: {np.max(np.abs(result - expected)):.2e}")
    print(f"  Verification: {np.allclose(result, expected, rtol=1e-4, atol=1e-4)}")
    print()

    # Test sin_sqrt
    buf_y.zeros()
    pipeline = Pipeline(sin_sqrt)
    pipeline([buf_x], output=buf_y, extra_args=[N])

    result = buf_y.download()
    with np.errstate(invalid='ignore'):
        expected = np.sqrt(np.sin(x))

    valid_mask = ~np.isnan(expected)

    print("sin_sqrt kernel:")
    print(f"  Input[:5]: {x[:5]}")
    print(f"  Output[:5]: {result[:5]}")
    print(f"  Expected[:5]: {expected[:5]}")
    print(f"  Max error: {np.max(np.abs(result[valid_mask] - expected[valid_mask])):.2e}")
    print(f"  Verification: {np.allclose(result[valid_mask], expected[valid_mask], rtol=1e-3, atol=2e-4)}")
    print()

    buf_x.free()
    buf_y.free()

    print("=" * 60)

if __name__ == "__main__":
    run()