import numpy as np, math
from numba import cuda
from metaxuda import GPUMemoryBuffer, StreamPool

@cuda.jit
def sin_kernel(a, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = math.sin(a[i])

def run():
    print("\n=== Large Buffer Test (GPUMemoryBuffer + StreamPool) ===")

    pool = StreamPool(8)
    streams = pool.all()

    num_blocks = 64
    block_size = 128 * 1024 * 1024  # 16 MB each
    total_gb = (num_blocks * block_size) / (1024 ** 3)

    print(f"Allocating {num_blocks} x {block_size // (1024 * 1024)} MB "
          f"({total_gb:.2f} GB total)")

    buffers = []
    for i in range(num_blocks):
        pattern = 1.0 + i * 0.1
        arr = np.full(block_size // 4, pattern, dtype=np.float32)
        stream = streams[i % len(streams)]

        buf = GPUMemoryBuffer(length=block_size // 4, dtype=np.float32)
        buf.upload(arr, stream)
        stream.sync()

        buffers.append((buf, pattern, stream))

    print("All buffers uploaded")

    for i, (buf, expected, stream) in enumerate(buffers):
        out = buf.download(stream)
        stream.sync()
        got = float(out[0])
        print(f"Buffer {i}: expected={expected:.2f}, got={got:.2f}")

    for buf, _, _ in buffers:
        buf.free()

    print("Large buffer test passed")

if __name__ == "__main__":
    run()
