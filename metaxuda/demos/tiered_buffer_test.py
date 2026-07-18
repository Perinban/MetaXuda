import numpy as np
from metaxuda import StreamPool
from metaxuda.buffer import TieredBuffer

def run():
    print("\n=== Large Buffer Test (TieredBuffer + StreamPool) ===")

    pool = StreamPool(8)
    streams = pool.all()

    num_blocks = 100
    block_size = 128 * 1024 * 1024  # 16 MB
    total_gb = (num_blocks * block_size) / (1024 ** 3)

    print(f"Allocating {num_blocks} x {block_size // (1024 * 1024)} MB "
          f"({total_gb:.2f} GB total)")

    buffers = []
    for i in range(num_blocks):
        pattern = 1.0 + i * 0.1
        arr = np.full(block_size // 4, pattern, dtype=np.float32)
        stream = streams[i % len(streams)]

        buf = TieredBuffer(block_size)
        buf.upload(arr, stream)
        stream.sync()

        buffers.append((buf, pattern, stream))

    print("All tiered buffers uploaded")

    for i, (buf, expected, stream) in enumerate(buffers):
        out = buf.download((block_size // 4,), dtype=np.float32, stream=stream)
        got = float(out[0])
        print(f"Buffer {i}: expected={expected:.2f}, got={got:.2f}")
        assert abs(got - expected) < 1e-3, "Data mismatch"

    for buf, _, _ in buffers:
        buf.free()

    print("Tiered large buffer test passed")

if __name__ == "__main__":
    run()
