import numpy as np
from metaxuda import TieredBuffer

def run():
    print("\n=== TieredBuffer Upload Test ===")

    num_buffers = 100
    block_size = 128 * 1024 * 1024  # 128 MB
    total_gb = (num_buffers * block_size) / (1024 ** 3)

    print(f"Allocating {num_buffers} x {block_size // (1024 * 1024)} MB ({total_gb:.2f} GB total)")

    buffers = []
    for i in range(num_buffers):
        pattern = 1.0 + i * 0.1
        arr = np.full(block_size // 4, pattern, dtype=np.float32)

        buf = TieredBuffer(block_size)
        buf.upload(arr)
        buffers.append(buf)

        if (i + 1) % 10 == 0:
            print(f"  Uploaded {i + 1}/{num_buffers} buffers")

    print(f"\n✓ Successfully uploaded all {num_buffers} buffers ({total_gb:.2f} GB)")

    for buf in buffers:
        buf.free()

    print("✓ Test passed")

if __name__ == "__main__":
    run()