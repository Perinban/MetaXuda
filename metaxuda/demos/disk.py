import os
import shutil
import tempfile
import numpy as np
from metaxuda import TieredBuffer

TIER3_DIR = "block_store"
TIER3_PATH = os.path.join(tempfile.gettempdir(), TIER3_DIR)

def run():
    print("\n=== TieredBuffer Disk Spill Test ===")

    # Clean up old tier-3 storage
    if os.path.exists(TIER3_PATH):
        shutil.rmtree(TIER3_PATH, ignore_errors=True)

    # Enable disk spill and set path
    os.environ["MX_DISK_SPILL_ENABLED"] = "1"
    os.environ["MX_TIER3_INTERNAL_PATH"] = TIER3_PATH
    print(f"✓ Disk spill enabled")
    print(f"✓ Tier-3 path: {TIER3_PATH}\n")

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
            if os.path.exists(TIER3_PATH):
                total_size = sum(
                    os.path.getsize(os.path.join(TIER3_PATH, f))
                    for f in os.listdir(TIER3_PATH)
                    if f.endswith(".t3blk")
                )
                print(f"  Tier-3 storage: {total_size / (1024 ** 3):.2f} GB")

    print(f"\n✓ Successfully uploaded all {num_buffers} buffers ({total_gb:.2f} GB)")

    for buf in buffers:
        buf.free()

    print("✓ Test passed")

if __name__ == "__main__":
    run()