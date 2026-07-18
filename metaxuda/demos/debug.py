import os
import shutil
import tempfile
import numpy as np
from metaxuda import TieredBuffer

TIER3_DIR = "block_store"
TIER3_PATH = os.path.join(tempfile.gettempdir(), TIER3_DIR)

def run():
    print("\n=== TieredBuffer Debug Test ===")

    # Clean up old tier-3 storage
    if os.path.exists(TIER3_PATH):
        shutil.rmtree(TIER3_PATH, ignore_errors=True)

    # Disable disk spill, enable debug mode for validation
    os.environ["MX_DISK_SPILL_ENABLED"] = "0"
    os.environ["MX_DEBUG"] = "memory"
    print(f"✓ Disk spill disabled")
    print(f"✓ Debug mode: memory (no quantization)\n")

    num_buffers = 100
    block_size = 128 * 1024 * 1024  # 128 MB
    total_gb = (num_buffers * block_size) / (1024 ** 3)

    print(f"Allocating {num_buffers} x {block_size // (1024 * 1024)} MB ({total_gb:.2f} GB total)")

    buffers = []
    patterns = []

    # Upload phase
    for i in range(num_buffers):
        pattern = 1.0 + i * 0.1
        arr = np.full(block_size // 4, pattern, dtype=np.float32)

        buf = TieredBuffer(block_size)
        buf.upload(arr)
        buffers.append(buf)
        patterns.append(pattern)

        if (i + 1) % 10 == 0:
            print(f"  Uploaded {i + 1}/{num_buffers} buffers")

    print(f"\n✓ Successfully uploaded all {num_buffers} buffers ({total_gb:.2f} GB)")

    # Validation phase with data display
    print("\nValidating buffers...")
    failed = 0
    for i, (buf, expected_pattern) in enumerate(zip(buffers, patterns)):
        downloaded = buf.download(shape=(block_size // 4,), dtype=np.float32)

        # Show first, middle, and last values
        mid_idx = len(downloaded) // 2
        first_val = downloaded[0]
        mid_val = downloaded[mid_idx]
        last_val = downloaded[-1]

        all_match = np.allclose(downloaded, expected_pattern, atol=1e-6)

        if (i + 1) % 10 == 0 or not all_match:
            status = "✓" if all_match else "✗"
            print(f"  {status} Buffer {i:3d}: Expected={expected_pattern:.1f}, "
                  f"Got=[first={first_val:.6f}, mid={mid_val:.6f}, last={last_val:.6f}]")

        if not all_match:
            failed += 1

    if failed == 0:
        print(f"\n✓ All {num_buffers} buffers validated successfully")
    else:
        print(f"\n✗ {failed}/{num_buffers} buffers failed validation")

    # Cleanup
    for buf in buffers:
        buf.free()

    print("✓ Test completed")


if __name__ == "__main__":
    run()