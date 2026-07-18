import numpy as np
import metaxuda

# ------------------------------------------------
# Demo
# ------------------------------------------------
def run():
    x = np.linspace(0, 1, 10, dtype=np.float32)

    buf = metaxuda.GPUMemoryBuffer(x)
    print("Buffer allocated on GPU.")

    print("Syncing default stream…")
    metaxuda.DEFAULT_STREAM.sync()
    print("Stream sync complete.")

    buf.free()

# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":
    run()
