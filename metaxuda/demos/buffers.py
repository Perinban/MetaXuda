import metaxuda
import numpy as np

def run():
    x = np.arange(5, dtype=np.float32)
    buf = metaxuda.GPUMemoryBuffer(length=x.size, dtype=x.dtype)
    buf.upload(x)
    result = buf.download()

    print("Input: ", x)
    print("Output:", result)

    buf.free()

if __name__ == "__main__":
    run()
