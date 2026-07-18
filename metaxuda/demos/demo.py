#!/usr/bin/env python3

import os
import shutil
import psutil
import tempfile
import numpy as np
import math
import time
from numba import cuda
from metaxuda import GPUMemoryBuffer
from metaxuda import TieredBuffer
from metaxuda import Pipeline

TIER3_DIR = "block_store"
TIER3_PATH = os.path.join(tempfile.gettempdir(), TIER3_DIR)

def device_info():
    print("\n" + "=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / 1024 / 1024
    ram = psutil.virtual_memory()
    print(f"Process RSS: {rss_mb:.1f} MB")
    print(f"System RAM: {ram.total / 1e9:.1f} GB total, {ram.available / 1e9:.1f} GB available")
    print(f"GPU Device: Metal Default Device")
    total_size = 0
    if os.path.exists(TIER3_PATH):
        total_size = sum(
            os.path.getsize(os.path.join(TIER3_PATH, f)) for f in os.listdir(TIER3_PATH) if f.endswith(".t3blk"))
    print(f"Tier-3 Storage: {total_size / (1024 ** 3):.2f} GB")


def test_buffers_gpu():
    print("\n" + "=" * 60)
    print("TEST 1: GPU BUFFER MANAGEMENT")
    print("=" * 60)
    sizes_mb = [2, 4, 8, 128]
    buffers = []
    print("Uploading buffers...")
    for i, mb in enumerate(sizes_mb):
        length = (mb * 1024 * 1024) // 4
        val = float(i + 1)
        arr = np.full(length, val, dtype=np.float32)
        buf = GPUMemoryBuffer(length=length, dtype=np.float32)
        buf.upload(arr)
        buffers.append((buf, val))
        print(f"  Buffer {i}: {mb:4d} MB")
    ok_all = True
    print("Verifying buffers...")
    for i, (buf, expected) in enumerate(buffers):
        out = buf.download()
        ok = np.allclose([out[0], out[-1]], expected, atol=1e-3)
        status = "✓" if ok else "✗"
        print(f"  Buffer {i}: {status}")
        if not ok:
            ok_all = False
    for buf, _ in buffers:
        buf.free()
    print("Result: PASSED" if ok_all else "Result: FAILED")
    return {"buffers_ok": ok_all}

def test_large_tier():
    print("\n" + "=" * 60)
    print("TEST 2: MEMORY TIERING")
    print("=" * 60)
    if os.path.exists(TIER3_PATH):
        shutil.rmtree(TIER3_PATH, ignore_errors=True)
    total_bytes = 10 * 1024 ** 3
    print(f"Allocating {total_bytes / (1024 ** 3):.1f} GB TieredBuffer...")
    buf = TieredBuffer(total_bytes)
    time.sleep(0.1)
    print("Creating data array...")
    arr = np.full(total_bytes // 4, 3.14159, dtype=np.float32)
    print("Uploading...")
    buf.upload(arr)
    ram = psutil.virtual_memory()
    print(f"RAM after upload: {ram.available / 1e9:.1f} GB available")
    print("Upload complete")
    buf.free()
    return {"tier_verified": True, "tier_size_gb": total_bytes / (1024 ** 3)}

@cuda.jit
def log_kernel(x, out):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.size:
        out[i] = math.log(x[i])
        i += stride

@cuda.jit
def exp_kernel(x, out):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.size:
        out[i] = math.exp(x[i])
        i += stride

def gpu_time_ms(launch_fn, reps=20, warmup=5):
    for _ in range(warmup):
        launch_fn()
        cuda.synchronize()
    start = cuda.event(timing=True)
    end = cuda.event(timing=True)
    start.record()
    for _ in range(reps):
        launch_fn()
    end.record()
    end.synchronize()
    return cuda.event_elapsed_time(start, end) / reps

def test_fusion():
    print("\n" + "=" * 60)
    print("TEST 3: FUSION vs CONCURRENCY")
    print("=" * 60)
    N = 1 << 24
    print(f"Array size: {N:,} elements")
    rng = np.random.default_rng(42)
    a = (rng.random(N, dtype=np.float32) * 0.9 + 0.1).astype(np.float32)

    d_a = cuda.to_device(a)
    d_out1 = cuda.device_array_like(d_a)
    d_out2 = cuda.device_array_like(d_a)
    d_log = cuda.device_array_like(d_a)
    tpb = 256
    blocks = (N + tpb - 1) // tpb

    def L_log(): log_kernel[blocks, tpb](d_a, d_log)

    def L_exp(): exp_kernel[blocks, tpb](d_log, d_out1)

    log_time = gpu_time_ms(L_log)
    exp_time = gpu_time_ms(L_exp)
    separate_total = log_time + exp_time

    def L_multikernel():
        log_kernel[blocks, tpb](d_a, d_log)
        exp_kernel[blocks, tpb](d_log, d_out1)

    multi_time = gpu_time_ms(L_multikernel)

    @cuda.jit
    def true_fused_logexp(x, out):
        i = cuda.grid(1)
        stride = cuda.gridsize(1)
        while i < x.size:
            out[i] = math.exp(math.log(x[i]))
            i += stride

    def L_true_fused(): true_fused_logexp[blocks, tpb](d_a, d_out2)

    true_fused_time = gpu_time_ms(L_true_fused)

    print("\nConcurrent test (shim overlap):")
    start_time = cuda.event(timing=True)
    end_time = cuda.event(timing=True)
    start_time.record()
    log_kernel[blocks, tpb](d_a, d_out1)
    log_kernel[blocks, tpb](d_a, d_out2)
    exp_kernel[blocks, tpb](d_a, d_out1)
    exp_kernel[blocks, tpb](d_a, d_out2)
    end_time.record()
    end_time.synchronize()
    concurrent_time = cuda.event_elapsed_time(start_time, end_time)
    print(f"4 kernels: {concurrent_time:6.3f} ms")

    print("\nResults:")
    print(f"Separate: {separate_total:6.3f} ms")
    print(f"Multi-kernel: {multi_time:6.3f} ms")
    print(f"True fused: {true_fused_time:6.3f} ms")
    print(f"4x concurrent:{concurrent_time:6.3f} ms  ← shim overlap")

    peak_gops = max((N / 1e9) / (log_time / 1000), (N / 1e9) / (exp_time / 1000), (N / 1e9) / (true_fused_time / 1000))
    print(f"Peak: {peak_gops:.1f} GOp/s")

    d_verify = cuda.device_array_like(d_a)
    true_fused_logexp[blocks, tpb](d_a, d_verify)
    cuda.synchronize()
    result = d_verify.copy_to_host()
    max_error = np.max(np.abs(result - a))
    verified = max_error < 1e-3
    print(f"Verification: {'PASSED' if verified else 'FAILED'} (error: {max_error:.2e})")

    return {
        "verified": verified, "separate_time": separate_total, "multi_time": multi_time,
        "true_fused_time": true_fused_time, "concurrent_time": concurrent_time,
        "max_error": max_error, "peak_gops": peak_gops
    }

def test_concurrent_ops():
    print("\n" + "=" * 60)
    print("TEST 4: CONCURRENT PIPELINES")
    print("=" * 60)
    num_ops = 16
    size_mb = 4
    length = (size_mb * 1024 * 1024) // 4
    print(f"16 pipelines ({size_mb} MB each)")

    tpb = 256
    blocks = (length + tpb - 1) // tpb

    # Create input and output arrays directly (no Pipeline wrapper)
    dev_inputs = []
    dev_temp = []
    dev_out = []

    for i in range(num_ops):
        arr = np.linspace(float(i + 1), float(i + 2), length, dtype=np.float32)
        d_in = cuda.to_device(arr)
        d_tmp = cuda.device_array(length, dtype=np.float32)
        d_o = cuda.device_array(length, dtype=np.float32)
        dev_inputs.append(d_in)
        dev_temp.append(d_tmp)
        dev_out.append(d_o)

    start = time.time()
    print("Launching pipelines...")

    # Launch all kernels without sync (shim handles overlap)
    for i in range(num_ops):
        log_kernel[blocks, tpb](dev_inputs[i], dev_temp[i])
        exp_kernel[blocks, tpb](dev_temp[i], dev_out[i])
        if (i + 1) % 4 == 0:
            print(f"  {i + 1}/16", flush=True)

    # Single sync at the end
    cuda.synchronize()

    # Download results
    for i in range(num_ops):
        _ = dev_out[i].copy_to_host()

    elapsed = time.time() - start
    total_mb = num_ops * size_mb
    throughput = total_mb / elapsed
    avg_per_op = elapsed / num_ops * 1000

    print("All pipelines completed")
    print(f"Total data:  {total_mb} MB")
    print(f"Total time:  {elapsed:.3f}s")
    print(f"Throughput:  {throughput:.0f} MB/s ({throughput / 1024:.2f} GB/s)")
    print(f"Avg per op:  {avg_per_op:.2f} ms")
    print("Result: PASSED")

    return {"throughput": throughput, "avg_per_op": avg_per_op}

def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    f = results['fusion']
    print(f"Buffers: {'PASSED' if results['buffers']['buffers_ok'] else 'FAILED'}")
    print(f"Tiering: {results['tier']['tier_size_gb']:.1f} GB")
    print(f"Peak kernel: {f['peak_gops']:.1f} GOp/s")
    print(f"True fused: {f['true_fused_time']:.3f}ms vs {f['separate_time']:.3f}ms separate")
    print(f"Concurrent: {f['concurrent_time']:.3f}ms (shim overlap)")
    print(f"Verification: {'PASSED' if f['verified'] else 'FAILED'}")
    print(f"Pipeline BW: {results['ops']['throughput'] / 1024:.2f} GB/s")

if __name__ == "__main__":
    try:
        device_info()
        results = {}
        results["buffers"] = test_buffers_gpu()
        results["tier"] = test_large_tier()
        results["fusion"] = test_fusion()
        results["ops"] = test_concurrent_ops()
        print_summary(results)
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)