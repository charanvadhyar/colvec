"""
Speedup check: compare Rust kernel vs Python+torch on the full SciFact corpus.

Uses realistic inputs:
  - 33 query tokens (median for SciFact queries with MiniLM)
  - 908,493 corpus tokens (full SciFact M=32 index)

Usage:
    uv run python benchmarks/rust_speedup_check.py
"""
import numpy as np
import torch
import pickle
import time
import statistics
import colvec_kernel

# ---- Load the M=32 PQ index ----
print("Loading PQ index...")
with open("data/scifact_pq_m32.pkl", "rb") as f:
    pq_data = pickle.load(f)

codes     = pq_data["codes"]       # [N_tokens, M] uint8
codebooks = pq_data["codebooks"]   # [M, K, chunk_dim] float32
M         = pq_data["M"]
K         = pq_data["K"]
D         = pq_data["D"]
chunk_dim = D // M
N_tokens  = codes.shape[0]

print(f"  {N_tokens} tokens, M={M}, K={K}")

# Pre-transpose codes once at load time — Rust kernel expects [M, N_tokens]
print("Pre-transposing codes...")
codes_t  = np.ascontiguousarray(codes.T)            # [M, N_tokens]
codes_pt = [torch.from_numpy(codes[:, m].astype(np.int64)) for m in range(M)]

# ---- Build realistic query inputs ----
# num_q=33 is the median query token count for SciFact (from Day 12 diagnostic)
NUM_Q = 33
np.random.seed(42)
Q          = np.random.rand(NUM_Q, D).astype(np.float32)
Q_chunks   = Q.reshape(NUM_Q, M, chunk_dim)
lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, codebooks).astype(np.float32)
lookups_t  = torch.from_numpy(lookups_np)

print(f"  Query shape: [{NUM_Q}, {D}]")
print(f"  Lookup table: {lookups_np.shape}")
print()

N_RUNS = 5

# ---- Python+torch baseline ----
print(f"Python+torch baseline ({N_RUNS} runs)...")
times_py = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    sim = torch.zeros((NUM_Q, N_tokens), dtype=torch.float32)
    for m in range(M):
        sim += lookups_t[:, m, codes_pt[m]]
    elapsed = (time.perf_counter() - t0) * 1000
    times_py.append(elapsed)
    print(f"  run {i+1}: {elapsed:.1f} ms")

# ---- Rust kernel ----
print(f"\nRust kernel — cache-friendly layout ({N_RUNS} runs)...")
times_rs = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    sim = colvec_kernel.apply_pq_lookups(lookups_np, codes_t)
    elapsed = (time.perf_counter() - t0) * 1000
    times_rs.append(elapsed)
    print(f"  run {i+1}: {elapsed:.1f} ms")

# ---- Summary ----
py_median = statistics.median(times_py)
rs_median = statistics.median(times_rs)
speedup   = py_median / rs_median

print()
print("=" * 50)
print(f"Python+torch:        {py_median:.1f} ms")
print(f"Rust (naive):        {rs_median:.1f} ms")
print(f"Speedup:             {speedup:.1f}x")
print()

if speedup >= 5:
    print("✓ Good — cache-friendly layout is working.")
    print("  Next step: add AVX2 SIMD for another 4-8x.")
elif speedup >= 2:
    print("~ Partial improvement. Check that codes.T is contiguous.")
    print("  Run: print(np.iscontiguousarray(codes_t)) — should be True")
else:
    print("✗ No improvement or regression.")
    print("  Most likely cause: codes_t is not contiguous in memory.")
    print("  Fix: codes_t = np.ascontiguousarray(codes.T)")
    print(f"  codes_t is contiguous: {codes_t.flags['C_CONTIGUOUS']}")