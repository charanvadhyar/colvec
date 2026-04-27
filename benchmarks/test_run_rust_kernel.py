"""
Correctness check: Rust kernel output must match Python reference exactly.
Run this after every rebuild of the Rust kernel before benchmarking.

Usage:
    uv run python benchmarks/test_rust_kernel.py
"""
import numpy as np
import torch
import pickle
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

print(f"  {N_tokens} tokens, M={M}, K={K}, chunk_dim={chunk_dim}")

# Pre-transpose codes once — Rust kernel expects [M, N_tokens]
codes_t = np.ascontiguousarray(codes.T)   # [M, N_tokens]

# ---- Build a fake query (3 query tokens, realistic size) ----
np.random.seed(42)
Q          = np.random.rand(3, D).astype(np.float32)
Q_chunks   = Q.reshape(3, M, chunk_dim)
lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, codebooks).astype(np.float32)

# ---- Python reference implementation ----
print("Computing Python reference...")
lookups_torch = torch.from_numpy(lookups_np)
codes_torch   = [torch.from_numpy(codes[:, m].astype(np.int64)) for m in range(M)]

sim_python = torch.zeros((3, N_tokens), dtype=torch.float32)
for m in range(M):
    sim_python += lookups_torch[:, m, codes_torch[m]]
sim_python = sim_python.numpy()

# ---- Rust kernel ----
print("Computing Rust kernel output...")
sim_rust = colvec_kernel.apply_pq_lookups(lookups_np, codes_t)

# ---- Compare ----
print()
max_diff  = np.abs(sim_python - sim_rust).max()
mean_diff = np.abs(sim_python - sim_rust).mean()
print(f"Max absolute difference:  {max_diff:.2e}")
print(f"Mean absolute difference: {mean_diff:.2e}")

if max_diff < 1e-4:
    print("✓ Rust kernel output matches Python reference")
else:
    print("✗ MISMATCH — check the kernel implementation")
    # Show which positions differ
    diff_mask = np.abs(sim_python - sim_rust) > 1e-4
    n_diffs = diff_mask.sum()
    print(f"  {n_diffs} positions differ by more than 1e-4")
    print(f"  First differing positions: {np.argwhere(diff_mask)[:5]}")