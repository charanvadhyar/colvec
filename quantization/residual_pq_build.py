"""
Build residual PQ index on the chunked SciFact corpus.

Standard PQ compresses raw token vectors.
Residual PQ compresses the RESIDUAL — the difference between each
token vector and its nearest IVF centroid.

Why residuals compress better:
  - Raw ColBERT vectors span a large space with non-uniform distribution
  - Residuals are the "error" after centroid assignment — smaller magnitude,
    more uniformly distributed, easier for PQ codebooks to cover densely
  - Same M=32, K=256 PQ budget → better recall at same compression ratio

Pipeline:
  1. Train IVF centroids (k-means on all token vectors)
  2. Assign each token to its nearest centroid
  3. Compute residual: residual[i] = vector[i] - centroid[assignment[i]]
  4. Train PQ codebooks on residuals (not raw vectors)
  5. Encode residuals with trained codebooks → codes [N_tokens, M]

At query time:
  approximate_vector = centroid + decompress(codes)
  But we never actually decompress — we use asymmetric scoring:
  score = dot(query_chunk, centroid_chunk) + lookup(query_chunk, residual_code)

Output: data/scifact_residual_pq.pkl
"""
import pickle
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from quantization.pq_train import train_pq, encode_with_pq


# ---- Config ----
K_IVF        = 256     # IVF clusters — same as before on SciFact
M_PQ         = 32      # PQ subquantizers — sweet spot from Day 15
K_PQ         = 256     # codebook entries per chunk
SAMPLE_SIZE  = 200_000 # training sample for both IVF and PQ
OUT_PATH     = Path("data/scifact_residual_pq.pkl")


# ---- Load chunked corpus ----
print("Loading chunked SciFact corpus...")
with open("data/scifact_corpus_chunked.pkl", "rb") as f:
    corpus = pickle.load(f)

doc_ids        = corpus["doc_ids"]
all_chunk_vecs = corpus["vectors"]       # list of [n_tok, 128] per chunk
chunk_counts   = corpus["chunk_counts"]
token_to_doc   = corpus["token_to_doc"]  # [N_tokens] int32
NUM_DOCS       = len(doc_ids)

# Stack all token vectors
all_vectors = np.concatenate(
    [v for v in all_chunk_vecs], axis=0
).astype(np.float32)                     # [N_tokens, 128]

N_tokens, D = all_vectors.shape
print(f"  {NUM_DOCS} docs, {N_tokens} tokens, dim={D}")

# Build doc_offsets for posting list construction
doc_offsets = [0]
for v in all_chunk_vecs:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])


# ================================================================
# Step 1: Train IVF centroids
# ================================================================
print(f"\nStep 1: Training IVF with K={K_IVF}...")
t0 = time.perf_counter()

rng = np.random.default_rng(42)
if N_tokens > SAMPLE_SIZE:
    train_idx = rng.choice(N_tokens, size=SAMPLE_SIZE, replace=False)
    train_data = all_vectors[train_idx]
else:
    train_data = all_vectors

ivf = MiniBatchKMeans(
    n_clusters=K_IVF,
    batch_size=4096,
    n_init=5,
    max_iter=150,
    random_state=42,
    verbose=0,
)
ivf.fit(train_data)
ivf_centroids = ivf.cluster_centers_.astype(np.float32)   # [K_IVF, D]

# Assign ALL tokens to their nearest centroid
# Do in batches to avoid OOM on large corpora
print("  Assigning tokens to centroids...")
BATCH = 50_000
assignments = np.empty(N_tokens, dtype=np.int32)
for start in range(0, N_tokens, BATCH):
    end   = min(start + BATCH, N_tokens)
    batch = all_vectors[start:end]
    # Squared distance: ||x - c||² = ||x||² + ||c||² - 2·x·c
    x_sq  = (batch ** 2).sum(axis=1, keepdims=True)        # [B, 1]
    c_sq  = (ivf_centroids ** 2).sum(axis=1, keepdims=True).T  # [1, K]
    cross = batch @ ivf_centroids.T                          # [B, K]
    dists = x_sq + c_sq - 2 * cross                         # [B, K]
    assignments[start:end] = dists.argmin(axis=1).astype(np.int32)

t_ivf = time.perf_counter() - t0
print(f"  IVF trained + assigned in {t_ivf:.1f}s")

# Build posting lists
posting_lists = [[] for _ in range(K_IVF)]
for token_id, cluster_id in enumerate(assignments):
    posting_lists[cluster_id].append(token_id)
posting_lists = [np.array(pl, dtype=np.int32) for pl in posting_lists]

sizes = [len(pl) for pl in posting_lists]
print(f"  Cluster sizes: min={min(sizes)}, "
      f"median={int(np.median(sizes))}, max={max(sizes)}")


# ================================================================
# Step 2: Compute residuals
# ================================================================
print(f"\nStep 2: Computing residuals...")
t0 = time.perf_counter()

# residual[i] = vector[i] - centroid[assignment[i]]
# Do in batches for memory efficiency
residuals = np.empty_like(all_vectors)
for start in range(0, N_tokens, BATCH):
    end    = min(start + BATCH, N_tokens)
    batch  = all_vectors[start:end]
    assign = assignments[start:end]
    residuals[start:end] = batch - ivf_centroids[assign]

t_res = time.perf_counter() - t0
print(f"  Residuals computed in {t_res:.1f}s")
print(f"  Residual magnitude: mean={np.linalg.norm(residuals, axis=1).mean():.4f} "
      f"(raw vector magnitude: {np.linalg.norm(all_vectors, axis=1).mean():.4f})")
print(f"  Compression factor from smaller residuals: "
      f"{np.linalg.norm(all_vectors, axis=1).mean() / np.linalg.norm(residuals, axis=1).mean():.2f}x")


# ================================================================
# Step 3: Train PQ codebooks on residuals
# ================================================================
print(f"\nStep 3: Training PQ on residuals (M={M_PQ}, K={K_PQ})...")
t0 = time.perf_counter()

pq_codebooks = train_pq(
    residuals,
    M=M_PQ,
    K=K_PQ,
    sample_size=SAMPLE_SIZE,
)

# Encode residuals with trained codebooks
pq_codes = encode_with_pq(residuals, pq_codebooks)   # [N_tokens, M] uint8

t_pq = time.perf_counter() - t0
print(f"  PQ trained + encoded in {t_pq:.1f}s")


# ================================================================
# Step 4: Save
# ================================================================
print(f"\nStep 4: Saving index...")
with open(OUT_PATH, "wb") as f:
    pickle.dump({
        # IVF structures
        "ivf_centroids":  ivf_centroids,    # [K_IVF, D]
        "ivf_assignments": assignments,      # [N_tokens] int32
        "posting_lists":  posting_lists,     # list of K_IVF arrays

        # PQ structures (trained on residuals)
        "pq_codebooks":   pq_codebooks,     # [M, K, chunk_dim]
        "pq_codes":       pq_codes,         # [N_tokens, M] uint8

        # Doc structure
        "token_to_doc":   token_to_doc,     # [N_tokens] int32
        "doc_offsets":    doc_offsets,
        "doc_ids":        doc_ids,

        # Config
        "K_IVF": K_IVF,
        "M_PQ":  M_PQ,
        "K_PQ":  K_PQ,
        "D":     D,
        "N_tokens": N_tokens,
        "NUM_DOCS": NUM_DOCS,
    }, f)


# ================================================================
# Stats
# ================================================================
ivf_size = ivf_centroids.nbytes + sum(pl.nbytes for pl in posting_lists)
pq_size  = pq_codes.nbytes + pq_codebooks.nbytes
total_mb = OUT_PATH.stat().st_size / 1e6
raw_mb   = all_vectors.nbytes / 1e6

print(f"\nIndex breakdown:")
print(f"  IVF (centroids + posting lists): {ivf_size/1e6:.1f} MB")
print(f"  PQ  (codebooks + codes):         {pq_size/1e6:.1f} MB")
print(f"  Total index:                     {total_mb:.1f} MB")
print(f"  Raw vectors would be:            {raw_mb:.1f} MB")
print(f"  Effective compression:           {raw_mb/total_mb:.1f}x")
print(f"\nSaved to {OUT_PATH}")