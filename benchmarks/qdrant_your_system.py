"""
Build your own IVF+PQ index on the same MiniLM single-vector embeddings
that Qdrant uses. This makes the latency comparison fair — both systems
run on identical data.
"""
import pickle
import time
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantization.pq_train import train_pq, encode_with_pq

DATA_PATH = "data/scifact_singlevec.pkl"
OUT_PATH  = "data/scifact_singlevec_ivfpq.pkl"

# ---- Load ----
print("Loading MiniLM embeddings...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

doc_ids    = data["doc_ids"]
embeddings = data["embeddings"].astype(np.float32)   # [N_docs, 384]
N, D = embeddings.shape
print(f"  {N} docs, dim={D}")

# Single-vector: each doc IS the vector, so token_to_doc is trivial
# (one vector per doc, index i belongs to doc i)
token_to_doc = np.arange(N, dtype=np.int32)


# ---- IVF ----
# K ≈ sqrt(N) = sqrt(5183) ≈ 72, round up to 128
K_IVF = 128
print(f"\nTraining IVF with K={K_IVF}...")
t0 = time.perf_counter()
ivf = MiniBatchKMeans(n_clusters=K_IVF, batch_size=2048,
                      n_init=5, max_iter=100, random_state=42)
ivf.fit(embeddings)
ivf_centroids = ivf.cluster_centers_.astype(np.float32)
ivf_assignments = ivf.labels_
print(f"  trained in {time.perf_counter()-t0:.2f}s")

posting_lists = [[] for _ in range(K_IVF)]
for doc_id, cluster_id in enumerate(ivf_assignments):
    posting_lists[cluster_id].append(doc_id)
posting_lists = [np.array(pl, dtype=np.int32) for pl in posting_lists]
sizes = [len(pl) for pl in posting_lists]
print(f"  cluster sizes: min={min(sizes)}, median={int(np.median(sizes))}, "
      f"max={max(sizes)}")


# ---- PQ ----
# 384 dims, M=32 → chunk_dim=12. Slightly different from ColBERT's 128-dim.
# M=16 → chunk_dim=24, M=48 → chunk_dim=8 (need 384 % M == 0)
# Use M=48 so chunk_dim=8 (same as ColBERT's sweet spot)
M_PQ, K_PQ = 48, 256
assert D % M_PQ == 0, f"D={D} not divisible by M={M_PQ}"
print(f"\nTraining PQ with M={M_PQ}, K={K_PQ}...")
t0 = time.perf_counter()
pq_codebooks = train_pq(embeddings, M=M_PQ, K=K_PQ, sample_size=N)
pq_codes = encode_with_pq(embeddings, pq_codebooks)
print(f"  trained + encoded in {time.perf_counter()-t0:.2f}s")


# ---- Save ----
with open(OUT_PATH, "wb") as f:
    pickle.dump({
        "ivf_centroids": ivf_centroids,
        "posting_lists": posting_lists,
        "pq_codebooks":  pq_codebooks,
        "pq_codes":      pq_codes,
        "token_to_doc":  token_to_doc,
        "doc_ids":       doc_ids,
        "K_IVF":         K_IVF,
        "M_PQ":          M_PQ,
        "K_PQ":          K_PQ,
        "D":             D,
        "N":             N,
    }, f)

# ---- Stats ----
uncompressed_mb = embeddings.nbytes / 1e6
compressed_mb = (pq_codes.nbytes + pq_codebooks.nbytes) / 1e6
print(f"\nIndex stats:")
print(f"  Vectors uncompressed:  {uncompressed_mb:.1f} MB")
print(f"  PQ compressed:         {compressed_mb:.1f} MB")
print(f"  Compression:           {uncompressed_mb/compressed_mb:.1f}x")
print(f"  Saved to {OUT_PATH}")