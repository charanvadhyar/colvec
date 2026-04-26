"""
Build IVF + PQ index for SciFact.

We store:
  - IVF centroids (for candidate filtering)
  - IVF posting lists (which tokens belong to which cluster)
  - PQ codebooks (for scoring)
  - PQ codes (compressed token vectors)
  - token_to_doc mapping (for MaxSim aggregation)
  
At query time:
  1. Encode query (uncompressed float vectors)
  2. Use IVF to find candidate token IDs (nprobe nearest clusters per query token)
  3. Score those candidates using PQ lookup tables
  4. Aggregate with MaxSim per doc
"""
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantization.pq_train import train_pq, encode_with_pq


# ---- Config ----
K_IVF = 256          # number of IVF clusters (smaller corpus → smaller K)
M_PQ = 32            # PQ subquantizers (16x compression — sweet spot from Day 15)
K_PQ = 256
SAMPLE_SIZE = 100_000


# ---- Load corpus ----
print("Loading SciFact corpus...")
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
doc_ids = corpus["doc_ids"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS = len(doc_ids)

all_vectors = torch.cat(doc_vectors_list, dim=0).numpy()
N_tokens, D = all_vectors.shape
print(f"  {NUM_DOCS} docs, {N_tokens} tokens, dim={D}")

# Build doc_offsets and token_to_doc
doc_offsets = [0]
for v in doc_vectors_list:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc = np.empty(N_tokens, dtype=np.int32)
for i in range(NUM_DOCS):
    token_to_doc[doc_offsets[i]:doc_offsets[i+1]] = i


# ---- Train IVF (k-means over all tokens) ----
print(f"\nTraining IVF with K={K_IVF}...")
t0 = time.perf_counter()
ivf = MiniBatchKMeans(
    n_clusters=K_IVF,
    batch_size=4096,
    n_init=3,
    max_iter=100,
    random_state=42,
)
ivf.fit(all_vectors)
ivf_centroids = ivf.cluster_centers_.astype(np.float32)   # [K_IVF, D]
ivf_assignments = ivf.labels_                              # [N_tokens]
print(f"  trained in {time.perf_counter()-t0:.1f}s")

# Build IVF posting lists: cluster_id → list of token_ids
print("Building IVF posting lists...")
posting_lists = [[] for _ in range(K_IVF)]
for token_id, cluster_id in enumerate(ivf_assignments):
    posting_lists[cluster_id].append(token_id)
posting_lists = [np.array(pl, dtype=np.int32) for pl in posting_lists]

sizes = [len(pl) for pl in posting_lists]
print(f"  cluster sizes: min={min(sizes)}, median={int(np.median(sizes))}, "
      f"max={max(sizes)}")


# ---- Train PQ on the same tokens ----
print(f"\nTraining PQ with M={M_PQ}, K={K_PQ}...")
t0 = time.perf_counter()
pq_codebooks = train_pq(all_vectors, M=M_PQ, K=K_PQ, sample_size=SAMPLE_SIZE)
pq_codes = encode_with_pq(all_vectors, pq_codebooks)
print(f"  trained + encoded in {time.perf_counter()-t0:.1f}s")


# ---- Save everything ----
out_path = Path("data/scifact_ivf_pq.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "ivf_centroids":    ivf_centroids,
        "posting_lists":    posting_lists,
        "pq_codebooks":     pq_codebooks,
        "pq_codes":         pq_codes,
        "token_to_doc":     token_to_doc,
        "doc_offsets":      doc_offsets,
        "K_IVF":            K_IVF,
        "M_PQ":             M_PQ,
        "K_PQ":             K_PQ,
        "D":                D,
    }, f)


# ---- Stats ----
ivf_size = ivf_centroids.nbytes + sum(pl.nbytes for pl in posting_lists)
pq_size = pq_codes.nbytes + pq_codebooks.nbytes
total_size = ivf_size + pq_size + token_to_doc.nbytes
total_mb = total_size / 1e6

uncompressed_mb = all_vectors.nbytes / 1e6

print(f"\nIndex breakdown:")
print(f"  IVF (centroids + posting lists): {ivf_size/1e6:.2f} MB")
print(f"  PQ (codebooks + codes):          {pq_size/1e6:.2f} MB")
print(f"  Token-to-doc lookup:             {token_to_doc.nbytes/1e6:.2f} MB")
print(f"  Total index:                     {total_mb:.2f} MB")
print(f"  Uncompressed corpus would be:    {uncompressed_mb:.1f} MB")
print(f"  Effective compression:           {uncompressed_mb/total_mb:.1f}×")
print(f"\nSaved to {out_path}")