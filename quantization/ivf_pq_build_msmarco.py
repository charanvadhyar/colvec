"""
Build IVF + PQ index for the MS MARCO corpus (the bigger one).
Same architecture as scifact build, just different input/output paths
and slightly larger K_IVF since the corpus is bigger.
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
# K_IVF scales with corpus size. sqrt(2.32M tokens) ≈ 1525, round to 1024.
K_IVF = 1024
M_PQ = 32
K_PQ = 256
SAMPLE_SIZE = 200_000

INPUT_PATH = "data/corpus.pkl"
OUTPUT_PATH = "data/msmarco_ivf_pq.pkl"


# ---- Load ----
print("Loading MS MARCO corpus...")
with open(INPUT_PATH, "rb") as f:
    corpus = pickle.load(f)
texts = corpus["texts"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS = len(texts)

all_vectors = torch.cat(doc_vectors_list, dim=0).numpy()
N_tokens, D = all_vectors.shape
print(f"  {NUM_DOCS} docs, {N_tokens} tokens, dim={D}")

doc_offsets = [0]
for v in doc_vectors_list:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc = np.empty(N_tokens, dtype=np.int32)
for i in range(NUM_DOCS):
    token_to_doc[doc_offsets[i]:doc_offsets[i+1]] = i


# ---- Train IVF ----
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
ivf_centroids = ivf.cluster_centers_.astype(np.float32)
ivf_assignments = ivf.labels_
print(f"  trained in {time.perf_counter()-t0:.1f}s")

print("Building IVF posting lists...")
posting_lists = [[] for _ in range(K_IVF)]
for token_id, cluster_id in enumerate(ivf_assignments):
    posting_lists[cluster_id].append(token_id)
posting_lists = [np.array(pl, dtype=np.int32) for pl in posting_lists]

sizes = [len(pl) for pl in posting_lists]
print(f"  cluster sizes: min={min(sizes)}, median={int(np.median(sizes))}, "
      f"max={max(sizes)}")


# ---- Train PQ ----
print(f"\nTraining PQ with M={M_PQ}, K={K_PQ}...")
t0 = time.perf_counter()
pq_codebooks = train_pq(all_vectors, M=M_PQ, K=K_PQ, sample_size=SAMPLE_SIZE)
pq_codes = encode_with_pq(all_vectors, pq_codebooks)
print(f"  trained + encoded in {time.perf_counter()-t0:.1f}s")


# ---- Save ----
out_path = Path(OUTPUT_PATH)
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
print(f"  IVF:    {ivf_size/1e6:.1f} MB")
print(f"  PQ:     {pq_size/1e6:.1f} MB")
print(f"  Lookup: {token_to_doc.nbytes/1e6:.1f} MB")
print(f"  Total:  {total_mb:.1f} MB  (uncompressed: {uncompressed_mb:.0f} MB)")
print(f"  Effective compression: {uncompressed_mb/total_mb:.1f}×")
print(f"\nSaved to {out_path}")