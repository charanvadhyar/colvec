import pickle
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

# ---- Load corpus ----
print("Loading corpus...")
with open("data/corpus.pkl", "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
doc_vectors = data["vectors"]
print(f"Loaded {len(texts)} passages.")

# ---- Stack all token vectors into one big array ----
all_doc_vectors = torch.cat(doc_vectors, dim=0)   # [total_tokens, 128]
N, D = all_doc_vectors.shape
print(f"Total tokens: {N}, dim: {D}")

# Build the same token_to_doc lookup we use at query time
doc_offsets = [0]
for vec in doc_vectors:
    doc_offsets.append(doc_offsets[-1] + vec.shape[0])

token_to_doc = np.empty(N, dtype=np.int32)
for doc_id in range(len(doc_vectors)):
    start, end = doc_offsets[doc_id], doc_offsets[doc_id + 1]
    token_to_doc[start:end] = doc_id

# ---- Run k-means ----
# K = 256 (small, power of 2, near sqrt(77000) ~= 278)
# MiniBatchKMeans is much faster than KMeans on large data — same algorithm
# you implemented yesterday, just processes data in batches instead of all at once.
K = 1024 # was 256
print(f"\nRunning k-means with K={K}...")

t0 = time.perf_counter()
kmeans = MiniBatchKMeans(
    n_clusters=K,
    batch_size=4096,
    n_init=3,           # try 3 different inits, keep the best
    max_iter=100,
    random_state=42,
    verbose=0,
)
kmeans.fit(all_doc_vectors.numpy())
t_kmeans = time.perf_counter() - t0
print(f"K-means done in {t_kmeans:.1f}s")

centroids = kmeans.cluster_centers_              # [K, 128]
assignments = kmeans.labels_                     # [N] — which cluster each token belongs to

# ---- Build the inverted file (posting lists) ----
print("\nBuilding posting lists...")
posting_lists = [[] for _ in range(K)]
for token_idx, cluster_id in enumerate(assignments):
    posting_lists[cluster_id].append(token_idx)

# Convert each posting list to a numpy array for fast indexing later
posting_lists = [np.array(pl, dtype=np.int32) for pl in posting_lists]

# ---- Stats ----
sizes = [len(pl) for pl in posting_lists]
print(f"\nCluster size distribution:")
print(f"  min:    {min(sizes)}")
print(f"  median: {int(np.median(sizes))}")
print(f"  max:    {max(sizes)}")
print(f"  mean:   {np.mean(sizes):.1f}")
print(f"  empty:  {sum(1 for s in sizes if s == 0)}")

# ---- Save index to disk ----
out_path = Path("data/ivf_index.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "centroids": centroids,                 # [K, 128]
        "posting_lists": posting_lists,         # list of K numpy arrays
        "assignments": assignments,             # [N] cluster id per token
        "token_to_doc": token_to_doc,           # [N] doc id per token
        "doc_offsets": doc_offsets,             # so we can rebuild stacked structure
        "K": K,
    }, f)
print(f"\nSaved index to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")