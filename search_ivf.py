import pickle
import time
import statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ---- Load model + projection ----
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

print("Loading projection layer...")
weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state = load_file(weights_path)
linear = nn.Linear(768, 128, bias=False)
linear.weight.data = state["linear.weight"]
linear.eval()

# ---- Load corpus ----
print("Loading corpus...")
with open("data/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
texts = corpus["texts"]
doc_vectors = corpus["vectors"]

all_doc_vectors = torch.cat(doc_vectors, dim=0)
NUM_DOCS = len(doc_vectors)

doc_offsets = [0]
for vec in doc_vectors:
    doc_offsets.append(doc_offsets[-1] + vec.shape[0])

token_to_doc = torch.empty(all_doc_vectors.shape[0], dtype=torch.long)
for i in range(NUM_DOCS):
    start, end = doc_offsets[i], doc_offsets[i + 1]
    token_to_doc[start:end] = i

# ---- Load IVF index ----
print("Loading IVF index...")
with open("data/ivf_index.pkl", "rb") as f:
    index = pickle.load(f)

centroids = torch.from_numpy(index["centroids"]).float()        # [K, 128]
posting_lists = index["posting_lists"]                           # list of K arrays
K = index["K"]
print(f"K={K}, corpus tokens={all_doc_vectors.shape[0]}")

# ---- Query encoder ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1)

# ---- Brute force search (ground truth) ----
def search_bruteforce(query_text, top_k=10):
    Q = encode_query(query_text)
    sim_all = Q @ all_doc_vectors.T
    num_q = sim_all.shape[0]
    
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_all.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc.unsqueeze(0).expand(num_q, -1),
        src=sim_all,
        reduce="amax",
        include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    return list(zip(top_indices.tolist(), top_values.tolist()))

# ---- IVF search ----
def search_ivf(query_text, nprobe=8, top_k=10):
    """
    1. Encode query.
    2. For each query token, find the nprobe closest centroids.
    3. Union all token IDs in those clusters → candidate token set.
    4. Score only the candidate tokens (not the full corpus).
    """
    Q = encode_query(query_text)                          # [num_q, 128]
    num_q = Q.shape[0]
    
    # Step 1: distances from each query token to every centroid
    centroid_sim = Q @ centroids.T                        # [num_q, K]
    
    # Step 2: top nprobe centroids per query token
    _, top_centroid_ids = centroid_sim.topk(nprobe, dim=1)   # [num_q, nprobe]
    
    # Step 3: union all token IDs from those clusters
    # Flatten and dedupe to get the candidate token set
    probed_clusters = top_centroid_ids.flatten().unique().tolist()
    candidate_token_ids = np.concatenate([posting_lists[c] for c in probed_clusters])
    candidate_token_ids = np.unique(candidate_token_ids)   # dedupe
    
    n_candidates = len(candidate_token_ids)
    
    # Step 4: score query against ONLY the candidate tokens
    candidate_token_ids_t = torch.from_numpy(candidate_token_ids).long()
    candidate_vectors = all_doc_vectors[candidate_token_ids_t]    # [n_candidates, 128]
    candidate_doc_ids = token_to_doc[candidate_token_ids_t]       # [n_candidates]
    
    sim = Q @ candidate_vectors.T                                  # [num_q, n_candidates]
    
    # Group by doc and take max + sum
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=candidate_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=sim,
        reduce="amax",
        include_self=True,
    )
    
    # Some docs got NO candidate tokens — their score stays at -inf, won't appear in top-k
    scores = per_doc_max.sum(dim=0)
    
    # Find top-k among docs that actually got scored
    top_values, top_indices = scores.topk(top_k)
    
    return (
        list(zip(top_indices.tolist(), top_values.tolist())),
        n_candidates,                       # how many tokens we scored
        len(probed_clusters),               # how many clusters we hit
    )

# ---- Quick smoke test ----
print("\n" + "=" * 60)
print("Smoke test: brute force vs IVF on one query")
print("=" * 60)

query = "history of the manhattan project"

bf_results = search_bruteforce(query, top_k=10)
print(f"\nBrute force top 5 for {query!r}:")
for rank, (doc_id, score) in enumerate(bf_results[:5], 1):
    print(f"  {rank}. [{score:.2f}] {texts[doc_id][:90]}...")

ivf_results, n_cands, n_probed = search_ivf(query, nprobe=8, top_k=10)
print(f"\nIVF (nprobe=8) top 5: probed {n_probed} clusters, "
      f"scored {n_cands}/{all_doc_vectors.shape[0]} tokens "
      f"({100*n_cands/all_doc_vectors.shape[0]:.1f}% of corpus)")
for rank, (doc_id, score) in enumerate(ivf_results[:5], 1):
    print(f"  {rank}. [{score:.2f}] {texts[doc_id][:90]}...")

bf_ids = [d for d, _ in bf_results]
ivf_ids = [d for d, _ in ivf_results]
overlap = len(set(bf_ids) & set(ivf_ids))
print(f"\nRecall@10 for this query: {overlap}/10 = {overlap*10}%")

# ---- Recall vs nprobe sweep ----
print("\n" + "=" * 60)
print("Recall@10 vs nprobe sweep")
print("=" * 60)

queries = [
    "history of the manhattan project",
    "vitamin D deficiency symptoms",
    "weather forecast",
    "what causes thunderstorms",
]

# Compute brute-force ground truth once for each query
print("\nComputing ground truth...")
ground_truth = {q: set(d for d, _ in search_bruteforce(q, top_k=10)) for q in queries}

nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

print(f"\n{'nprobe':>7} | {'recall@10':>10} | {'lat (ms)':>9} | {'cands':>7} | {'%corpus':>8}")
print("-" * 60)

# Warm up
for q in queries[:2]:
    search_ivf(q, nprobe=8)
    search_bruteforce(q)

# Brute force latency baseline
bf_times = []
for q in queries:
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        search_bruteforce(q, top_k=10)
        runs.append(time.perf_counter() - t0)
    bf_times.append(statistics.median(runs))
bf_latency = statistics.median(bf_times) * 1000

for nprobe in nprobe_values:
    recalls = []
    times = []
    cand_counts = []
    
    for q in queries:
        # Recall over 3 runs (just average — recall is deterministic)
        results, n_cands, _ = search_ivf(q, nprobe=nprobe, top_k=10)
        retrieved = set(d for d, _ in results)
        recall = len(retrieved & ground_truth[q]) / 10
        recalls.append(recall)
        cand_counts.append(n_cands)
        
        # Latency over 3 runs
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            search_ivf(q, nprobe=nprobe, top_k=10)
            runs.append(time.perf_counter() - t0)
        times.append(statistics.median(runs))
    
    avg_recall = statistics.mean(recalls)
    avg_lat = statistics.median(times) * 1000
    avg_cands = statistics.mean(cand_counts)
    pct_corpus = 100 * avg_cands / all_doc_vectors.shape[0]
    
    print(f"{nprobe:>7} | {avg_recall:>10.2%} | {avg_lat:>9.1f} | "
          f"{int(avg_cands):>7} | {pct_corpus:>7.1f}%")

print(f"\nBrute force baseline: {bf_latency:.1f} ms (recall = 100% by definition)")