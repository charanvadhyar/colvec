"""
SciFact benchmark using PQ-compressed corpus.
Same as scifact_benchmark.py but compresses the corpus 32x.
"""
import pickle
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ir_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ---- Setup model ----
print("Setting up model...")
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state = load_file(weights_path)
linear = nn.Linear(768, 128, bias=False)
linear.weight.data = state["linear.weight"]
linear.eval()

device = "cpu"   # PQ scoring is CPU-bound numpy work anyway
model = model.to(device)
linear = linear.to(device)

# ---- Load PQ index ----
print("Loading PQ-compressed corpus...")
with open("data/scifact_pq.pkl", "rb") as f:
    pq_data = pickle.load(f)

codebooks = pq_data["codebooks"]    # [M, K, chunk_dim] float32
codes = pq_data["codes"]            # [N_tokens, M] uint8
M = pq_data["M"]
K = pq_data["K"]
D = pq_data["D"]
chunk_dim = D // M

print(f"  M={M}, K={K}, D={D}, chunk_dim={chunk_dim}")
print(f"  Compressed corpus: {len(codes)} tokens × {M} bytes = {codes.nbytes/1e6:.1f} MB")

# ---- Load doc structure (need this to map tokens → docs) ----
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

doc_ids = corpus["doc_ids"]
doc_vectors = corpus["vectors"]    # only used to build offsets
NUM_DOCS = len(doc_ids)

# Build token → doc lookup
doc_offsets = [0]
for v in doc_vectors:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc = np.empty(len(codes), dtype=np.int32)
for i in range(NUM_DOCS):
    token_to_doc[doc_offsets[i]:doc_offsets[i+1]] = i

del doc_vectors   # free memory — we don't need uncompressed vectors anymore!

# ---- Query encoder (same as before) ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1).numpy()    # numpy now, not torch

# ---- The PQ scoring function — this is the core of Phase 3 ----
def search_pq(query_text, top_k=100):
    """
    Search using PQ-compressed corpus.
    
    For each query token:
      1. Build a [M, K] lookup table: distance from query's chunk m
         to each of K centroids in codebook m.
      2. To score a corpus token, sum lookups[m, codes[token, m]] for m in 0..M-1.
         This replaces a 128-dim dot product with M=16 lookups + 15 adds.
    """
    Q = encode_query(query_text)    # [num_q_tokens, D]
    num_q = Q.shape[0]
    
    # ---- Build lookup tables ----
    # For each query token and each chunk m, compute dot products against
    # all K codebook entries. Result: [num_q, M, K].
    # 
    # Using dot product (not Euclidean) because vectors are L2-normalized
    # and we want similarity, not distance. Higher = better.
    lookups = np.zeros((num_q, M, K), dtype=np.float32)
    for m in range(M):
        # Query chunk: [num_q, chunk_dim]
        # Codebook m:  [K, chunk_dim]
        # Dot product: [num_q, K]
        q_chunk = Q[:, m * chunk_dim : (m + 1) * chunk_dim]
        lookups[:, m, :] = q_chunk @ codebooks[m].T
    
    # ---- Score every corpus token using lookups ----
    # codes is [N_tokens, M] of uint8 indices.
    # For each token n and chunk m, look up lookups[:, m, codes[n, m]].
    # Then sum across m to get the per-query-token similarity to that token.
    # 
    # Vectorized version:
    # Want: scores[q, n] = sum over m of lookups[q, m, codes[n, m]]
    # Use fancy indexing: lookups[:, np.arange(M), codes[n, :]] gives [num_q, M]
    # Sum along M gives [num_q]. Do this for all n at once with broadcasting.
    
    # Shape: lookups[:, m, codes[:, m]] has shape [num_q, N_tokens] for each m
    # Sum across m gives [num_q, N_tokens]
    sim = np.zeros((num_q, len(codes)), dtype=np.float32)
    for m in range(M):
        # codes[:, m] is [N_tokens], use it to index into lookups[:, m, :]
        sim += lookups[:, m, codes[:, m]]   # [num_q, N_tokens]
    
    # ---- MaxSim aggregation: per-doc max, then sum across query tokens ----
    # Same as before, but now in numpy.
    #per_doc_max = np.full((num_q, NUM_DOCS), -np.inf, dtype=np.float32)
    #np.maximum.at(per_doc_max, (slice(None), token_to_doc[None, :]), sim)
    # Note: np.maximum.at is the numpy equivalent of scatter_reduce_(reduce="amax")
    # It's slow due to lack of vectorization in older numpy versions; fine for
    # benchmarking, would need optimization for production.
    
    # Actually the np.maximum.at indexing above is wrong. Let me redo it
    # with a simpler loop that's also more readable:
    per_doc_max = np.full((num_q, NUM_DOCS), -np.inf, dtype=np.float32)
    for doc_id in range(NUM_DOCS):
        token_slice = sim[:, doc_offsets[doc_id]:doc_offsets[doc_id+1]]
        if token_slice.shape[1] > 0:
            per_doc_max[:, doc_id] = token_slice.max(axis=1)
    
    scores = per_doc_max.sum(axis=0)   # [NUM_DOCS]
    
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(doc_ids[i], scores[i]) for i in top_indices]

# ---- Metrics (same as before) ----
def dcg_at_k(rel_scores, k):
    rel_scores = rel_scores[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores))

def ndcg_at_k(retrieved, relevant, k):
    rel_scores = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg = dcg_at_k(rel_scores, k)
    n_rel = min(len(relevant), k)
    idcg = dcg_at_k([1] * n_rel + [0] * (k - n_rel), k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

# ---- Load qrels ----
print("\nLoading queries and qrels...")
dataset = ir_datasets.load("beir/scifact/test")
queries = list(dataset.queries_iter())
qrels_per_query = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels_per_query[qrel.query_id].add(qrel.doc_id)

queries = [q for q in queries if q.query_id in qrels_per_query]
print(f"Evaluating on {len(queries)} queries")

# ---- Run benchmark ----
print("\nRunning PQ search on all queries...")
import statistics

ndcgs = []
recalls = []
latencies = []

for q in tqdm(queries):
    relevant = qrels_per_query[q.query_id]
    
    t0 = time.perf_counter()
    results = search_pq(q.text, top_k=100)
    latencies.append((time.perf_counter() - t0) * 1000)
    
    retrieved = [d for d, _ in results]
    ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
    recalls.append(recall_at_k(retrieved, relevant, 100))

# ---- Report ----
mean_ndcg = statistics.mean(ndcgs)
mean_recall = statistics.mean(recalls)
median_lat = statistics.median(latencies)

print("\n" + "=" * 60)
print("SciFact results — PQ-compressed MaxSim")
print("=" * 60)
print(f"  Queries evaluated:  {len(queries)}")
print(f"  M={M}, K={K}, compression: 32x")
print(f"  nDCG@10:            {mean_ndcg:.4f}")
print(f"  Recall@100:         {mean_recall:.4f}")
print(f"  Median latency:     {median_lat:.1f} ms")
print()
print(f"  Brute force baseline (Day 11):  nDCG@10 = 0.6122")
print(f"  PQ result:                      nDCG@10 = {mean_ndcg:.4f}")
print(f"  nDCG drop from compression:     {0.6122 - mean_ndcg:+.4f}")
print()

# ---- Persist ----
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system": "pq_maxsim",
    "M": M, "K": K, "compression_ratio": 32,
    "dataset": "scifact",
    "num_queries": len(queries),
    "ndcg_at_10": mean_ndcg,
    "recall_at_100": mean_recall,
    "median_latency_ms": median_lat,
}
with open("benchmarks/results/scifact_history.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")

print(f"Saved to benchmarks/results/scifact_history.jsonl")