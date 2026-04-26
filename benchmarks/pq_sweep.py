"""
Sweep M ∈ {4, 8, 16, 32, 64} for product quantization.
Train, encode, benchmark, save results.
"""
import pickle
import json
import time
import datetime
import statistics
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

# Reuse training code
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from quantization.pq_train import train_pq, encode_with_pq

# ---- Config ----
M_VALUES = [4, 8, 16, 32, 64]
K = 256

# ---- Setup model (once) ----
print("Loading model...")
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state = load_file(weights_path)
linear = nn.Linear(768, 128, bias=False)
linear.weight.data = state["linear.weight"]
linear.eval()

# ---- Load corpus once ----
print("Loading SciFact corpus...")
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

doc_ids = corpus["doc_ids"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS = len(doc_ids)
all_vectors = torch.cat(doc_vectors_list, dim=0).numpy()
D = all_vectors.shape[1]

doc_offsets = [0]
for v in doc_vectors_list:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc = np.empty(len(all_vectors), dtype=np.int32)
for i in range(NUM_DOCS):
    token_to_doc[doc_offsets[i]:doc_offsets[i+1]] = i

print(f"  {NUM_DOCS} docs, {len(all_vectors)} tokens")

# ---- Load qrels and queries once ----
dataset = ir_datasets.load("beir/scifact/test")
queries = list(dataset.queries_iter())
qrels_per_query = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels_per_query[qrel.query_id].add(qrel.doc_id)
queries = [q for q in queries if q.query_id in qrels_per_query]
print(f"  Evaluating on {len(queries)} queries")


# ---- Helpers ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1).numpy()


def search_pq(query_text, codebooks, codes, top_k=100):
    """Same as Day 14, parameterized by codebooks/codes."""
    M, K, chunk_dim = codebooks.shape
    Q = encode_query(query_text)
    num_q = Q.shape[0]
    
    # Build lookup tables
    lookups = np.zeros((num_q, M, K), dtype=np.float32)
    for m in range(M):
        q_chunk = Q[:, m * chunk_dim : (m + 1) * chunk_dim]
        lookups[:, m, :] = q_chunk @ codebooks[m].T
    
    # Score every corpus token
    sim = np.zeros((num_q, len(codes)), dtype=np.float32)
    for m in range(M):
        sim += lookups[:, m, codes[:, m]]
    
    # Per-doc max + sum across query tokens
    per_doc_max = np.full((num_q, NUM_DOCS), -np.inf, dtype=np.float32)
    for d in range(NUM_DOCS):
        slice_ = sim[:, doc_offsets[d]:doc_offsets[d+1]]
        if slice_.shape[1] > 0:
            per_doc_max[:, d] = slice_.max(axis=1)
    scores = per_doc_max.sum(axis=0)
    
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(doc_ids[i], scores[i]) for i in top_indices]


def dcg_at_k(rel_scores, k):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores[:k]))


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


# ---- Main sweep ----
results = []

print(f"\n{'='*70}")
print(f"PQ M sweep on SciFact (K={K} for all)")
print(f"{'='*70}\n")

for M in M_VALUES:
    if D % M != 0:
        print(f"SKIP M={M}: D={D} not divisible")
        continue
    
    chunk_dim = D // M
    print(f"\n--- M={M} (chunk_dim={chunk_dim}, compression={D*4/M:.1f}x) ---")
    
    # Train codebooks
    t0 = time.perf_counter()
    codebooks = train_pq(all_vectors, M=M, K=K, sample_size=100_000)
    t_train = time.perf_counter() - t0
    
    # Encode corpus
    t0 = time.perf_counter()
    codes = encode_with_pq(all_vectors, codebooks)
    t_encode = time.perf_counter() - t0
    
    # Storage stats
    storage_mb = codes.nbytes / 1e6
    codebook_mb = codebooks.nbytes / 1e6
    compression = all_vectors.nbytes / codes.nbytes
    
    print(f"  Trained in {t_train:.1f}s, encoded in {t_encode:.1f}s")
    print(f"  Storage: {storage_mb:.1f} MB codes + {codebook_mb:.2f} MB codebooks "
          f"(compression: {compression:.1f}x)")
    
    # Benchmark on all queries
    print(f"  Running 300-query benchmark...")
    ndcgs, recalls, latencies = [], [], []
    for q in tqdm(queries, ncols=80, leave=False):
        relevant = qrels_per_query[q.query_id]
        t0 = time.perf_counter()
        res = search_pq(q.text, codebooks, codes, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [d for d, _ in res]
        ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        recalls.append(recall_at_k(retrieved, relevant, 100))
    
    mean_ndcg = statistics.mean(ndcgs)
    mean_recall = statistics.mean(recalls)
    median_lat = statistics.median(latencies)
    
    print(f"  nDCG@10:    {mean_ndcg:.4f}")
    print(f"  Recall@100: {mean_recall:.4f}")
    print(f"  Median lat: {median_lat:.0f} ms")
    
    results.append({
        "M": M, "K": K, "chunk_dim": chunk_dim,
        "compression": compression,
        "storage_mb": storage_mb,
        "ndcg_at_10": mean_ndcg,
        "recall_at_100": mean_recall,
        "median_latency_ms": median_lat,
    })


# ---- Final summary table ----
print(f"\n\n{'='*70}")
print("PQ M sweep summary")
print(f"{'='*70}\n")
print(f"{'M':>4} | {'compress':>9} | {'storage':>9} | {'nDCG@10':>8} | "
      f"{'Recall':>7} | {'lat ms':>7}")
print("-" * 70)
for r in results:
    print(f"{r['M']:>4} | {r['compression']:>7.1f}x  | {r['storage_mb']:>6.1f} MB | "
          f"{r['ndcg_at_10']:>8.4f} | {r['recall_at_100']:>7.4f} | "
          f"{r['median_latency_ms']:>7.0f}")

# Reference line
print("-" * 70)
print(f"  uncompressed (Day 11):       465.2 MB |  0.6122  |  0.8666  |     108")

# ---- Persist ----
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "experiment": "pq_m_sweep",
    "dataset": "scifact",
    "results": results,
}
out_path = Path("benchmarks/results/pq_m_sweep.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(record, f, indent=2)
print(f"\nSaved to {out_path}")