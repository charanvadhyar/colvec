"""
SciFact benchmark with Rust AVX2 PQ kernel.

Same as scifact_pq_benchmark_fast.py (Day 16) but replaces the
Python+torch apply-lookups loop with the Rust AVX2 kernel.

The one change:
  OLD: for m in range(M): sim += lookups_t[:, m, codes_t_per_chunk[m]]
  NEW: sim = colvec_kernel.apply_pq_lookups(lookups_np, codes_transposed)

Everything else — encoding, IVF lookup, scatter_reduce — stays in Python.
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

import colvec_kernel   # the Rust extension


# ---- Setup model ----
print("Setting up model...")
model_name = "colbert-ir/colbertv2.0"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)
model.eval()

weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state  = load_file(weights_path)
linear = nn.Linear(768, 128, bias=False)
linear.weight.data = state["linear.weight"]
linear.eval()

device = "cpu"
model  = model.to(device)
linear = linear.to(device)


# ---- Load PQ index (M=32) ----
print("Loading PQ-compressed corpus...")
PQ_PATH = "data/scifact_pq_m32.pkl"

if not Path(PQ_PATH).exists():
    print(f"  {PQ_PATH} not found, training M=32 PQ...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from quantization.pq_train import train_pq, encode_with_pq

    with open("data/scifact_corpus.pkl", "rb") as f:
        _corpus = pickle.load(f)
    _all = torch.cat(_corpus["vectors"], dim=0).numpy()

    _codebooks = train_pq(_all, M=32, K=256, sample_size=100_000)
    _codes     = encode_with_pq(_all, _codebooks)

    with open(PQ_PATH, "wb") as f:
        pickle.dump({
            "codebooks": _codebooks, "codes": _codes,
            "M": 32, "K": 256, "D": 128,
        }, f)
    print(f"  saved to {PQ_PATH}")

with open(PQ_PATH, "rb") as f:
    pq_data = pickle.load(f)

codebooks = pq_data["codebooks"].astype(np.float32)   # [M, K, chunk_dim]
codes     = pq_data["codes"]                           # [N_tokens, M] uint8
M         = pq_data["M"]
K         = pq_data["K"]
D         = pq_data["D"]
chunk_dim = D // M
N_tokens  = codes.shape[0]

print(f"  M={M}, K={K}, D={D}, chunk_dim={chunk_dim}")
print(f"  Compressed corpus: {N_tokens} tokens × {M} bytes = {codes.nbytes/1e6:.1f} MB")

# ---- Pre-transpose codes ONCE at startup ----
# Rust kernel expects [M, N_tokens] layout for cache-friendly inner loop.
# This is a one-time cost at startup, not per query.
print("Pre-transposing codes for Rust kernel...")
codes_transposed = np.ascontiguousarray(codes.T)   # [M, N_tokens]
print(f"  codes_transposed shape: {codes_transposed.shape}, "
      f"C-contiguous: {codes_transposed.flags['C_CONTIGUOUS']}")


# ---- Load doc structure ----
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
doc_ids         = corpus["doc_ids"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS        = len(doc_ids)

doc_offsets = [0]
for v in doc_vectors_list:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc_np = np.empty(N_tokens, dtype=np.int32)
for i in range(NUM_DOCS):
    token_to_doc_np[doc_offsets[i]:doc_offsets[i+1]] = i
token_to_doc_t = torch.from_numpy(token_to_doc_np).long()

del doc_vectors_list   # free memory


# ---- Query encoder ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1).numpy()


# ---- RUST-powered PQ search ----
def search_pq_rust(query_text, top_k=100):
    Q     = encode_query(query_text)      # [num_q, D]
    num_q = Q.shape[0]

    # Build lookup tables (still in numpy — this is fast already)
    Q_chunks   = Q.reshape(num_q, M, chunk_dim)
    lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, codebooks)  # [num_q, M, K]

    # ── RUST KERNEL: replaces the Python for-m loop ──
    # Input:  lookups [num_q, M, K] float32
    #         codes_transposed [M, N_tokens] uint8
    # Output: sim [num_q, N_tokens] float32
    sim_np = colvec_kernel.apply_pq_lookups(lookups_np, codes_transposed)

    # Per-doc max + sum (scatter_reduce — same as Day 16)
    sim_t = torch.from_numpy(sim_np)
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_t.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc_t.unsqueeze(0).expand(num_q, -1),
        src=sim_t,
        reduce="amax",
        include_self=True,
    )
    scores = per_doc_max.sum(dim=0)

    top_values, top_indices = scores.topk(top_k)
    return [(doc_ids[i], top_values[idx].item())
            for idx, i in enumerate(top_indices.tolist())]


# ---- Metrics ----
def dcg_at_k(rel_scores, k):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores[:k]))

def ndcg_at_k(retrieved, relevant, k):
    rel_scores = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg  = dcg_at_k(rel_scores, k)
    n_rel = min(len(relevant), k)
    idcg = dcg_at_k([1]*n_rel + [0]*(k-n_rel), k)
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


# ---- Warm-up ----
print("\nWarming up...")
for _ in range(3):
    search_pq_rust(queries[0].text, top_k=100)


# ---- Diagnostic: per-stage timing ----
print("\n" + "=" * 60)
print("Diagnostic: per-stage timing (averaged over 5 runs)")
print("=" * 60)

q_text = queries[0].text
N_DIAG = 5
t_encode = t_lookups = t_rust = t_scatter = t_topk = 0.0

for _ in range(N_DIAG):
    t0 = time.perf_counter()
    Q     = encode_query(q_text)
    num_q = Q.shape[0]
    t1 = time.perf_counter()

    Q_chunks   = Q.reshape(num_q, M, chunk_dim)
    lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, codebooks)
    t2 = time.perf_counter()

    sim_np = colvec_kernel.apply_pq_lookups(lookups_np, codes_transposed)
    t3 = time.perf_counter()

    sim_t = torch.from_numpy(sim_np)
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_t.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc_t.unsqueeze(0).expand(num_q, -1),
        src=sim_t,
        reduce="amax",
        include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    t4 = time.perf_counter()

    scores.topk(100)
    t5 = time.perf_counter()

    t_encode  += t1 - t0
    t_lookups += t2 - t1
    t_rust    += t3 - t2
    t_scatter += t4 - t3
    t_topk    += t5 - t4

avg = lambda x: x / N_DIAG * 1000
print(f"  query encode:           {avg(t_encode):7.1f} ms")
print(f"  build lookups (einsum): {avg(t_lookups):7.1f} ms")
print(f"  apply lookups (Rust):   {avg(t_rust):7.1f} ms   ← was 682ms in Python")
print(f"  scatter_reduce + sum:   {avg(t_scatter):7.1f} ms")
print(f"  topk:                   {avg(t_topk):7.1f} ms")
total = avg(t_encode + t_lookups + t_rust + t_scatter + t_topk)
print(f"  total:                  {total:7.1f} ms")


# ---- Full benchmark ----
print("\n" + "=" * 60)
print(f"Full benchmark — Rust AVX2 PQ kernel (M={M}, 16× compression)")
print("=" * 60)

ndcgs, recalls, latencies = [], [], []
for q in tqdm(queries):
    relevant = qrels_per_query[q.query_id]
    t0 = time.perf_counter()
    results = search_pq_rust(q.text, top_k=100)
    latencies.append((time.perf_counter() - t0) * 1000)
    retrieved = [d for d, _ in results]
    ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
    recalls.append(recall_at_k(retrieved, relevant, 100))

mean_ndcg   = statistics.mean(ndcgs)
mean_recall = statistics.mean(recalls)
median_lat  = statistics.median(latencies)
p99_lat     = sorted(latencies)[int(len(latencies) * 0.99)]

print("\n" + "=" * 60)
print(f"SciFact results — Rust AVX2 PQ (M={M}, 16× compression)")
print("=" * 60)
print(f"  Queries evaluated:  {len(queries)}")
print(f"  nDCG@10:            {mean_ndcg:.4f}")
print(f"  Recall@100:         {mean_recall:.4f}")
print(f"  Median latency:     {median_lat:.1f} ms")
print(f"  p99 latency:        {p99_lat:.1f} ms")
print()
print("  Progression:")
print(f"  Brute force (Day 11):       nDCG=0.6122, lat=108 ms")
print(f"  PQ slow kernel (Day 15):    nDCG=0.6124, lat=4036 ms")
print(f"  PQ torch kernel (Day 16):   nDCG=0.6124, lat=1141 ms")
print(f"  PQ Rust AVX2 (today):       nDCG={mean_ndcg:.4f}, lat={median_lat:.1f} ms")
print()


# ---- Persist ----
record = {
    "timestamp":        datetime.datetime.now().isoformat(),
    "system":           "pq_rust_avx2",
    "M": M, "K": K,    "compression_ratio": 16,
    "dataset":          "scifact",
    "num_queries":      len(queries),
    "ndcg_at_10":       mean_ndcg,
    "recall_at_100":    mean_recall,
    "median_latency_ms": median_lat,
    "p99_latency_ms":   p99_lat,
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/scifact_history.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
print(f"Saved to benchmarks/results/scifact_history.jsonl")