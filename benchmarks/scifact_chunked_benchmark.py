"""
SciFact benchmark using chunked document encoding.

Compares quality against the non-chunked baseline (Day 11: nDCG=0.6122).

Scoring with chunks:
  doc_score = sum over query tokens of
              max over ALL tokens in ALL chunks of that doc
  (scatter_reduce handles this naturally — token_to_doc maps every
  token across all chunks to its parent doc, so the per-doc max
  automatically takes the best match anywhere in the document.)

Expected result on SciFact: neutral (most docs fit in one chunk).
Expected result on NFCorpus/TREC-COVID: meaningful recall improvement.

Usage:
    uv run python benchmarks/scifact_chunked_benchmark.py
"""
import json
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import ir_datasets
from tqdm import tqdm

from colvec.encoder import load_model, encode_query

# ---- Load model ----
print("Loading model...")
load_model(device="cpu")

# ---- Load chunked corpus ----
print("Loading chunked SciFact corpus...")
with open("data/scifact_corpus_chunked.pkl", "rb") as f:
    corpus = pickle.load(f)

doc_ids       = corpus["doc_ids"]
all_chunk_vecs = corpus["vectors"]
chunk_counts  = corpus["chunk_counts"]
token_to_doc  = torch.from_numpy(corpus["token_to_doc"]).long()
NUM_DOCS      = len(doc_ids)
total_chunks  = len(all_chunk_vecs)

# Stack all chunk vectors into one tensor
all_doc_vectors = torch.cat(
    [torch.from_numpy(v) for v in all_chunk_vecs], dim=0
)
total_tokens = all_doc_vectors.shape[0]

print(f"  {NUM_DOCS} docs, {total_chunks} chunks, {total_tokens:,} tokens")
print(f"  Chunks/doc: median={statistics.median(chunk_counts):.1f}, "
      f"max={max(chunk_counts)}")
print(f"  Docs with >1 chunk: {sum(1 for c in chunk_counts if c > 1)}")


# ---- Load qrels ----
print("\nLoading SciFact qrels...")
dataset = ir_datasets.load("beir/scifact/test")
queries = list(dataset.queries_iter())
qrels   = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels[qrel.query_id].add(qrel.doc_id)
queries = [q for q in queries if q.query_id in qrels]
print(f"  {len(queries)} queries with relevance judgments")


# ---- Search ----
def search(query_text, top_k=100, augment=False):
    """
    Brute-force MaxSim search over chunked corpus.

    token_to_doc maps every token (across all chunks) to its parent doc.
    scatter_reduce naturally takes the max over all chunks per doc.
    """
    Q     = torch.from_numpy(encode_query(query_text, augment=augment))
    sim   = Q @ all_doc_vectors.T    # [num_q, N_tokens]
    num_q = sim.shape[0]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"))
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc.unsqueeze(0).expand(num_q, -1),
        src=sim, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    return [(doc_ids[i], v.item())
            for i, v in zip(top_indices.tolist(), top_values)]


# ---- Metrics ----
def dcg_at_k(rel, k):
    return sum(r / np.log2(i+2) for i, r in enumerate(rel[:k]))

def ndcg_at_k(retrieved, relevant, k):
    rel  = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg  = dcg_at_k(rel, k)
    n    = min(len(relevant), k)
    idcg = dcg_at_k([1]*n + [0]*(k-n), k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


# ---- Warm up ----
print("\nWarming up...")
search(queries[0].text, top_k=10)


# ---- Run benchmark ----
print("\nRunning chunked benchmark (300 queries)...")
ndcgs, recalls, latencies = [], [], []

for q in tqdm(queries, ncols=80):
    relevant = qrels[q.query_id]
    t0       = time.perf_counter()
    results  = search(q.text, top_k=100, augment=False)
    latencies.append((time.perf_counter() - t0) * 1000)
    retrieved = [d for d, _ in results]
    ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
    recalls.append(recall_at_k(retrieved, relevant, 100))

mean_ndcg   = statistics.mean(ndcgs)
mean_recall = statistics.mean(recalls)
median_lat  = statistics.median(latencies)
p99_lat     = sorted(latencies)[int(len(latencies) * 0.99)]

# ---- Report ----
BASELINE_NDCG   = 0.6122
BASELINE_RECALL = 0.8666
BASELINE_LAT    = 108.4

print("\n" + "=" * 60)
print("SciFact results — chunked document encoding")
print("=" * 60)
print(f"  Queries:         {len(queries)}")
print(f"  nDCG@10:         {mean_ndcg:.4f}")
print(f"  Recall@100:      {mean_recall:.4f}")
print(f"  Median latency:  {median_lat:.1f} ms")
print(f"  p99 latency:     {p99_lat:.1f} ms")
print()
print(f"  Baseline (Day 11, no chunking):")
print(f"    nDCG@10:    {BASELINE_NDCG:.4f}")
print(f"    Recall@100: {BASELINE_RECALL:.4f}")
print(f"    Latency:    {BASELINE_LAT:.1f} ms")
print()
print(f"  Delta nDCG@10:    {mean_ndcg - BASELINE_NDCG:+.4f}")
print(f"  Delta Recall@100: {mean_recall - BASELINE_RECALL:+.4f}")
print(f"  Delta latency:    {median_lat - BASELINE_LAT:+.1f} ms "
      f"({total_tokens/all_doc_vectors.shape[0]*100:.0f}% more tokens)")
print()

# Verdict
delta = mean_ndcg - BASELINE_NDCG
if delta > 0.005:
    print("✓ Chunking improves quality on SciFact (unexpected but good).")
elif delta < -0.005:
    print("✗ Chunking hurts quality — check token_to_doc mapping.")
else:
    print("~ Neutral on SciFact (expected — most docs fit in one chunk).")
    print("  Real benefit will show on longer medical documents.")
    print(f"  Docs with >1 chunk: "
          f"{sum(1 for c in chunk_counts if c > 1)}/{NUM_DOCS} "
          f"({100*sum(1 for c in chunk_counts if c > 1)/NUM_DOCS:.1f}%)")

# ---- Persist ----
record = {
    "system":           "brute_force_chunked",
    "dataset":          "scifact",
    "chunk_size":       corpus["chunk_size"],
    "chunk_overlap":    corpus["chunk_overlap"],
    "num_queries":      len(queries),
    "ndcg_at_10":       mean_ndcg,
    "recall_at_100":    mean_recall,
    "median_latency_ms": median_lat,
    "total_chunks":     total_chunks,
    "multi_chunk_docs": sum(1 for c in chunk_counts if c > 1),
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/scifact_history.jsonl", "a") as f:
    import datetime
    record["timestamp"] = datetime.datetime.now().isoformat()
    f.write(json.dumps(record) + "\n")
print(f"\nSaved to benchmarks/results/scifact_history.jsonl")