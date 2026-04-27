"""
Head-to-head comparison: your IVF+PQ vs Qdrant HNSW on SciFact.

Both systems use sentence-transformers/all-MiniLM-L6-v2 (384-dim).
Both systems measure FULL pipeline latency: encode query + search.
This is the only fair comparison — if encoding is outside the timer
for one system but not the other, you're measuring different things.

Run after:
  1. benchmarks/qdrant_encode.py         (creates data/scifact_singlevec.pkl)
  2. benchmarks/qdrant_ingest.py         (ingests corpus into Qdrant)
  3. benchmarks/qdrant_your_system.py    (builds your IVF+PQ on same vectors)
"""
import json
import pickle
import statistics
import time
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import ir_datasets
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Connect to Qdrant
# ============================================================
COLLECTION = "scifact_v1"
QDRANT_URL = "http://localhost:6333"

print(f"Connecting to Qdrant at {QDRANT_URL}...")
qdrant = QdrantClient(url=QDRANT_URL)
info   = qdrant.get_collection(COLLECTION)
print(f"  Collection '{COLLECTION}': {info.points_count} points, "
      f"status={info.status}")


# ============================================================
# 2. Load encoder (shared by both systems)
# ============================================================
print("\nLoading MiniLM encoder...")
with open("data/scifact_singlevec.pkl", "rb") as f:
    vec_data = pickle.load(f)

doc_ids_list = vec_data["doc_ids"]
model_name   = vec_data["model"]
D            = vec_data["dim"]
N            = len(doc_ids_list)

model = SentenceTransformer(model_name)
print(f"  Model: {model_name}, dim={D}, corpus={N} docs")


# ============================================================
# 3. Load your IVF+PQ index
# ============================================================
print("\nLoading your IVF+PQ index...")
with open("data/scifact_singlevec_ivfpq.pkl", "rb") as f:
    idx = pickle.load(f)

ivf_centroids_t = torch.from_numpy(
    idx["ivf_centroids"].astype(np.float32)
)                                                     # [K_IVF, D]
posting_lists   = idx["posting_lists"]                # list of arrays
pq_codebooks    = idx["pq_codebooks"].astype(np.float32)  # [M, K, chunk_dim]
pq_codes        = idx["pq_codes"]                     # [N, M] uint8
K_IVF           = idx["K_IVF"]
M               = idx["M_PQ"]
K               = idx["K_PQ"]
chunk_dim       = D // M

# Pre-convert codes to torch once — avoids per-query conversion cost
codes_t_per_chunk = [
    torch.from_numpy(pq_codes[:, m].astype(np.int64))
    for m in range(M)
]

print(f"  K_IVF={K_IVF}, M={M}, K={K}, chunk_dim={chunk_dim}")


# ============================================================
# 4. Load SciFact qrels
# ============================================================
print("\nLoading SciFact qrels...")
dataset = ir_datasets.load("beir/scifact/test")
queries_all      = list(dataset.queries_iter())
qrels_per_query  = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels_per_query[qrel.query_id].add(qrel.doc_id)
queries = [q for q in queries_all if q.query_id in qrels_per_query]
print(f"  {len(queries)} queries with relevance judgments")


# ============================================================
# 5. Metrics
# ============================================================
def dcg_at_k(rel_scores, k):
    return sum(rel / np.log2(i + 2)
               for i, rel in enumerate(rel_scores[:k]))

def ndcg_at_k(retrieved, relevant, k):
    rel_scores = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg  = dcg_at_k(rel_scores, k)
    n_rel = min(len(relevant), k)
    idcg = dcg_at_k([1] * n_rel + [0] * (k - n_rel), k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


# ============================================================
# 6. Search functions — encoding INSIDE the timer for both
# ============================================================

def search_qdrant(query_text, ef=128, top_k=100):
    """
    Full pipeline: MiniLM encode → HNSW search in Qdrant.
    ef controls accuracy/speed tradeoff (higher = more accurate, slower).
    """
    q_vec = model.encode(query_text, normalize_embeddings=True)
    try:
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=q_vec.tolist(),
            limit=top_k,
            search_params={"hnsw_ef": ef},
        ).points
    except Exception:
        # Fallback for older qdrant-client versions
        results = qdrant.search(
            collection_name=COLLECTION,
            query_vector=q_vec.tolist(),
            limit=top_k,
            search_params={"hnsw_ef": ef},
        )
    return [r.payload["scifact_id"] for r in results]


def search_your_system(query_text, nprobe=8, top_k=100):
    """
    Full pipeline: MiniLM encode → IVF candidate lookup → PQ scoring.
    nprobe controls accuracy/speed tradeoff (higher = more accurate, slower).
    """
    # Encode — same model as Qdrant, measured inside the timer
    q_vec = model.encode(query_text, normalize_embeddings=True)
    q_np  = q_vec[np.newaxis, :].astype(np.float32)     # [1, D]
    q_t   = torch.from_numpy(q_np)                       # [1, D]

    # IVF: find candidate doc IDs
    centroid_sim = q_t @ ivf_centroids_t.T               # [1, K_IVF]
    _, top_clusters = centroid_sim.topk(
        min(nprobe, K_IVF), dim=1
    )
    probed = top_clusters.flatten().unique().tolist()
    candidate_ids = np.unique(
        np.concatenate([posting_lists[c] for c in probed])
    )
    n_cands = len(candidate_ids)
    if n_cands == 0:
        return []

    # PQ: build lookup table and score only candidate docs
    Q_chunks   = q_np.reshape(1, M, chunk_dim)
    lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, pq_codebooks)
    lookups_t  = torch.from_numpy(lookups_np)            # [1, M, K]

    cand_codes   = pq_codes[candidate_ids]               # [n_cands, M]
    cand_codes_t = [
        torch.from_numpy(cand_codes[:, m].astype(np.int64))
        for m in range(M)
    ]

    sim = torch.zeros((1, n_cands), dtype=torch.float32)
    for m in range(M):
        sim += lookups_t[:, m, cand_codes_t[m]]

    scores = sim.squeeze(0)
    top_k_actual = min(top_k, n_cands)
    _, top_idx   = scores.topk(top_k_actual)
    return [doc_ids_list[candidate_ids[i]] for i in top_idx.tolist()]


# ============================================================
# 7. Benchmark runner
# ============================================================

def run_benchmark(name, search_fn, queries, top_k=100):
    """
    Run all queries through search_fn. search_fn takes a query text string.
    Returns dict of quality and latency metrics.
    """
    ndcgs, recalls, latencies = [], [], []

    for q in tqdm(queries, desc=name, ncols=80):
        relevant = qrels_per_query[q.query_id]

        t0 = time.perf_counter()
        retrieved = search_fn(q.text)
        latencies.append((time.perf_counter() - t0) * 1000)

        ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        recalls.append(recall_at_k(retrieved, relevant, 100))

    lat_sorted = sorted(latencies)
    return {
        "name":          name,
        "ndcg_at_10":    statistics.mean(ndcgs),
        "recall_at_100": statistics.mean(recalls),
        "median_lat_ms": statistics.median(latencies),
        "p99_lat_ms":    lat_sorted[int(len(lat_sorted) * 0.99)],
        "p50_lat_ms":    lat_sorted[int(len(lat_sorted) * 0.50)],
    }


# ============================================================
# 8. Warm-up (lazy init, model warm-up, cache warm-up)
# ============================================================
print("\nWarming up (3 queries each)...")
for _ in range(3):
    search_qdrant(queries[0].text, ef=128)
    search_your_system(queries[0].text, nprobe=8)
print("  Done.")


# ============================================================
# 9. Run all configurations
# ============================================================
print("\n" + "=" * 70)
print("Running benchmarks — full pipeline latency (encode + search)")
print("=" * 70)

results = []

# Your system at four nprobe values
for nprobe in [4, 8, 16, 32]:
    r = run_benchmark(
        name      = f"Your IVF+PQ (nprobe={nprobe})",
        search_fn = lambda qt, np=nprobe: search_your_system(qt, nprobe=np),
        queries   = queries,
    )
    r["system"]  = "yours"
    r["setting"] = f"nprobe={nprobe}"
    results.append(r)

# Qdrant at four ef values
for ef in [32, 64, 128, 256]:
    r = run_benchmark(
        name      = f"Qdrant HNSW (ef={ef})",
        search_fn = lambda qt, e=ef: search_qdrant(qt, ef=e),
        queries   = queries,
    )
    r["system"]  = "qdrant"
    r["setting"] = f"ef={ef}"
    results.append(r)


# ============================================================
# 10. Print the comparison table
# ============================================================
print("\n" + "=" * 82)
print("Final comparison — full pipeline latency (encode + search)")
print("=" * 82)
print(f"{'System':<32} | {'nDCG@10':>8} | {'Recall':>7} | {'Median':>8} | {'p99':>8}")
print("-" * 82)
for r in results:
    print(f"{r['name']:<32} | {r['ndcg_at_10']:>8.4f} | "
          f"{r['recall_at_100']:>7.4f} | "
          f"{r['median_lat_ms']:>7.1f}ms | "
          f"{r['p99_lat_ms']:>7.1f}ms")
print("-" * 82)
print()
print("Context:")
print(f"  Both systems: {model_name} ({D}-dim)")
print(f"  Qdrant:       HNSW m=16, ef_construct=100")
print(f"  Yours:        IVF+PQ K_IVF={K_IVF}, M={M}, K={K}")
print(f"  Dataset:      SciFact {N} docs, {len(queries)} dev queries")
print()
print("Timing note: encode + search measured together for both systems.")
print("  Encoding (~15-30ms on CPU) is the dominant cost at this scale.")


# ============================================================
# 11. Matched-recall comparison (the most honest number)
# ============================================================
print("\n" + "-" * 60)
print("Matched-recall comparison")
print("(latency at the same quality level — the fairest comparison)")
print("-" * 60)

your_results   = [r for r in results if r["system"] == "yours"]
qdrant_results = [r for r in results if r["system"] == "qdrant"]

printed_any = False
for yr in your_results:
    closest = min(
        qdrant_results,
        key=lambda qr: abs(qr["recall_at_100"] - yr["recall_at_100"])
    )
    delta = abs(yr["recall_at_100"] - closest["recall_at_100"])
    if delta < 0.05:
        printed_any = True
        ratio  = closest["median_lat_ms"] / yr["median_lat_ms"]
        winner = "Yours is faster" if ratio > 1 else "Qdrant is faster"
        print(f"\n  At ~{yr['recall_at_100']:.0%} recall:")
        print(f"    Yours  ({yr['setting']:<12}):  {yr['median_lat_ms']:>6.1f} ms median")
        print(f"    Qdrant ({closest['setting']:<12}):  {closest['median_lat_ms']:>6.1f} ms median")
        print(f"    → {winner}: {max(ratio, 1/ratio):.2f}×")

if not printed_any:
    print("\n  No matched pairs found within 5% recall delta.")
    print("  The two systems operate at different recall ranges.")
    print("  Try extending nprobe to {64, 128} to find overlap.")


# ============================================================
# 12. Plain-language summary
# ============================================================
print("\n" + "=" * 60)
print("Plain-language summary")
print("=" * 60)

best_yours  = max(your_results,   key=lambda r: r["recall_at_100"])
best_qdrant = max(qdrant_results, key=lambda r: r["recall_at_100"])

print(f"\n  Best quality your system: "
      f"nDCG={best_yours['ndcg_at_10']:.4f}, "
      f"Recall={best_yours['recall_at_100']:.4f} "
      f"({best_yours['setting']}, {best_yours['median_lat_ms']:.1f}ms)")
print(f"  Best quality Qdrant:      "
      f"nDCG={best_qdrant['ndcg_at_10']:.4f}, "
      f"Recall={best_qdrant['recall_at_100']:.4f} "
      f"({best_qdrant['setting']}, {best_qdrant['median_lat_ms']:.1f}ms)")

ndcg_gap   = best_qdrant["ndcg_at_10"]    - best_yours["ndcg_at_10"]
recall_gap = best_qdrant["recall_at_100"] - best_yours["recall_at_100"]
print(f"\n  Quality gap (Qdrant - yours): "
      f"nDCG={ndcg_gap:+.4f}, Recall={recall_gap:+.4f}")
print(f"  Algorithm gap: HNSW vs IVF at {N} docs")
print(f"  Implementation gap: Rust+SIMD vs Python+PyTorch")
print(f"  (both visible at this scale; algorithm gap dominates at 100K+ docs)")


# ============================================================
# 13. Persist
# ============================================================
record = {
    "timestamp":  datetime.datetime.now().isoformat(),
    "experiment": "qdrant_comparison_corrected",
    "dataset":    "scifact",
    "model":      model_name,
    "note":       "full pipeline latency: encode + search for both systems",
    "results":    results,
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/qdrant_comparison.json", "w") as f:
    json.dump(record, f, indent=2)
print(f"\nSaved to benchmarks/results/qdrant_comparison.json")