"""
SciFact benchmark: IVF + Residual PQ + Rust AVX2.

Full production pipeline on the chunked SciFact corpus:
  1. IVF candidate filtering  (K_IVF=1024, skip ~95% of corpus)
  2. Residual PQ compression  (11.6x storage reduction)
  3. Fast centroid scoring    (per probed cluster, not per token)
  4. Rust AVX2 residual scoring

Key fix vs previous version:
  Centroid scoring is now O(nprobe) not O(n_candidates).
  We score query chunks against the nprobe centroids ONCE,
  then broadcast each token's score from its cluster's centroid score.
  This drops centroid scoring from 394ms → ~1ms.

Scoring formula per (query_token q, candidate_token n in cluster c):
  score[q,n] = dot(q, centroid_c)    ← centroid term (per cluster)
             + lookup(q_chunk, residual_codebook, code[n]) ← residual term

Baselines:
  Brute force chunked (Day 24):    nDCG = 0.6691, lat = 134ms
  Residual PQ full scan (Day 25):  nDCG = 0.6655, lat = 2818ms
"""
import json
import pickle
import statistics
import time
import datetime
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import ir_datasets
from tqdm import tqdm

from colvec.encoder import load_model, encode_query
import colvec_kernel

# ---- Load model ----
print("Loading model...")
load_model(device="cpu")

# ---- Load residual PQ index ----
print("Loading residual PQ index...")
with open("data/scifact_residual_pq.pkl", "rb") as f:
    idx = pickle.load(f)

ivf_centroids   = idx["ivf_centroids"].astype(np.float32)
posting_lists   = idx["posting_lists"]
pq_codebooks    = idx["pq_codebooks"].astype(np.float32)
pq_codes        = idx["pq_codes"]
ivf_assignments = idx["ivf_assignments"]
token_to_doc    = idx["token_to_doc"]
doc_ids         = idx["doc_ids"]
K_IVF           = idx["K_IVF"]
M               = idx["M_PQ"]
K               = idx["K_PQ"]
D               = idx["D"]
N_tokens        = idx["N_tokens"]
NUM_DOCS        = idx["NUM_DOCS"]
chunk_dim       = D // M

print(f"  K_IVF={K_IVF}, M={M}, K={K}, D={D}")
print(f"  {N_tokens:,} tokens, {NUM_DOCS} docs")

# Pre-convert structures
ivf_centroids_t     = torch.from_numpy(ivf_centroids)
token_to_doc_t      = torch.from_numpy(token_to_doc.astype(np.int64))
pq_codes_T          = np.ascontiguousarray(pq_codes.T)           # [M, N_tokens]
centroids_chunked   = ivf_centroids.reshape(K_IVF, M, chunk_dim) # [K_IVF, M, chunk_dim]

# ---- Load chunked corpus for brute force baseline ----
print("Loading chunked corpus for brute force baseline...")
with open("data/scifact_corpus_chunked.pkl", "rb") as f:
    corpus = pickle.load(f)

all_doc_vectors = torch.cat(
    [torch.from_numpy(v) for v in corpus["vectors"]], dim=0
)
bf_token_to_doc = torch.from_numpy(corpus["token_to_doc"]).long()
print(f"  {all_doc_vectors.shape[0]:,} tokens")

# ---- Load qrels ----
print("\nLoading SciFact qrels...")
dataset = ir_datasets.load("beir/scifact/test")
queries = list(dataset.queries_iter())
qrels   = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels[qrel.query_id].add(qrel.doc_id)
queries = [q for q in queries if q.query_id in qrels]
print(f"  {len(queries)} queries")


# ---- Brute force (exact, chunked) ----
def search_bruteforce(query_text, top_k=100):
    Q     = torch.from_numpy(encode_query(query_text))
    sim   = Q @ all_doc_vectors.T
    num_q = sim.shape[0]
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"))
    per_doc_max.scatter_reduce_(
        dim=1,
        index=bf_token_to_doc.unsqueeze(0).expand(num_q, -1),
        src=sim, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    return [(doc_ids[i], v.item())
            for i, v in zip(top_indices.tolist(), top_values)]


# ---- IVF + Residual PQ + Rust ----
def search_ivf_residual_pq_rust(query_text, nprobe=8, top_k=100):
    """
    Fast residual PQ search with per-cluster centroid scoring.

    Centroid scoring: O(nprobe) einsum instead of O(n_candidates).
    Score the nprobe cluster centroids once, broadcast to their tokens.
    """
    Q_np  = encode_query(query_text)
    Q_t   = torch.from_numpy(Q_np)
    num_q = Q_np.shape[0]

    # ── Step 1: IVF candidate filtering ──
    centroid_sim = Q_t @ ivf_centroids_t.T
    _, top_clusters = centroid_sim.topk(min(nprobe, K_IVF), dim=1)
    probed = top_clusters.flatten().unique().tolist()

    candidate_ids = np.unique(
        np.concatenate([posting_lists[c] for c in probed])
    )
    n_cands = len(candidate_ids)
    if n_cands == 0:
        return [], 0

    # ── Step 2: Build lookup tables ──
    Q_chunks = Q_np.reshape(num_q, M, chunk_dim)

    # Residual lookup table: [num_q, M, K]
    residual_lookups = np.einsum(
        "qmd,mkd->qmk", Q_chunks, pq_codebooks
    ).astype(np.float32)

    # ── Step 3: Centroid scoring — O(nprobe), NOT O(n_candidates) ──
    # Score query chunks against the nprobe probed centroids only.
    probed_arr   = np.array(probed, dtype=np.int32)
    probed_cents = centroids_chunked[probed_arr]     # [n_probed, M, chunk_dim]

    # cluster_scores[q, p] = sum over m of dot(q_chunk_m, centroid_p_chunk_m)
    cluster_scores = np.einsum(
        "qmd,pmd->qp", Q_chunks, probed_cents
    ).astype(np.float32)                             # [num_q, n_probed]

    # Map each candidate token to its probed cluster index
    probed_to_idx    = {c: i for i, c in enumerate(probed)}
    cand_cluster_idx = np.array(
        [probed_to_idx[ivf_assignments[n]] for n in candidate_ids],
        dtype=np.int32,
    )                                                # [n_cands]

    # Broadcast cluster score to each token in that cluster
    centroid_scores = cluster_scores[:, cand_cluster_idx]  # [num_q, n_cands]

    # ── Step 4: Residual scoring via Rust AVX2 kernel ──
    cand_codes_T    = np.ascontiguousarray(pq_codes_T[:, candidate_ids])
    residual_scores = colvec_kernel.apply_pq_lookups(
        residual_lookups, cand_codes_T
    )                                                # [num_q, n_cands]

    # ── Step 5: Total score = centroid + residual ──
    total_scores = torch.from_numpy(
        centroid_scores + residual_scores
    )

    # ── Step 6: MaxSim per doc ──
    cand_doc_ids = token_to_doc_t[
        torch.from_numpy(candidate_ids.astype(np.int64))
    ]
    per_doc_max = torch.full(
        (num_q, NUM_DOCS), float("-inf"), dtype=total_scores.dtype
    )
    per_doc_max.scatter_reduce_(
        dim=1,
        index=cand_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=total_scores, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)

    top_values, top_indices = scores.topk(top_k)
    return [(doc_ids[i], top_values[j].item())
            for j, i in enumerate(top_indices.tolist())], n_cands


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
for _ in range(3):
    search_bruteforce(queries[0].text, top_k=10)
    search_ivf_residual_pq_rust(queries[0].text, nprobe=8, top_k=10)


# ---- Diagnostic: per-stage timing ----
print("\n" + "=" * 65)
print("Diagnostic: per-stage timing (nprobe=8, 5 runs)")
print("=" * 65)

q_text = queries[0].text
N_DIAG = 5
t_encode = t_ivf = t_residual_lut = t_centroid = t_rust = t_scatter = 0.0
diag_cands = 0

for _ in range(N_DIAG):
    t0    = time.perf_counter()
    Q_np  = encode_query(q_text)
    Q_t   = torch.from_numpy(Q_np)
    num_q = Q_np.shape[0]
    t1    = time.perf_counter()

    centroid_sim = Q_t @ ivf_centroids_t.T
    _, top_clusters = centroid_sim.topk(8, dim=1)
    probed        = top_clusters.flatten().unique().tolist()
    candidate_ids = np.unique(np.concatenate([posting_lists[c] for c in probed]))
    diag_cands    = len(candidate_ids)
    t2            = time.perf_counter()

    Q_chunks         = Q_np.reshape(num_q, M, chunk_dim)
    residual_lookups = np.einsum("qmd,mkd->qmk", Q_chunks, pq_codebooks).astype(np.float32)
    t3               = time.perf_counter()

    probed_arr       = np.array(probed, dtype=np.int32)
    probed_cents     = centroids_chunked[probed_arr]
    cluster_scores   = np.einsum("qmd,pmd->qp", Q_chunks, probed_cents).astype(np.float32)
    probed_to_idx    = {c: i for i, c in enumerate(probed)}
    cand_cluster_idx = np.array([probed_to_idx[ivf_assignments[n]] for n in candidate_ids], dtype=np.int32)
    centroid_scores  = cluster_scores[:, cand_cluster_idx]
    t4               = time.perf_counter()

    cand_codes_T    = np.ascontiguousarray(pq_codes_T[:, candidate_ids])
    residual_scores = colvec_kernel.apply_pq_lookups(residual_lookups, cand_codes_T)
    t5              = time.perf_counter()

    total_scores = torch.from_numpy(centroid_scores + residual_scores)
    cand_doc_ids = token_to_doc_t[torch.from_numpy(candidate_ids.astype(np.int64))]
    per_doc_max  = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=total_scores.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=cand_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=total_scores, reduce="amax", include_self=True,
    )
    per_doc_max.sum(dim=0).topk(10)
    t6 = time.perf_counter()

    t_encode       += t1 - t0
    t_ivf          += t2 - t1
    t_residual_lut += t3 - t2
    t_centroid     += t4 - t3
    t_rust         += t5 - t4
    t_scatter      += t6 - t5

avg = lambda x: x / N_DIAG * 1000
print(f"  query encode:           {avg(t_encode):7.1f} ms")
print(f"  IVF lookup:             {avg(t_ivf):7.1f} ms  ({diag_cands:,} candidates)")
print(f"  build residual lookups: {avg(t_residual_lut):7.1f} ms")
print(f"  centroid scoring:       {avg(t_centroid):7.1f} ms  (O(nprobe={len(probed)}))")
print(f"  Rust AVX2 residual:     {avg(t_rust):7.1f} ms")
print(f"  scatter_reduce:         {avg(t_scatter):7.1f} ms")
total_diag = avg(t_encode + t_ivf + t_residual_lut + t_centroid + t_rust + t_scatter)
print(f"  total:                  {total_diag:7.1f} ms")


# ---- nprobe sweep ----
print("\n" + "=" * 75)
print(f"IVF+Residual PQ+Rust sweep — SciFact chunked (K_IVF={K_IVF})")
print("=" * 75)

nprobe_values = [1, 2, 4, 8, 16, 32, 64]
sweep_results = []

for nprobe in nprobe_values:
    ndcgs, recalls, latencies, cand_counts = [], [], [], []
    for q in tqdm(queries, desc=f"nprobe={nprobe:>3}", ncols=80, leave=False):
        relevant = qrels[q.query_id]
        t0       = time.perf_counter()
        results, n_cands = search_ivf_residual_pq_rust(q.text, nprobe=nprobe, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)
        cand_counts.append(n_cands)
        retrieved = [d for d, _ in results]
        ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        recalls.append(recall_at_k(retrieved, relevant, 100))

    mean_ndcg  = statistics.mean(ndcgs)
    mean_recall = statistics.mean(recalls)
    median_lat  = statistics.median(latencies)
    mean_cands  = statistics.mean(cand_counts)
    pct_corpus  = 100 * mean_cands / N_tokens

    sweep_results.append({
        "nprobe":        nprobe,
        "ndcg_at_10":   mean_ndcg,
        "recall_at_100": mean_recall,
        "latency_ms":   median_lat,
        "candidates":   mean_cands,
        "pct_corpus":   pct_corpus,
    })
    print(f"  nprobe={nprobe:>3} | nDCG={mean_ndcg:.4f} | "
          f"Recall={mean_recall:.4f} | lat={median_lat:>7.1f}ms | "
          f"cands={int(mean_cands):>7} ({pct_corpus:.1f}%)")


# ---- Brute force baseline ----
print(f"\nRunning brute force baseline (300 queries)...")
ndcgs_bf, recalls_bf, latencies_bf = [], [], []
for q in tqdm(queries, desc="brute force", ncols=80, leave=False):
    relevant = qrels[q.query_id]
    t0       = time.perf_counter()
    results  = search_bruteforce(q.text, top_k=100)
    latencies_bf.append((time.perf_counter() - t0) * 1000)
    retrieved = [d for d, _ in results]
    ndcgs_bf.append(ndcg_at_k(retrieved, relevant, 10))
    recalls_bf.append(recall_at_k(retrieved, relevant, 100))

bf_ndcg  = statistics.mean(ndcgs_bf)
bf_recall = statistics.mean(recalls_bf)
bf_lat   = statistics.median(latencies_bf)


# ---- Final report ----
print("\n" + "=" * 80)
print(f"Final summary — IVF+Residual PQ+Rust on SciFact chunked (K_IVF={K_IVF})")
print("=" * 80)
print(f"\n{'nprobe':>7} | {'nDCG@10':>8} | {'Recall':>7} | "
      f"{'lat ms':>8} | {'vs BF':>6} | {'cands':>8} | {'%corpus':>8}")
print("-" * 80)
for r in sweep_results:
    speedup = bf_lat / r["latency_ms"]
    marker  = " ←" if speedup >= 1.0 else ""
    print(f"{r['nprobe']:>7} | {r['ndcg_at_10']:>8.4f} | "
          f"{r['recall_at_100']:>7.4f} | {r['latency_ms']:>7.1f}ms | "
          f"{speedup:>5.2f}x | {int(r['candidates']):>8} | "
          f"{r['pct_corpus']:>7.1f}%{marker}")
print("-" * 80)
print(f"{'BF':>7} | {bf_ndcg:>8.4f} | {bf_recall:>7.4f} | "
      f"{bf_lat:>7.1f}ms | {'1.00x':>6} | {N_tokens:>8} | {'100.0%':>8}")

print(f"\nReference:")
print(f"  Brute force chunked (Day 24):    nDCG = 0.6691, lat = 134ms")
print(f"  Residual PQ full scan (Day 25):  nDCG = 0.6655, lat = 2818ms")
print(f"  ColBERTv2 published:             nDCG = 0.6930")

best = max(sweep_results, key=lambda r: r["ndcg_at_10"])
print(f"\nBest result:")
print(f"  nprobe={best['nprobe']}: nDCG={best['ndcg_at_10']:.4f}, "
      f"lat={best['latency_ms']:.1f}ms")
gap = 0.6930 - bf_ndcg
print(f"  Gap to ColBERTv2: {gap:.4f} (BF is {bf_ndcg:.4f})")


# ---- Persist ----
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system":    "ivf_residual_pq_rust",
    "dataset":   "scifact_chunked",
    "K_IVF":     K_IVF,
    "M_PQ":      M,
    "bf_ndcg":   bf_ndcg,
    "bf_lat_ms": bf_lat,
    "sweep":     sweep_results,
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/scifact_ivf_residual_pq_rust.json", "w") as f:
    json.dump(record, f, indent=2)
print(f"\nSaved to benchmarks/results/scifact_ivf_residual_pq_rust.json")