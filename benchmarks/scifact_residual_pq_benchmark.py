"""
SciFact benchmark using residual PQ index.

Residual PQ vs vanilla PQ:
  Vanilla PQ:   compress(raw_vector) → codes
  Residual PQ:  compress(raw_vector - centroid) → codes

At query time, asymmetric scoring:
  For each query chunk m and candidate token n:
    score_contribution = dot(q_chunk_m, centroid_chunk_m[assignment[n]])
                       + lookup(q_chunk_m, residual_codebook_m, codes[n, m])

The centroid term is precomputed once per (query_token, cluster).
The residual term uses the same PQ lookup table mechanism as vanilla PQ.

This combines:
  - Chunked corpus (Day 24): covers full document content
  - Residual PQ (Day 25):    better compression quality

Baseline to beat: nDCG@10 = 0.6691 (chunked, vanilla BF, Day 24)
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
import colvec_kernel


# ---- Load model ----
print("Loading model...")
load_model(device="cpu")


# ---- Load residual PQ index ----
print("Loading residual PQ index...")
with open("data/scifact_residual_pq.pkl", "rb") as f:
    idx = pickle.load(f)

ivf_centroids  = idx["ivf_centroids"].astype(np.float32)   # [K_IVF, D]
posting_lists  = idx["posting_lists"]
pq_codebooks   = idx["pq_codebooks"].astype(np.float32)    # [M, K, chunk_dim]
pq_codes       = idx["pq_codes"]                            # [N_tokens, M] uint8
token_to_doc   = idx["token_to_doc"]                        # [N_tokens] int32
doc_ids        = idx["doc_ids"]
K_IVF          = idx["K_IVF"]
M              = idx["M_PQ"]
K              = idx["K_PQ"]
D              = idx["D"]
N_tokens       = idx["N_tokens"]
NUM_DOCS       = idx["NUM_DOCS"]
chunk_dim      = D // M

print(f"  K_IVF={K_IVF}, M={M}, K={K}, D={D}")
print(f"  {N_tokens} tokens, {NUM_DOCS} docs")

# Pre-convert to torch
ivf_centroids_t = torch.from_numpy(ivf_centroids)
token_to_doc_t  = torch.from_numpy(token_to_doc.astype(np.int64))

# Pre-transpose codes for Rust kernel [M, N_tokens]
print("Pre-transposing codes for Rust kernel...")
pq_codes_T = np.ascontiguousarray(pq_codes.T)


# ---- Load also the chunked corpus for brute force comparison ----
print("Loading chunked corpus for brute force baseline...")
with open("data/scifact_corpus_chunked.pkl", "rb") as f:
    corpus = pickle.load(f)

all_chunk_vecs  = corpus["vectors"]
all_doc_vectors = torch.cat(
    [torch.from_numpy(v) for v in all_chunk_vecs], dim=0
)
bf_token_to_doc = torch.from_numpy(corpus["token_to_doc"]).long()
print(f"  Brute force corpus: {all_doc_vectors.shape[0]} tokens")


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


# ---- Brute force search (chunked corpus, exact) ----
def search_bruteforce(query_text, top_k=100):
    Q     = torch.from_numpy(encode_query(query_text, augment=False))
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


# ---- Residual PQ search ----
def search_residual_pq(query_text, nprobe=None, top_k=100):
    """
    Asymmetric residual PQ scoring.

    For each query token q and candidate token n:
      score[q,n] = sum over m of:
        dot(q_chunk_m, centroid_chunk_m[assignment[n]])   ← centroid term
        + lookup(q_chunk_m, residual_codebook_m, code[n,m]) ← residual term

    The centroid term is handled by precomputing centroid scores per cluster.
    The residual term uses the Rust AVX2 kernel (same as vanilla PQ).

    With nprobe=None: brute force over all tokens (full recall).
    With nprobe=int: IVF candidate filtering first.
    """
    Q_np  = encode_query(query_text, augment=False)   # [num_q, D]
    Q_t   = torch.from_numpy(Q_np)
    num_q = Q_np.shape[0]

    # ── Candidate selection ──
    if nprobe is not None:
        centroid_sim = Q_t @ ivf_centroids_t.T
        _, top_clusters = centroid_sim.topk(min(nprobe, K_IVF), dim=1)
        probed = top_clusters.flatten().unique().tolist()
        candidate_ids = np.unique(
            np.concatenate([posting_lists[c] for c in probed])
        )
    else:
        candidate_ids = np.arange(N_tokens, dtype=np.int32)

    n_cands = len(candidate_ids)
    if n_cands == 0:
        return []

    # ── Build residual PQ lookup tables ──
    # For each query token q and chunk m:
    #   residual_lookup[q, m, k] = dot(q_chunk_m, residual_codebook_m_entry_k)
    Q_chunks            = Q_np.reshape(num_q, M, chunk_dim)
    residual_lookups_np = np.einsum(
        "qmd,mkd->qmk", Q_chunks, pq_codebooks
    ).astype(np.float32)                              # [num_q, M, K]

    # ── Build centroid lookup tables ──
    # For each query token q, chunk m, and IVF centroid c:
    #   centroid_lookup[q, m, c] = dot(q_chunk_m, centroid_c_chunk_m)
    # Then for each candidate token, add centroid_lookup[q, m, assignment[c]]
    #
    # We do this by reshaping centroids into chunks too
    centroids_chunked = ivf_centroids.reshape(K_IVF, M, chunk_dim)  # [K_IVF, M, chunk_dim]
    centroid_lookups  = np.einsum(
        "qmd,cmd->qmc", Q_chunks, centroids_chunked
    ).astype(np.float32)                              # [num_q, M, K_IVF]

    # ── Score candidates ──
    # Total score = centroid_score + residual_score
    #
    # Centroid score: for each candidate token n,
    #   sum over m of centroid_lookups[q, m, assignment[n]]
    # Residual score: for each candidate token n,
    #   sum over m of residual_lookups[q, m, pq_codes[n, m]]
    #   (this is the Rust kernel)

    # Get assignments and codes for candidates
    cand_assignments = idx["ivf_assignments"][candidate_ids]  # [n_cands]
    cand_codes_T     = np.ascontiguousarray(
        pq_codes_T[:, candidate_ids]
    )                                                          # [M, n_cands]

    # Centroid score via lookup (numpy — fast, small table)
    # centroid_lookups[:, :, cand_assignments] → [num_q, M, n_cands]
    centroid_scores_np = centroid_lookups[:, :, cand_assignments]  # [num_q, M, n_cands]
    centroid_scores    = centroid_scores_np.sum(axis=1)            # [num_q, n_cands]

    # Residual score via Rust AVX2 kernel
    residual_scores_np = colvec_kernel.apply_pq_lookups(
        residual_lookups_np, cand_codes_T
    )                                                          # [num_q, n_cands]

    # Total score
    total_scores_np = centroid_scores + residual_scores_np    # [num_q, n_cands]

    # ── MaxSim per doc ──
    total_scores_t = torch.from_numpy(total_scores_np)
    cand_doc_ids   = token_to_doc_t[
        torch.from_numpy(candidate_ids.astype(np.int64))
    ]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=total_scores_t.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=cand_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=total_scores_t, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)

    top_values, top_indices = scores.topk(top_k)
    return [(doc_ids[i], top_values[j].item())
            for j, i in enumerate(top_indices.tolist())]


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


def run_benchmark(name, search_fn, queries, top_k=100):
    ndcgs, recalls, latencies = [], [], []
    for q in tqdm(queries, desc=name, ncols=80):
        relevant = qrels[q.query_id]
        t0       = time.perf_counter()
        results  = search_fn(q.text)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [d for d, _ in results]
        ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        recalls.append(recall_at_k(retrieved, relevant, 100))

    return {
        "name":           name,
        "ndcg_at_10":     statistics.mean(ndcgs),
        "recall_at_100":  statistics.mean(recalls),
        "median_lat_ms":  statistics.median(latencies),
        "p99_lat_ms":     sorted(latencies)[int(len(latencies)*0.99)],
    }


# ---- Warm up ----
print("\nWarming up...")
search_bruteforce(queries[0].text, top_k=10)
search_residual_pq(queries[0].text, top_k=10)


# ---- Run benchmarks ----
print("\n" + "=" * 65)
print("SciFact benchmark: residual PQ vs brute force")
print("=" * 65)

r_bf  = run_benchmark(
    "Brute force (chunked, exact)",
    lambda t: search_bruteforce(t, top_k=100),
    queries,
)

r_rpq = run_benchmark(
    "Residual PQ (brute force candidates)",
    lambda t: search_residual_pq(t, nprobe=None, top_k=100),
    queries,
)

# ---- Report ----
print("\n" + "=" * 65)
print("Results")
print("=" * 65)
print(f"\n{'System':<35} | {'nDCG@10':>8} | {'Recall':>7} | {'Median lat':>11}")
print("-" * 65)

CHUNKED_BASELINE_NDCG   = 0.6691
CHUNKED_BASELINE_RECALL = 0.8910

for r in [r_bf, r_rpq]:
    print(f"{r['name']:<35} | {r['ndcg_at_10']:>8.4f} | "
          f"{r['recall_at_100']:>7.4f} | {r['median_lat_ms']:>10.1f}ms")

print("-" * 65)
print(f"\nReference baselines:")
print(f"  Chunked BF (Day 24):          nDCG = {CHUNKED_BASELINE_NDCG:.4f}")
print(f"  ColBERTv2 published:          nDCG = 0.6930")
print()

delta_ndcg   = r_rpq["ndcg_at_10"]    - r_bf["ndcg_at_10"]
delta_recall = r_rpq["recall_at_100"] - r_bf["recall_at_100"]
print(f"Residual PQ vs brute force:")
print(f"  nDCG delta:    {delta_ndcg:+.4f}")
print(f"  Recall delta:  {delta_recall:+.4f}")
print()

gap_to_colbert = 0.6930 - r_bf["ndcg_at_10"]
print(f"Gap to ColBERTv2 published: {gap_to_colbert:+.4f}")
print()

if abs(delta_ndcg) < 0.005:
    print("~ Residual PQ is neutral vs brute force on this corpus.")
    print("  Compression quality is good — codebooks represent residuals well.")
elif delta_ndcg > 0.005:
    print("✓ Residual PQ IMPROVES over brute force — unexpected but check for bugs.")
else:
    print(f"~ Residual PQ costs {abs(delta_ndcg):.3f} nDCG vs exact scoring.")
    print(f"  This is the compression-quality tradeoff.")
    if abs(delta_ndcg) < 0.02:
        print("  Small gap — residual PQ is working well.")
    else:
        print("  Large gap — consider more M subquantizers (M=48 or M=64).")


# ---- Persist ----
results = [r_bf, r_rpq]
record  = {
    "experiment":  "residual_pq_vs_brute_force",
    "dataset":     "scifact_chunked",
    "K_IVF":       K_IVF,
    "M_PQ":        M,
    "K_PQ":        K,
    "results":     results,
}
import datetime
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/residual_pq_results.json", "w") as f:
    import json
    record["timestamp"] = datetime.datetime.now().isoformat()
    json.dump(record, f, indent=2)

for r in results:
    r["timestamp"] = datetime.datetime.now().isoformat()
    with open("benchmarks/results/scifact_history.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")

print(f"\nSaved to benchmarks/results/residual_pq_results.json")