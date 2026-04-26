"""
SciFact benchmark: IVF + PQ combined.

Architecture:
  1. Encode query (uncompressed)
  2. IVF: for each query token, find nprobe nearest centroids
     → union of posting lists = candidate token IDs
  3. PQ: build lookup tables [num_q, M, K] from query
  4. Score only candidate tokens using PQ lookups (massively smaller than full corpus)
  5. MaxSim aggregate per doc
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

device = "cpu"
model = model.to(device)
linear = linear.to(device)


# ---- Load combined index ----
print("Loading IVF+PQ index...")
with open("data/scifact_ivf_pq.pkl", "rb") as f:
    idx = pickle.load(f)

ivf_centroids = idx["ivf_centroids"].astype(np.float32)    # [K_IVF, D]
posting_lists = idx["posting_lists"]                        # list of K_IVF arrays
pq_codebooks = idx["pq_codebooks"].astype(np.float32)      # [M, K, chunk_dim]
pq_codes = idx["pq_codes"]                                  # [N_tokens, M] uint8
token_to_doc = idx["token_to_doc"]                          # [N_tokens]

K_IVF = idx["K_IVF"]
M = idx["M_PQ"]
K = idx["K_PQ"]
D = idx["D"]
chunk_dim = D // M
N_tokens = pq_codes.shape[0]

# Load doc info
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
doc_ids = corpus["doc_ids"]
NUM_DOCS = len(doc_ids)

print(f"  K_IVF={K_IVF}, M_PQ={M}, K_PQ={K}, D={D}")
print(f"  Corpus: {N_tokens} tokens, {NUM_DOCS} docs")


# ---- Pre-convert to torch tensors (once, at startup) ----
ivf_centroids_t = torch.from_numpy(ivf_centroids)              # [K_IVF, D]
pq_codebooks_t = torch.from_numpy(pq_codebooks)                # [M, K, chunk_dim]
token_to_doc_t = torch.from_numpy(token_to_doc.astype(np.int64))


# ---- Query encoder ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1).numpy()


# ---- IVF + PQ search ----
def search_ivf_pq(query_text, nprobe=8, top_k=100):
    Q = encode_query(query_text)                                # [num_q, D]
    num_q = Q.shape[0]

    # ── Step 1: IVF candidate generation ──
    # For each query token, find the nprobe nearest centroids
    Q_t = torch.from_numpy(Q)
    centroid_sim = Q_t @ ivf_centroids_t.T                       # [num_q, K_IVF]
    _, top_clusters = centroid_sim.topk(nprobe, dim=1)           # [num_q, nprobe]
    
    # Union all token IDs from the probed clusters across all query tokens
    probed = top_clusters.flatten().unique().tolist()
    candidate_token_ids = np.concatenate([posting_lists[c] for c in probed])
    candidate_token_ids = np.unique(candidate_token_ids)
    n_candidates = len(candidate_token_ids)

    if n_candidates == 0:
        return [], 0

    # ── Step 2: Build PQ lookup tables ──
    Q_chunks = Q.reshape(num_q, M, chunk_dim)
    lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, pq_codebooks)
    lookups_t = torch.from_numpy(lookups_np)                     # [num_q, M, K]

    # ── Step 3: Score ONLY candidates using PQ lookups ──
    # Slice out only the codes for candidate tokens
    candidate_codes = pq_codes[candidate_token_ids]              # [n_cands, M] uint8
    candidate_codes_t = [
        torch.from_numpy(candidate_codes[:, m].astype(np.int64))
        for m in range(M)
    ]
    
    sim = torch.zeros((num_q, n_candidates), dtype=torch.float32)
    for m in range(M):
        sim += lookups_t[:, m, candidate_codes_t[m]]             # [num_q, n_cands]

    # ── Step 4: Per-doc max + sum ──
    candidate_doc_ids = token_to_doc_t[
        torch.from_numpy(candidate_token_ids.astype(np.int64))
    ]                                                            # [n_cands]
    
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=candidate_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=sim,
        reduce="amax",
        include_self=True,
    )
    scores = per_doc_max.sum(dim=0)

    top_values, top_indices = scores.topk(top_k)
    return [
        (doc_ids[i], top_values[idx].item())
        for idx, i in enumerate(top_indices.tolist())
    ], n_candidates


# ---- Metrics ----
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
    search_ivf_pq(queries[0].text, nprobe=8, top_k=100)


# ---- Sweep nprobe ----
print("\n" + "=" * 70)
print(f"IVF+PQ sweep on SciFact (M={M}, K_IVF={K_IVF})")
print("=" * 70)

nprobe_values = [1, 2, 4, 8, 16, 32, 64]
sweep_results = []

for nprobe in nprobe_values:
    ndcgs, recalls, latencies, cand_counts = [], [], [], []
    
    for q in tqdm(queries, ncols=80, leave=False, desc=f"nprobe={nprobe}"):
        relevant = qrels_per_query[q.query_id]
        t0 = time.perf_counter()
        results, n_cands = search_ivf_pq(q.text, nprobe=nprobe, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)
        cand_counts.append(n_cands)
        
        retrieved = [d for d, _ in results]
        ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        recalls.append(recall_at_k(retrieved, relevant, 100))
    
    mean_ndcg = statistics.mean(ndcgs)
    mean_recall = statistics.mean(recalls)
    median_lat = statistics.median(latencies)
    mean_cands = statistics.mean(cand_counts)
    pct_corpus = 100 * mean_cands / N_tokens
    
    sweep_results.append({
        "nprobe": nprobe,
        "ndcg_at_10": mean_ndcg,
        "recall_at_100": mean_recall,
        "median_latency_ms": median_lat,
        "mean_candidates": mean_cands,
        "pct_corpus": pct_corpus,
    })


# ---- Report ----
print("\n" + "=" * 70)
print("Final summary: IVF+PQ on SciFact")
print("=" * 70)
print(f"{'nprobe':>7} | {'nDCG@10':>8} | {'Recall':>7} | {'lat ms':>7} | "
      f"{'cands':>7} | {'%corpus':>8}")
print("-" * 70)
for r in sweep_results:
    print(f"{r['nprobe']:>7} | {r['ndcg_at_10']:>8.4f} | "
          f"{r['recall_at_100']:>7.4f} | {r['median_latency_ms']:>7.1f} | "
          f"{int(r['mean_candidates']):>7} | {r['pct_corpus']:>7.1f}%")
print("-" * 70)
print()
print("Reference points:")
print(f"  Brute force (Day 11):     nDCG=0.6122, lat=108 ms")
print(f"  PQ-only fast (Day 16):    nDCG=0.6124, lat=1141 ms")
print()


# ---- Persist ----
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system": "ivf_pq_combined",
    "K_IVF": K_IVF, "M_PQ": M, "K_PQ": K,
    "dataset": "scifact",
    "num_queries": len(queries),
    "sweep": sweep_results,
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/ivf_pq_sweep.json", "w") as f:
    json.dump(record, f, indent=2)
print(f"Saved to benchmarks/results/ivf_pq_sweep.json")