"""
IVF+PQ sweep on MS MARCO.

Uses the same 4 hardcoded test queries as Day 10 (no MS MARCO qrels here).
Recall is computed against brute-force ground truth, just like Day 9-10.

The point of this benchmark is the SCALING comparison: does IVF+PQ
actually win at this scale, where brute force is slower (~250 ms vs
SciFact's 108 ms)?
"""
import pickle
import json
import time
import datetime
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


# ---- Load MS MARCO corpus AND IVF+PQ index ----
print("Loading MS MARCO corpus...")
with open("data/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
texts = corpus["texts"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS = len(texts)

# Pre-stack uncompressed vectors for brute force comparison
all_doc_vectors = torch.cat(doc_vectors_list, dim=0)
print(f"  {NUM_DOCS} docs, {all_doc_vectors.shape[0]} tokens")

print("Loading IVF+PQ index...")
with open("data/msmarco_ivf_pq.pkl", "rb") as f:
    idx = pickle.load(f)

ivf_centroids = idx["ivf_centroids"].astype(np.float32)
posting_lists = idx["posting_lists"]
pq_codebooks = idx["pq_codebooks"].astype(np.float32)
pq_codes = idx["pq_codes"]
token_to_doc = idx["token_to_doc"]

K_IVF = idx["K_IVF"]
M = idx["M_PQ"]
K = idx["K_PQ"]
D = idx["D"]
chunk_dim = D // M
N_tokens = pq_codes.shape[0]

# Pre-convert to torch
ivf_centroids_t = torch.from_numpy(ivf_centroids)
pq_codebooks_t = torch.from_numpy(pq_codebooks)
token_to_doc_t = torch.from_numpy(token_to_doc.astype(np.int64))

print(f"  K_IVF={K_IVF}, M={M}, K={K}")


# ---- Query encoder ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1)


# ---- Brute force search (uses uncompressed vectors) ----
def search_bruteforce(query_text, top_k=10):
    Q = encode_query(query_text)
    sim_all = Q @ all_doc_vectors.T
    num_q = sim_all.shape[0]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_all.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc_t.unsqueeze(0).expand(num_q, -1),
        src=sim_all, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    return list(zip(top_indices.tolist(), top_values.tolist()))


# ---- IVF+PQ search ----
def search_ivf_pq(query_text, nprobe=8, top_k=10):
    Q_t = encode_query(query_text)
    Q = Q_t.numpy()
    num_q = Q.shape[0]

    # IVF: find candidate clusters
    centroid_sim = Q_t @ ivf_centroids_t.T
    _, top_clusters = centroid_sim.topk(nprobe, dim=1)
    
    probed = top_clusters.flatten().unique().tolist()
    candidate_token_ids = np.concatenate([posting_lists[c] for c in probed])
    candidate_token_ids = np.unique(candidate_token_ids)
    n_candidates = len(candidate_token_ids)

    if n_candidates == 0:
        return [], 0

    # PQ lookups
    Q_chunks = Q.reshape(num_q, M, chunk_dim)
    lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, pq_codebooks)
    lookups_t = torch.from_numpy(lookups_np)

    candidate_codes = pq_codes[candidate_token_ids]
    candidate_codes_t = [
        torch.from_numpy(candidate_codes[:, m].astype(np.int64))
        for m in range(M)
    ]

    sim = torch.zeros((num_q, n_candidates), dtype=torch.float32)
    for m in range(M):
        sim += lookups_t[:, m, candidate_codes_t[m]]

    candidate_doc_ids = token_to_doc_t[
        torch.from_numpy(candidate_token_ids.astype(np.int64))
    ]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=candidate_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=sim, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)

    top_values, top_indices = scores.topk(top_k)
    return list(zip(top_indices.tolist(), top_values.tolist())), n_candidates


# ---- Test queries (same set used since Day 6) ----
queries = [
    "history of the manhattan project",
    "vitamin D deficiency symptoms",
    "weather forecast",
    "what causes thunderstorms",
]


# ---- Warm up ----
print("\nWarming up...")
for q in queries[:2]:
    search_bruteforce(q, top_k=10)
    search_ivf_pq(q, nprobe=8, top_k=10)


# ---- Compute brute force ground truth + baseline latency ----
print("\nComputing brute-force ground truth (3 runs each)...")
ground_truth = {}
bf_latencies = []

for q in queries:
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        results = search_bruteforce(q, top_k=10)
        runs.append(time.perf_counter() - t0)
    bf_latencies.append(statistics.median(runs))
    ground_truth[q] = set(d for d, _ in results)

bf_median_lat = statistics.median(bf_latencies) * 1000
print(f"Brute force median latency: {bf_median_lat:.1f} ms")


# ---- Sweep nprobe ----
print("\n" + "=" * 70)
print(f"IVF+PQ sweep on MS MARCO ({NUM_DOCS} docs, {N_tokens} tokens)")
print("=" * 70)

nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]
sweep_results = []

for nprobe in nprobe_values:
    recalls = []
    latencies = []
    cand_counts = []

    for q in queries:
        # Recall (deterministic — single run)
        results, n_cands = search_ivf_pq(q, nprobe=nprobe, top_k=10)
        retrieved = set(d for d, _ in results)
        recall = len(retrieved & ground_truth[q]) / 10
        recalls.append(recall)
        cand_counts.append(n_cands)

        # Latency over 3 runs
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            search_ivf_pq(q, nprobe=nprobe, top_k=10)
            runs.append(time.perf_counter() - t0)
        latencies.append(statistics.median(runs))

    mean_recall = statistics.mean(recalls)
    median_lat = statistics.median(latencies) * 1000
    mean_cands = statistics.mean(cand_counts)
    pct_corpus = 100 * mean_cands / N_tokens

    sweep_results.append({
        "nprobe": nprobe,
        "recall_at_10": mean_recall,
        "median_latency_ms": median_lat,
        "mean_candidates": mean_cands,
        "pct_corpus": pct_corpus,
        "speedup_vs_bf": bf_median_lat / median_lat,
    })

    print(f"  nprobe={nprobe:>3} | recall={mean_recall:.2%} | "
          f"lat={median_lat:>6.1f} ms | "
          f"cands={int(mean_cands):>7} | "
          f"speedup vs BF: {bf_median_lat/median_lat:>4.2f}x")


# ---- Final report ----
print("\n" + "=" * 70)
print(f"Final summary: IVF+PQ on MS MARCO ({NUM_DOCS} docs)")
print("=" * 70)
print(f"{'nprobe':>7} | {'recall':>7} | {'lat ms':>7} | {'speedup':>8} | "
      f"{'cands':>7} | {'%corpus':>8}")
print("-" * 70)
for r in sweep_results:
    marker = " ←" if r["speedup_vs_bf"] >= 1.0 and r["recall_at_10"] >= 0.7 else ""
    print(f"{r['nprobe']:>7} | {r['recall_at_10']:>6.1%} | "
          f"{r['median_latency_ms']:>7.1f} | "
          f"{r['speedup_vs_bf']:>7.2f}x | "
          f"{int(r['mean_candidates']):>7} | "
          f"{r['pct_corpus']:>7.1f}%{marker}")
print("-" * 70)
print(f"  Brute force baseline: 100% recall, {bf_median_lat:.1f} ms")
print()

# Reference: SciFact result for the comparison chart
print("Reference: SciFact (5K docs) IVF+PQ never beat brute force at any nprobe.")
print("           Day 10 IVF-only on this corpus: nprobe=8 was 1.95x faster than BF.")


# ---- Persist ----
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system": "ivf_pq_combined",
    "dataset": "msmarco",
    "num_docs": NUM_DOCS,
    "num_tokens": N_tokens,
    "K_IVF": K_IVF, "M_PQ": M, "K_PQ": K,
    "bf_median_latency_ms": bf_median_lat,
    "sweep": sweep_results,
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/msmarco_ivf_pq_sweep.json", "w") as f:
    json.dump(record, f, indent=2)
print(f"Saved to benchmarks/results/msmarco_ivf_pq_sweep.json")