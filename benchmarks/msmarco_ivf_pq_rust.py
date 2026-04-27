"""
IVF+PQ+Rust benchmark on MS MARCO 30K corpus.

Architecture:
  1. Encode query (ColBERT, uncompressed)
  2. IVF: find nprobe nearest clusters → candidate token IDs
  3. Slice PQ codes for candidates only
  4. Rust AVX2 kernel: score candidates via lookup tables
  5. scatter_reduce MaxSim aggregation per doc

This combines three optimizations for the first time:
  - IVF candidate filtering (skip ~96% of corpus at nprobe=8)
  - PQ compression (16x smaller storage)
  - Rust AVX2 kernel (6x faster scoring than Python)
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
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import colvec_kernel


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


# ---- Load MS MARCO corpus (for brute force baseline) ----
print("Loading MS MARCO corpus...")
with open("data/corpus.pkl", "rb") as f:
    corpus = pickle.load(f)
texts            = corpus["texts"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS         = len(texts)

all_doc_vectors = torch.cat(doc_vectors_list, dim=0)
print(f"  {NUM_DOCS} docs, {all_doc_vectors.shape[0]} tokens")


# ---- Load IVF+PQ index ----
print("Loading IVF+PQ index...")
with open("data/msmarco_ivf_pq.pkl", "rb") as f:
    idx = pickle.load(f)

ivf_centroids = torch.from_numpy(idx["ivf_centroids"].astype(np.float32))
posting_lists = idx["posting_lists"]
pq_codebooks  = idx["pq_codebooks"].astype(np.float32)
pq_codes      = idx["pq_codes"]                       # [N_tokens, M] uint8
token_to_doc  = idx["token_to_doc"]

K_IVF = idx["K_IVF"]
M     = idx["M_PQ"]
K     = idx["K_PQ"]
D     = idx["D"]
chunk_dim = D // M
N_tokens  = pq_codes.shape[0]

print(f"  K_IVF={K_IVF}, M={M}, K={K}, D={D}")
print(f"  {N_tokens} tokens, {NUM_DOCS} docs")

# Pre-convert token_to_doc once
token_to_doc_t = torch.from_numpy(token_to_doc.astype(np.int64))

# ---- Pre-transpose PQ codes for Rust kernel ----
# Rust kernel expects [M, N_tokens] layout.
# We transpose the FULL corpus here, but only slice candidates per query.
# The slice of a transposed array is contiguous per chunk row — perfect.
print("Pre-transposing PQ codes for Rust kernel...")
pq_codes_T = np.ascontiguousarray(pq_codes.T)         # [M, N_tokens]
print(f"  pq_codes_T shape: {pq_codes_T.shape}, "
      f"C-contiguous: {pq_codes_T.flags['C_CONTIGUOUS']}")


# ---- Query encoder ----
def encode_query(text, max_len=32):
    inputs = tokenizer("[Q] " + text, return_tensors="pt",
                       truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1)


# ---- Brute force baseline ----
def search_bruteforce(query_text, top_k=10):
    Q = encode_query(query_text)
    sim_all = Q @ all_doc_vectors.T
    num_q   = sim_all.shape[0]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"))
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc_t.unsqueeze(0).expand(num_q, -1),
        src=sim_all, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    return list(zip(top_indices.tolist(), top_values.tolist()))


# ---- IVF+PQ+Rust search ----
def search_ivf_pq_rust(query_text, nprobe=8, top_k=10):
    Q_t   = encode_query(query_text)               # [num_q, D] torch
    Q_np  = Q_t.numpy()
    num_q = Q_np.shape[0]

    # ── Step 1: IVF candidate filtering ──
    centroid_sim = Q_t @ ivf_centroids.T            # [num_q, K_IVF]
    _, top_clusters = centroid_sim.topk(
        min(nprobe, K_IVF), dim=1
    )
    probed = top_clusters.flatten().unique().tolist()
    candidate_ids = np.unique(
        np.concatenate([posting_lists[c] for c in probed])
    )
    n_cands = len(candidate_ids)
    if n_cands == 0:
        return [], 0

    # ── Step 2: Build PQ lookup tables ──
    Q_chunks   = Q_np.reshape(num_q, M, chunk_dim)
    lookups_np = np.einsum(
        "qmd,mkd->qmk", Q_chunks, pq_codebooks
    ).astype(np.float32)                            # [num_q, M, K]

    # ── Step 3: Slice candidate codes ──
    # pq_codes_T is [M, N_tokens].
    # Slicing columns gives [M, n_cands] — each chunk row is still contiguous.
    candidate_codes_T = np.ascontiguousarray(
        pq_codes_T[:, candidate_ids]
    )                                               # [M, n_cands]

    # ── Step 4: Rust AVX2 kernel ──
    # Input:  lookups [num_q, M, K], candidate_codes_T [M, n_cands]
    # Output: sim [num_q, n_cands]
    sim_np = colvec_kernel.apply_pq_lookups(lookups_np, candidate_codes_T)

    # ── Step 5: MaxSim per doc ──
    sim_t = torch.from_numpy(sim_np)                # [num_q, n_cands]

    candidate_doc_ids = token_to_doc_t[
        torch.from_numpy(candidate_ids.astype(np.int64))
    ]                                               # [n_cands]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_t.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=candidate_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=sim_t, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)

    top_values, top_indices = scores.topk(top_k)
    return list(zip(top_indices.tolist(), top_values.tolist())), n_cands


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
    search_ivf_pq_rust(q, nprobe=8, top_k=10)


# ---- Brute force ground truth + baseline latency ----
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

bf_median_ms = statistics.median(bf_latencies) * 1000
print(f"  Brute force median: {bf_median_ms:.1f} ms")


# ---- Diagnostic: per-stage timing ----
print("\n" + "=" * 65)
print("Diagnostic: per-stage timing on one query (5 runs)")
print("=" * 65)

q_text = queries[0]
N_DIAG = 5
t_encode = t_ivf = t_lookups = t_slice = t_rust = t_scatter = 0.0

for _ in range(N_DIAG):
    t0 = time.perf_counter()
    Q_t   = encode_query(q_text)
    Q_np  = Q_t.numpy()
    num_q = Q_np.shape[0]
    t1 = time.perf_counter()

    centroid_sim = Q_t @ ivf_centroids.T
    _, top_clusters = centroid_sim.topk(8, dim=1)
    probed = top_clusters.flatten().unique().tolist()
    candidate_ids = np.unique(np.concatenate([posting_lists[c] for c in probed]))
    n_cands = len(candidate_ids)
    t2 = time.perf_counter()

    Q_chunks   = Q_np.reshape(num_q, M, chunk_dim)
    lookups_np = np.einsum("qmd,mkd->qmk", Q_chunks, pq_codebooks).astype(np.float32)
    t3 = time.perf_counter()

    candidate_codes_T = np.ascontiguousarray(pq_codes_T[:, candidate_ids])
    t4 = time.perf_counter()

    sim_np = colvec_kernel.apply_pq_lookups(lookups_np, candidate_codes_T)
    t5 = time.perf_counter()

    sim_t = torch.from_numpy(sim_np)
    candidate_doc_ids = token_to_doc_t[torch.from_numpy(candidate_ids.astype(np.int64))]
    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_t.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=candidate_doc_ids.unsqueeze(0).expand(num_q, -1),
        src=sim_t, reduce="amax", include_self=True,
    )
    per_doc_max.sum(dim=0).topk(10)
    t6 = time.perf_counter()

    t_encode  += t1 - t0
    t_ivf     += t2 - t1
    t_lookups += t3 - t2
    t_slice   += t4 - t3
    t_rust    += t5 - t4
    t_scatter += t6 - t5

avg = lambda x: x / N_DIAG * 1000
print(f"  query encode:         {avg(t_encode):7.1f} ms")
print(f"  IVF candidate lookup: {avg(t_ivf):7.1f} ms")
print(f"  build lookups:        {avg(t_lookups):7.1f} ms")
print(f"  slice candidate codes:{avg(t_slice):7.1f} ms")
print(f"  Rust AVX2 scoring:    {avg(t_rust):7.1f} ms  ({n_cands} candidates)")
print(f"  scatter_reduce:       {avg(t_scatter):7.1f} ms")
total = avg(t_encode + t_ivf + t_lookups + t_slice + t_rust + t_scatter)
print(f"  total:                {total:7.1f} ms")
print(f"\n  Brute force baseline: {bf_median_ms:.1f} ms")
print(f"  Speedup vs BF:        {bf_median_ms/total:.2f}x")


# ---- nprobe sweep ----
print("\n" + "=" * 70)
print(f"IVF+PQ+Rust sweep on MS MARCO ({NUM_DOCS} docs)")
print("=" * 70)

nprobe_values = [1, 2, 4, 8, 16, 32, 64]
sweep_results = []

for nprobe in nprobe_values:
    recalls, latencies, cand_counts = [], [], []

    for q in queries:
        # Recall
        results, n_cands = search_ivf_pq_rust(q, nprobe=nprobe, top_k=10)
        retrieved = set(d for d, _ in results)
        recalls.append(len(retrieved & ground_truth[q]) / 10)
        cand_counts.append(n_cands)

        # Latency (3 runs)
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            search_ivf_pq_rust(q, nprobe=nprobe, top_k=10)
            runs.append(time.perf_counter() - t0)
        latencies.append(statistics.median(runs))

    mean_recall  = statistics.mean(recalls)
    median_lat   = statistics.median(latencies) * 1000
    mean_cands   = statistics.mean(cand_counts)
    pct_corpus   = 100 * mean_cands / N_tokens
    speedup      = bf_median_ms / median_lat

    sweep_results.append({
        "nprobe":      nprobe,
        "recall":      mean_recall,
        "latency_ms":  median_lat,
        "candidates":  mean_cands,
        "pct_corpus":  pct_corpus,
        "speedup":     speedup,
    })

    marker = " ←" if speedup >= 1.0 and mean_recall >= 0.7 else ""
    print(f"  nprobe={nprobe:>3} | recall={mean_recall:.1%} | "
          f"lat={median_lat:>6.1f}ms | cands={int(mean_cands):>7} | "
          f"speedup={speedup:>5.2f}x{marker}")


# ---- Final report ----
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"\n  Brute force:        {bf_median_ms:.1f} ms (100% recall)")
print()
print("  IVF+PQ+Rust:")
for r in sweep_results:
    print(f"    nprobe={r['nprobe']:>3}:  {r['latency_ms']:>6.1f} ms, "
          f"{r['recall']:.1%} recall, {r['speedup']:.2f}x vs BF")

print()
print("  Full progression (apply-lookups stage):")
print("    Python slow (Day 15): 2761 ms")
print("    Python torch (Day 16):  682 ms")
print("    Rust full corpus:       121 ms")
print(f"    Rust candidates only:  {avg(t_rust):.1f} ms  "
      f"({n_cands}/{N_tokens} tokens = "
      f"{100*n_cands/N_tokens:.1f}% of corpus)")


# ---- Persist ----
record = {
    "timestamp":   datetime.datetime.now().isoformat(),
    "system":      "ivf_pq_rust",
    "dataset":     "msmarco",
    "num_docs":    NUM_DOCS,
    "num_tokens":  N_tokens,
    "K_IVF":       K_IVF,
    "M_PQ":        M,
    "bf_median_ms": bf_median_ms,
    "sweep":       sweep_results,
}
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/msmarco_ivf_pq_rust_sweep.json", "w") as f:
    json.dump(record, f, indent=2)
print(f"\nSaved to benchmarks/results/msmarco_ivf_pq_rust_sweep.json")