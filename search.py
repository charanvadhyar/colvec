from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import statistics

# ---- Setup ----
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

print("Loading projection layer...")
weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state = load_file(weights_path)
linear = nn.Linear(768, 128, bias=False)
linear.weight.data = state["linear.weight"]
linear.eval()

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device)
linear = linear.to(device)

# ---- Load corpus ----
print("Loading corpus...")
with open("data/corpus.pkl", "rb") as f:
    data = pickle.load(f)
texts = data["texts"]
doc_vectors = data["vectors"]
print(f"Loaded {len(texts)} passages.")

# Pre-stack
all_doc_vectors = torch.cat(doc_vectors, dim=0)
doc_offsets = [0]
for D in doc_vectors:
    doc_offsets.append(doc_offsets[-1] + D.shape[0])
print(f"Stacked corpus shape: {all_doc_vectors.shape}")

# ========================================================================
# NEW (Day 7): Build a token-to-doc index ONCE at startup.
# token_to_doc[j] = which doc the j-th row of all_doc_vectors belongs to.
# This is what lets us replace the per-query Python loop with a single
# scatter_reduce call — we need to know how to "group by doc" in tensor land.
# This loop runs only once at startup, not per query.
# ========================================================================
NUM_DOCS = len(doc_vectors)
token_to_doc = torch.empty(all_doc_vectors.shape[0], dtype=torch.long)
for i in range(NUM_DOCS):
    start, end = doc_offsets[i], doc_offsets[i + 1]
    token_to_doc[start:end] = i
print(f"token_to_doc shape: {token_to_doc.shape}  (one doc_id per corpus token)")


# ---- Encoder ----
def encode_query(text, max_len=32):
    inputs = tokenizer(
        "[Q] " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1).cpu()


# ---- OLD Search (kept for side-by-side comparison) ----
# Iterates over every doc in a Python for-loop. At 10K docs this takes
# ~170 ms and is 76% of total query latency. Left intact so we can
# prove the new version gives identical results.
def search(query_text, top_k=10):
    Q = encode_query(query_text)
    sim_all = Q @ all_doc_vectors.T

    scores = []
    for i in range(len(doc_vectors)):
        start, end = doc_offsets[i], doc_offsets[i + 1]
        sim = sim_all[:, start:end]
        score = sim.max(dim=1).values.sum().item()
        scores.append(score)

    return sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]


# ========================================================================
# NEW (Day 7): Vectorized search.
#
# What changed:
#   - The Python for-loop is gone. In its place, a single PyTorch call
#     (scatter_reduce_) computes the per-doc, per-query-token max similarity
#     in one shot, implemented in C++ with no Python overhead.
#   - sorted(...) replaced with .topk(k), which is an optimized C++ primitive
#     for selecting the top k values without fully sorting the array.
#
# Algorithm (identical to the old one, just expressed as tensor ops):
#   1. sim_all[i, j] = cosine similarity between query token i and corpus
#      token j. Same as before.
#   2. For each query token i and each doc d, find the max similarity across
#      all tokens of doc d. The scatter_reduce groups sim_all columns by
#      token_to_doc, reducing with amax.
#   3. Sum across query tokens → one MaxSim score per doc.
#   4. topk for the final ranking.
#
# Why scatter_reduce works here:
#   - We want: for each (query_token i, doc d), max_{j in doc d} sim_all[i, j]
#   - scatter_reduce_(dim=1, index=token_to_doc, src=sim_all, reduce="amax")
#     does exactly that: values from src are scattered into the destination
#     at positions given by index, combining collisions with max.
# ========================================================================
def search_vectorized(query_text, top_k=10):
    Q = encode_query(query_text)                  # [num_q_tokens, 128]
    sim_all = Q @ all_doc_vectors.T               # [num_q_tokens, total_tokens]
    num_q_tokens = sim_all.shape[0]

    # For each query token × each doc, start from -inf and take max.
    # -inf means "no tokens yet" — any real similarity overwrites it.
    per_doc_max = torch.full(
        (num_q_tokens, NUM_DOCS), float("-inf"), dtype=sim_all.dtype
    )

    # Broadcast token_to_doc from [total_tokens] to [num_q_tokens, total_tokens]
    # so its shape matches sim_all. expand() is a zero-copy view, not a real
    # allocation, so this is essentially free.
    index = token_to_doc.unsqueeze(0).expand(num_q_tokens, -1)

    # The big win: one call replaces the entire 10K-iteration Python loop.
    per_doc_max.scatter_reduce_(
        dim=1,
        index=index,
        src=sim_all,
        reduce="amax",
        include_self=True,
    )
    # per_doc_max[i, d] is now "best similarity between query token i and any
    # token in doc d" — same as sim.max(dim=1) did per-doc in the old code.

    # Sum across query tokens → final MaxSim score per doc. Same as the
    # .sum() that came after .max() in the old per-doc scoring.
    scores = per_doc_max.sum(dim=0)               # [NUM_DOCS]

    # Top-k via PyTorch's optimized routine instead of sorted(). Fast and
    # avoids building a Python list of 10K (doc_id, score) tuples first.
    top_values, top_indices = scores.topk(top_k)
    return list(zip(top_indices.tolist(), top_values.tolist()))


# ---- Queries ----
queries = [
    "history of the manhattan project",
    "vitamin D deficiency symptoms",
    "weather forecast",
    "what causes thunderstorms",
]

# ---- Warm-up (both versions) ----
print("\nWarming up...")
for q in queries[:2]:
    search(q, top_k=10)
    search_vectorized(q, top_k=10)   # NEW: warm up the vectorized path too


# ========================================================================
# NEW (Day 7): Correctness check BEFORE benchmarking.
# A faster-but-wrong implementation is worse than a slow correct one.
# If top-10 doc IDs don't match, something is broken and we shouldn't
# trust any speedup numbers.
# ========================================================================
print("\n" + "=" * 60)
print("Correctness check: old vs vectorized")
print("=" * 60)

for test_query in queries:
    old_results = search(test_query, top_k=10)
    new_results = search_vectorized(test_query, top_k=10)

    old_ids = [doc_id for doc_id, _ in old_results]
    new_ids = [doc_id for doc_id, _ in new_results]

    old_scores = {i: s for i, s in old_results}
    new_scores = {i: s for i, s in new_results}

    if old_ids == new_ids:
        # Compare scores too — should be near-identical (float rounding OK)
        max_diff = max(abs(old_scores[i] - new_scores[i]) for i in old_ids)
        print(f"  ✓ {test_query!r:50s}  identical top-10  (max score diff {max_diff:.2e})")
    else:
        overlap = set(old_ids) & set(new_ids)
        print(f"  ✗ {test_query!r:50s}  MISMATCH  overlap={len(overlap)}/10")
        # Show the divergence for debugging
        for doc_id in list(set(old_ids) - set(new_ids))[:3]:
            print(f"      only in old: doc {doc_id} score {old_scores[doc_id]:.4f}")
        for doc_id in list(set(new_ids) - set(old_ids))[:3]:
            print(f"      only in new: doc {doc_id} score {new_scores[doc_id]:.4f}")


# ---- OLD benchmark (unchanged) ----
print("\n" + "=" * 60)
print("OLD benchmark: Python for-loop scoring (median of 5 runs)")
print("=" * 60)

all_results = {}
for query in queries:
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        results = search(query, top_k=10)
        runs.append(time.perf_counter() - start)

    median_ms = statistics.median(runs) * 1000
    p99_ms = max(runs) * 1000
    all_results[query] = (results, median_ms, p99_ms)

    print(f"\nQuery: {query!r}")
    print(f"  median: {median_ms:.0f} ms   p99: {p99_ms:.0f} ms")
    for rank, (doc_id, score) in enumerate(results[:3], 1):
        snippet = texts[doc_id][:100].replace("\n", " ")
        print(f"  {rank}. [{score:5.2f}] {snippet}...")

medians = [m for _, m, _ in all_results.values()]
overall_median = statistics.median(medians)
print(f"\nOld overall median: {overall_median:.0f} ms  ({1000/overall_median:.1f} qps)")


# ========================================================================
# NEW (Day 7): Benchmark the vectorized version.
# Same 4 queries, same 5-run-median protocol — only the implementation
# changed. This makes the comparison clean: any latency difference is
# directly attributable to the scatter_reduce rewrite.
# ========================================================================
print("\n" + "=" * 60)
print("NEW benchmark: vectorized scatter_reduce (median of 5 runs)")
print("=" * 60)

new_all_results = {}
for query in queries:
    runs = []
    for _ in range(5):
        start = time.perf_counter()
        results = search_vectorized(query, top_k=10)
        runs.append(time.perf_counter() - start)

    median_ms = statistics.median(runs) * 1000
    p99_ms = max(runs) * 1000
    new_all_results[query] = (results, median_ms, p99_ms)

    print(f"\nQuery: {query!r}")
    print(f"  median: {median_ms:.0f} ms   p99: {p99_ms:.0f} ms")
    for rank, (doc_id, score) in enumerate(results[:3], 1):
        snippet = texts[doc_id][:100].replace("\n", " ")
        print(f"  {rank}. [{score:5.2f}] {snippet}...")

new_medians = [m for _, m, _ in new_all_results.values()]
new_overall = statistics.median(new_medians)
print(f"\nNew overall median: {new_overall:.0f} ms  ({1000/new_overall:.1f} qps)")


# ========================================================================
# NEW (Day 7): Side-by-side speedup summary.
# ========================================================================
print("\n" + "=" * 60)
print("Summary: Day 6 (Python loop) vs Day 7 (vectorized)")
print("=" * 60)
print(f"  Latency:    {overall_median:6.0f} ms  →  {new_overall:6.0f} ms   "
      f"({overall_median/new_overall:.1f}x faster)")
print(f"  Throughput: {1000/overall_median:6.1f} qps →  {1000/new_overall:6.1f} qps   "
      f"({(1000/new_overall) / (1000/overall_median):.1f}x more)")


# ---- OLD profile (unchanged) ----
print("\n" + "=" * 60)
print("OLD profile breakdown (Python for-loop scoring, avg of 10)")
print("=" * 60)

profile_query = queries[0]
N = 10
t_encode, t_matmul, t_score = 0.0, 0.0, 0.0

for _ in range(N):
    t0 = time.perf_counter()
    Q = encode_query(profile_query)
    t1 = time.perf_counter()

    sim_all = Q @ all_doc_vectors.T
    t2 = time.perf_counter()

    scores = []
    for i in range(len(doc_vectors)):
        start, end = doc_offsets[i], doc_offsets[i + 1]
        sim = sim_all[:, start:end]
        score = sim.max(dim=1).values.sum().item()
        scores.append(score)
    sorted(enumerate(scores), key=lambda x: -x[1])[:10]
    t3 = time.perf_counter()

    t_encode += (t1 - t0)
    t_matmul += (t2 - t1)
    t_score  += (t3 - t2)

avg_encode_old = t_encode / N * 1000
avg_matmul_old = t_matmul / N * 1000
avg_score_old  = t_score  / N * 1000
avg_total_old  = avg_encode_old + avg_matmul_old + avg_score_old

print(f"  Query: {profile_query!r}")
print(f"  Query encoding:     {avg_encode_old:6.1f} ms  ({avg_encode_old/avg_total_old*100:4.1f}%)")
print(f"  Matmul vs corpus:   {avg_matmul_old:6.1f} ms  ({avg_matmul_old/avg_total_old*100:4.1f}%)")
print(f"  Per-doc scoring:    {avg_score_old:6.1f} ms  ({avg_score_old/avg_total_old*100:4.1f}%)")
print(f"  Total:              {avg_total_old:6.1f} ms")


# ========================================================================
# NEW (Day 7): Profile the vectorized version.
# Same structure as the old profile but scoring is now broken into:
#   - scatter_reduce (the replacement for the Python loop)
#   - topk (new, replaces sorted())
# Watch the "Per-doc scoring" row collapse. Whatever rises to #1 in the
# percentage column is the NEXT bottleneck to attack (probably query
# encoding or the matmul itself).
# ========================================================================
print("\n" + "=" * 60)
print("NEW profile breakdown (vectorized, avg of 10)")
print("=" * 60)

t_encode, t_matmul, t_score, t_topk = 0.0, 0.0, 0.0, 0.0

for _ in range(N):
    t0 = time.perf_counter()
    Q = encode_query(profile_query)
    t1 = time.perf_counter()

    sim_all = Q @ all_doc_vectors.T
    t2 = time.perf_counter()

    num_q_tokens = sim_all.shape[0]
    per_doc_max = torch.full(
        (num_q_tokens, NUM_DOCS), float("-inf"), dtype=sim_all.dtype
    )
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc.unsqueeze(0).expand(num_q_tokens, -1),
        src=sim_all,
        reduce="amax",
        include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    t3 = time.perf_counter()

    scores.topk(10)
    t4 = time.perf_counter()

    t_encode += (t1 - t0)
    t_matmul += (t2 - t1)
    t_score  += (t3 - t2)
    t_topk   += (t4 - t3)

avg_encode = t_encode / N * 1000
avg_matmul = t_matmul / N * 1000
avg_score  = t_score  / N * 1000
avg_topk   = t_topk   / N * 1000
avg_total  = avg_encode + avg_matmul + avg_score + avg_topk

print(f"  Query: {profile_query!r}")
print(f"  Query encoding:     {avg_encode:6.1f} ms  ({avg_encode/avg_total*100:4.1f}%)")
print(f"  Matmul vs corpus:   {avg_matmul:6.1f} ms  ({avg_matmul/avg_total*100:4.1f}%)")
print(f"  Vectorized score:   {avg_score:6.1f} ms  ({avg_score/avg_total*100:4.1f}%)")
print(f"  Top-k:              {avg_topk:6.1f} ms  ({avg_topk/avg_total*100:4.1f}%)")
print(f"  Total:              {avg_total:6.1f} ms")