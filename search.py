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

# ---- Search ----
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

# ---- Queries ----
queries = [
    "history of the manhattan project",
    "vitamin D deficiency symptoms",
    "weather forecast",
    "what causes thunderstorms",
]

# ---- Warm-up ----
# First call always pays one-time costs (lazy init, kernel launches, page-ins).
# Run a couple of throwaway queries so the benchmark measures steady-state.
print("\nWarming up...")
for q in queries[:2]:
    search(q, top_k=10)

# ---- Real benchmark: 5 runs per query, take median ----
print("\n" + "=" * 60)
print("Benchmark (median of 5 runs per query)")
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
print(f"\nOverall median latency: {overall_median:.0f} ms")
print(f"Overall throughput:     {1000/overall_median:.1f} queries/sec")

# ---- Profile a single query: where does the time actually go? ----
print("\n" + "=" * 60)
print("Profile breakdown (single query, averaged over 10 runs)")
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

avg_encode = t_encode / N * 1000
avg_matmul = t_matmul / N * 1000
avg_score  = t_score  / N * 1000
avg_total  = avg_encode + avg_matmul + avg_score

print(f"  Query: {profile_query!r}")
print(f"  Query encoding:     {avg_encode:6.1f} ms  ({avg_encode/avg_total*100:4.1f}%)")
print(f"  Matmul vs corpus:   {avg_matmul:6.1f} ms  ({avg_matmul/avg_total*100:4.1f}%)")
print(f"  Per-doc scoring:    {avg_score:6.1f} ms  ({avg_score/avg_total*100:4.1f}%)")
print(f"  Total:              {avg_total:6.1f} ms")