from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time

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
        start, end = doc_offsets[i], doc_offsets[i+1]
        sim = sim_all[:, start:end]
        score = sim.max(dim=1).values.sum().item()
        scores.append(score)
    
    return sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]

# ---- Run queries ----
queries = [
    "history of the manhattan project",
    "vitamin D deficiency symptoms",
    "weather forecast",
    "what causes thunderstorms",
]

print("\n" + "="*60)
times = []
for query in queries:
    start = time.perf_counter()
    results = search(query, top_k=5)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    
    print(f"\nQuery: {query!r}  ({elapsed*1000:.0f} ms)")
    for rank, (doc_id, score) in enumerate(results, 1):
        snippet = texts[doc_id][:120].replace("\n", " ")
        print(f"  {rank}. [{score:5.2f}] {snippet}...")

avg_ms = sum(times) / len(times) * 1000
print(f"\nAverage latency: {avg_ms:.0f} ms / query")