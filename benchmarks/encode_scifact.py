"""Encode the SciFact corpus with ColBERTv2."""
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import ir_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ---- Setup model + projection (same as encode_corpus.py) ----
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
print(f"Using device: {device}")

# ---- Encoder (same shape as encode_corpus.py, with [D] marker) ----
def encode(text, max_len=180, is_query=False):
    marker = "[Q] " if is_query else "[D] "
    inputs = tokenizer(
        marker + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]
        out = linear(out)
    out = out[1:-1]
    return F.normalize(out, dim=1).cpu()

# ---- Load and encode SciFact corpus ----
print("\nLoading SciFact dataset...")
dataset = ir_datasets.load("beir/scifact/test")

docs = list(dataset.docs_iter())
print(f"Corpus: {len(docs)} documents")

# SciFact docs have title + text — concat them like real ColBERT does
def doc_to_text(doc):
    title = getattr(doc, "title", "") or ""
    text = getattr(doc, "text", "") or ""
    return f"{title}. {text}".strip()

print("\nEncoding...")
doc_vectors = []
doc_ids = []
for doc in tqdm(docs):
    doc_ids.append(doc.doc_id)
    doc_vectors.append(encode(doc_to_text(doc), is_query=False))

# ---- Save ----
out_path = Path("data/scifact_corpus.pkl")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump({
        "doc_ids": doc_ids,                    # SciFact doc_ids (strings)
        "vectors": doc_vectors,                # token vectors per doc
        "texts": [doc_to_text(d) for d in docs],
    }, f)

# ---- Stats ----
total_tokens = sum(v.shape[0] for v in doc_vectors)
size_mb = out_path.stat().st_size / 1e6
print(f"\nDone.")
print(f"  Documents:    {len(doc_ids)}")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Vector dim:   {doc_vectors[0].shape[1]}")
print(f"  File size:    {size_mb:.1f} MB")