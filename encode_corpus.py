from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path

# ---- Setup ----
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Manually load the 768 -> 128 projection layer that AutoModel ignored
print("Loading projection layer...")
weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
state = load_file(weights_path)
linear = nn.Linear(768, 128, bias=False)
linear.weight.data = state["linear.weight"]
linear.eval()

# Device
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device)
linear = linear.to(device)
print(f"Using device: {device}")

# ---- Load corpus ----
N_DOCS = 30000 # was 10000 previously scaled it to 30k for better results on ivf
print(f"Loading {N_DOCS} passages from MS MARCO...")
ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True)

passages = []
seen = set()
for row in ds:
    for text in row["passages"]["passage_text"]:
        if text not in seen:            # Avoid duplicate passages
            seen.add(text)
            passages.append(text)
            if len(passages) >= N_DOCS:
                break
    if len(passages) >= N_DOCS:
        break

print(f"Got {len(passages)} unique passages.")

# ---- Encoder ----
def encode(text, max_len=180, is_query=False):
    # Prepend [Q] or [D] marker (real ColBERT does this)
    marker = "[Q] " if is_query else "[D] "
    inputs = tokenizer(
        marker + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]   # [num_tokens, 768]
        out = linear(out)                             # [num_tokens, 128] ← projection!
    out = out[1:-1]                                   # drop [CLS] and [SEP]
    out = F.normalize(out, dim=1)
    return out.cpu()

# ---- Encode all passages ----
print("Encoding...")
doc_vectors = []
for text in tqdm(passages):
    doc_vectors.append(encode(text, is_query=False))

# ---- Save ----
out_path = Path("data/corpus.pkl")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump({"texts": passages, "vectors": doc_vectors}, f)

# ---- Stats ----
total_tokens = sum(v.shape[0] for v in doc_vectors)
size_mb = out_path.stat().st_size / 1e6
print(f"\nDone.")
print(f"  Passages:     {len(passages)}")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Vector dim:   {doc_vectors[0].shape[1]}  (was 768)")
print(f"  File size:    {size_mb:.1f} MB  (was ~230 MB)")