from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path

# ---- Setup ----
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Use GPU/MPS if available
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device)
print(f"Using device: {device}")

# ---- Load corpus ----
N_DOCS = 1000
print(f"Loading {N_DOCS} passages from MS MARCO...")
ds = load_dataset("ms_marco", "v2.1", split="train", streaming=True)

# MS MARCO has nested passages per query — flatten to unique passage texts
passages = []
seen = set()
for row in ds:
    for text in row["passages"]["passage_text"]:
        if text not in seen:
            seen.add(text)
            passages.append(text)
            if len(passages) >= N_DOCS:
                break
    if len(passages) >= N_DOCS:
        break

print(f"Got {len(passages)} unique passages.")

# ---- Encode each passage ----
def encode(text, max_len=180):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]   # [num_tokens, 768]
    out = F.normalize(out, dim=1)                    # unit vectors
    return out.cpu()                                 # back to CPU for storage

print("Encoding...")
doc_vectors = []   # list of [num_tokens_i, 768] tensors
for text in tqdm(passages):
    doc_vectors.append(encode(text))

# ---- Save to disk ----
out_path = Path("data/corpus.pkl")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump({"texts": passages, "vectors": doc_vectors}, f)

# ---- Stats ----
total_tokens = sum(v.shape[0] for v in doc_vectors)
avg_tokens = total_tokens / len(doc_vectors)
size_mb = out_path.stat().st_size / 1e6

print(f"\nDone.")
print(f"  Passages:     {len(passages)}")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Avg tokens:   {avg_tokens:.1f}")
print(f"  File size:    {size_mb:.1f} MB")