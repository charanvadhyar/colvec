"""
Encode the SciFact corpus with a sentence-transformer (one vector per doc).
This is a different encoding from your ColBERT pipeline — necessary for
single-vector comparison with Qdrant.

Model: all-MiniLM-L6-v2 (384 dims, fast, well-known baseline).
Run this once. Output saved to data/scifact_singlevec.pkl.
"""
import pickle
from pathlib import Path

import ir_datasets
from sentence_transformers import SentenceTransformer

# ---- Config ----
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_PATH   = Path("data/scifact_singlevec.pkl")

# ---- Load model ----
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"  Embedding dim: {embedding_dim}")

# ---- Load SciFact ----
print("\nLoading SciFact corpus...")
dataset = ir_datasets.load("beir/scifact/test")
docs    = list(dataset.docs_iter())
print(f"  {len(docs)} documents")

def doc_to_text(doc):
    title = getattr(doc, "title", "") or ""
    text  = getattr(doc, "text",  "") or ""
    return f"{title}. {text}".strip()

# ---- Encode ----
print("\nEncoding corpus...")
texts   = [doc_to_text(d) for d in docs]
doc_ids = [d.doc_id for d in docs]

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,    # cosine similarity = dot product
    convert_to_numpy=True,
)

print(f"\n  Embeddings shape: {embeddings.shape}")
print(f"  Storage: {embeddings.nbytes / 1e6:.1f} MB")

# ---- Save ----
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump({
        "doc_ids":    doc_ids,
        "texts":      texts,
        "embeddings": embeddings,
        "model":      MODEL_NAME,
        "dim":        embedding_dim,
    }, f)

print(f"\nSaved to {OUT_PATH}")