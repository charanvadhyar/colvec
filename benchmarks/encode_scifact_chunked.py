"""
Encode SciFact corpus with document chunking.

Instead of truncating each document at 180 tokens, split long documents
into overlapping chunks of 150 tokens with 50-token overlap.

Each chunk is indexed as a separate unit. At retrieval time:
  doc_score = max over chunks of chunk_score

For SciFact (short abstracts ~175 tokens): most docs fit in one chunk.
For longer medical documents (NFCorpus, TREC-COVID): critical for recall.

Output: data/scifact_corpus_chunked.pkl
"""
import pickle
import statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import ir_datasets
from tqdm import tqdm

from colvec.encoder import load_model, encode_doc_chunks

# ---- Config ----
CHUNK_SIZE    = 500 
CHUNK_OVERLAP = 100
OUT_PATH      = Path("data/scifact_corpus_chunked.pkl")

# ---- Load model ----
print("Loading model...")
load_model(device="cpu")

# ---- Load SciFact ----
print("\nLoading SciFact corpus...")
dataset = ir_datasets.load("beir/scifact/test")
docs    = list(dataset.docs_iter())
print(f"  {len(docs)} documents")

def doc_to_text(doc):
    title = getattr(doc, "title", "") or ""
    text  = getattr(doc, "text",  "") or ""
    return f"{title}. {text}".strip()

# ---- Encode with chunking ----
print(f"\nEncoding with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")

doc_ids       = []
all_chunk_vecs = []    # flat list of chunk vectors [n_tok, 128] each
chunk_counts  = []     # number of chunks per doc
token_to_doc  = []     # which doc each token belongs to
token_to_chunk = []    # which chunk (within doc) each token belongs to
texts         = []

for doc in tqdm(docs, ncols=80):
    text   = doc_to_text(doc)
    chunks = encode_doc_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    doc_idx = len(doc_ids)
    doc_ids.append(doc.doc_id)
    texts.append(text)
    chunk_counts.append(len(chunks))

    for chunk_idx, chunk_vec in enumerate(chunks):
        all_chunk_vecs.append(chunk_vec)
        n_tok = chunk_vec.shape[0]
        token_to_doc.extend([doc_idx] * n_tok)
        token_to_chunk.extend([chunk_idx] * n_tok)

# ---- Stats ----
total_tokens  = sum(v.shape[0] for v in all_chunk_vecs)
total_chunks  = len(all_chunk_vecs)
multi_chunked = sum(1 for c in chunk_counts if c > 1)

print(f"\nEncoding complete:")
print(f"  Documents:          {len(doc_ids)}")
print(f"  Total chunks:       {total_chunks}")
print(f"  Total tokens:       {total_tokens:,}")
print(f"  Chunks/doc:         min={min(chunk_counts)}, "
      f"median={statistics.median(chunk_counts):.1f}, "
      f"max={max(chunk_counts)}")
print(f"  Docs with >1 chunk: {multi_chunked} "
      f"({100*multi_chunked/len(doc_ids):.1f}%)")

# ---- Save ----
token_to_doc_arr    = np.array(token_to_doc,   dtype=np.int32)
token_to_chunk_arr  = np.array(token_to_chunk, dtype=np.int32)

OUT_PATH.parent.mkdir(exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump({
        "doc_ids":         doc_ids,
        "texts":           texts,
        "vectors":         all_chunk_vecs,       # list of [n_tok, 128] arrays
        "chunk_counts":    chunk_counts,          # chunks per doc
        "token_to_doc":    token_to_doc_arr,
        "token_to_chunk":  token_to_chunk_arr,
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
    }, f)

size_mb = OUT_PATH.stat().st_size / 1e6
print(f"\nSaved to {OUT_PATH}  ({size_mb:.1f} MB)")