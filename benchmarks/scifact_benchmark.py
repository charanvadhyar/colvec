"""
Run SciFact benchmark: brute force + IVF, compute nDCG@10 and Recall@100.
Compare to ColBERTv2 published baseline.
"""
import pickle
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ir_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# ---- Setup model (need it for query encoding) ----
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

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model = model.to(device)
linear = linear.to(device)

# ---- Load SciFact corpus ----
print("Loading SciFact corpus...")
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

doc_ids = corpus["doc_ids"]                    # list of SciFact doc_ids (strings)
doc_vectors = corpus["vectors"]
NUM_DOCS = len(doc_ids)

# Build token->doc lookup
all_doc_vectors = torch.cat(doc_vectors, dim=0)
doc_offsets = [0]
for v in doc_vectors:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc = torch.empty(all_doc_vectors.shape[0], dtype=torch.long)
for i in range(NUM_DOCS):
    token_to_doc[doc_offsets[i]:doc_offsets[i+1]] = i

print(f"Corpus: {NUM_DOCS} docs, {all_doc_vectors.shape[0]} tokens")

# ---- Query encoder ----
def encode_query(text, max_len=32):
    """
    Encode a query, padding with [MASK] tokens up to max_len.
    This is the query augmentation trick from the ColBERT paper.
    """
    # Step 1: tokenize WITHOUT padding to find natural query length
    inputs = tokenizer(
        "[Q] " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,                      # ← we'll pad manually below
    )
    
    input_ids = inputs["input_ids"][0]      # [num_tokens]
    attention_mask = inputs["attention_mask"][0]


    if not hasattr(encode_query, "_logged"):
        print(f"\n[DEBUG] encode_query: input has {len(input_ids)} tokens, "
              f"will pad to {max_len}, mask_token_id={tokenizer.mask_token_id}")
        encode_query._logged = True
    
    # Step 2: pad with [MASK] up to max_len
    # tokenizer.mask_token_id is the integer ID for [MASK] in BERT's vocab
    n_real = len(input_ids)
    n_pad = max_len - n_real
    if n_pad > 0:
        mask_padding = torch.full(
            (n_pad,), tokenizer.mask_token_id, dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids, mask_padding])
        # CRITICAL: attention_mask must be ALL 1s, including for [MASK] tokens.
        # ColBERT wants the model to attend to [MASK] tokens, not ignore them.
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(n_pad, dtype=attention_mask.dtype),
        ])
    
    # Step 3: forward pass + projection
    inputs = {
        "input_ids": input_ids.unsqueeze(0).to(device),
        "attention_mask": attention_mask.unsqueeze(0).to(device),
    }
    with torch.no_grad():
        out = model(**inputs).last_hidden_state[0]    # [max_len, 768]
        out = linear(out)                              # [max_len, 128]
    
    # Step 4: drop [CLS] (first) and [SEP] (somewhere in the middle now)
    # Find the original [SEP] position before padding
    sep_token_id = tokenizer.sep_token_id
    
    # Build a mask: keep everything EXCEPT [CLS] (position 0) and the original [SEP]
    keep_mask = torch.ones(max_len, dtype=torch.bool)
    keep_mask[0] = False                                # drop [CLS]
    
    # Find the original [SEP] in the unpadded portion
    for i in range(n_real):
        if input_ids[i] == sep_token_id:
            keep_mask[i] = False                        # drop [SEP]
            break
    
    out = out[keep_mask]                               # [num_kept, 128]
    return F.normalize(out, dim=1).cpu()
# ---- Brute force search (returns SciFact doc_ids) ----
def search_bruteforce(query_text, top_k=100):
    Q = encode_query(query_text)
    sim_all = Q @ all_doc_vectors.T
    num_q = sim_all.shape[0]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"), dtype=sim_all.dtype)
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc.unsqueeze(0).expand(num_q, -1),
        src=sim_all,
        reduce="amax",
        include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    # Map internal indices back to SciFact doc_ids
    return [(doc_ids[i], v) for i, v in zip(top_indices.tolist(), top_values.tolist())]

# ---- Metrics ----
def dcg_at_k(rel_scores, k):
    """Discounted Cumulative Gain at k."""
    rel_scores = rel_scores[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores))

def ndcg_at_k(retrieved_docs, relevant_docs, k):
    """nDCG@k using binary relevance."""
    rel_scores = [1 if d in relevant_docs else 0 for d in retrieved_docs[:k]]
    dcg = dcg_at_k(rel_scores, k)
    ideal_rel = sorted(rel_scores, reverse=True)
    # IDCG: best possible DCG given the number of relevant docs
    n_relevant = min(len(relevant_docs), k)
    ideal_rel_full = [1] * n_relevant + [0] * (k - n_relevant)
    idcg = dcg_at_k(ideal_rel_full, k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved_docs, relevant_docs, k):
    """Recall@k: fraction of relevant docs in top k retrieved."""
    if not relevant_docs:
        return 0.0
    retrieved_set = set(retrieved_docs[:k])
    return len(retrieved_set & relevant_docs) / len(relevant_docs)

# ---- Load qrels ----
print("Loading queries and qrels...")
dataset = ir_datasets.load("beir/scifact/test")
queries = list(dataset.queries_iter())
qrels_per_query = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels_per_query[qrel.query_id].add(qrel.doc_id)

# Only evaluate on queries that have at least one judged relevant doc
queries = [q for q in queries if q.query_id in qrels_per_query]
print(f"Evaluating on {len(queries)} queries with relevance judgments")

# ---- Run brute force on all queries ----
print("\nRunning brute force on all queries...")
ndcgs = []
recalls = []
latencies = []

for q in tqdm(queries):
    relevant_docs = qrels_per_query[q.query_id]

    t0 = time.perf_counter()
    results = search_bruteforce(q.text, top_k=100)
    latencies.append((time.perf_counter() - t0) * 1000)

    retrieved_doc_ids = [d for d, _ in results]
    ndcgs.append(ndcg_at_k(retrieved_doc_ids, relevant_docs, 10))
    recalls.append(recall_at_k(retrieved_doc_ids, relevant_docs, 100))

# ---- Report ----
import statistics
mean_ndcg = statistics.mean(ndcgs)
mean_recall = statistics.mean(recalls)
median_lat = statistics.median(latencies)

print("\n" + "=" * 60)
print("SciFact results — brute force MaxSim")
print("=" * 60)
print(f"  Queries evaluated:  {len(queries)}")
print(f"  nDCG@10:            {mean_ndcg:.4f}")
print(f"  Recall@100:         {mean_recall:.4f}")
print(f"  Median latency:     {median_lat:.1f} ms")
print()
print(f"  ColBERTv2 paper:    nDCG@10 = 0.693, Recall@100 = 0.971")
print(f"  Your number:        nDCG@10 = {mean_ndcg:.3f}, Recall@100 = {mean_recall:.3f}")
print()

# ---- Persist results ----
result_record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "system": "bruteforce_maxsim",
    "dataset": "scifact",
    "num_queries": len(queries),
    "ndcg_at_10": mean_ndcg,
    "recall_at_100": mean_recall,
    "median_latency_ms": median_lat,
}

Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
with open("benchmarks/results/scifact_history.jsonl", "a") as f:
    f.write(json.dumps(result_record) + "\n")

print(f"Saved to benchmarks/results/scifact_history.jsonl")

test_q = "vitamin D deficiency"
Q = encode_query(test_q)
print(f"Query: {test_q!r}")
print(f"Output shape: {Q.shape}")          # Should be ~[30, 128] (32 - [CLS] - [SEP])
print(f"First vector norm: {torch.norm(Q[0]):.4f}") 