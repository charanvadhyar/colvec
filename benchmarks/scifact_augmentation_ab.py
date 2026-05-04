"""
A/B test: query augmentation on/off, same 300 SciFact queries.

Expected result on SciFact: neutral (queries already ~18 tokens).
This test establishes the baseline and confirms the flag works.
Will show bigger delta on short-query medical datasets (NFCorpus).
"""
import pickle
import statistics
import time
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import ir_datasets
from tqdm import tqdm

from colvec.encoder import load_model, encode_query

# ---- Load model ----
print("Loading model...")
load_model(device="cpu")

# ---- Load SciFact corpus ----
print("Loading SciFact corpus...")
with open("data/scifact_corpus.pkl", "rb") as f:
    corpus = pickle.load(f)

doc_ids          = corpus["doc_ids"]
doc_vectors_list = corpus["vectors"]
NUM_DOCS         = len(doc_ids)

all_doc_vectors = torch.cat(doc_vectors_list, dim=0)

doc_offsets = [0]
for v in doc_vectors_list:
    doc_offsets.append(doc_offsets[-1] + v.shape[0])

token_to_doc = torch.empty(all_doc_vectors.shape[0], dtype=torch.long)
for i in range(NUM_DOCS):
    token_to_doc[doc_offsets[i]:doc_offsets[i+1]] = i

# ---- Load qrels ----
print("Loading qrels...")
dataset = ir_datasets.load("beir/scifact/test")
queries = list(dataset.queries_iter())
qrels   = defaultdict(set)
for qrel in dataset.qrels_iter():
    if qrel.relevance > 0:
        qrels[qrel.query_id].add(qrel.doc_id)
queries = [q for q in queries if q.query_id in qrels]
print(f"  {len(queries)} queries")


# ---- Brute force search ----
def search(query_vec_np, top_k=100):
    Q   = torch.from_numpy(query_vec_np)
    sim = Q @ all_doc_vectors.T
    num_q = sim.shape[0]

    per_doc_max = torch.full((num_q, NUM_DOCS), float("-inf"))
    per_doc_max.scatter_reduce_(
        dim=1,
        index=token_to_doc.unsqueeze(0).expand(num_q, -1),
        src=sim, reduce="amax", include_self=True,
    )
    scores = per_doc_max.sum(dim=0)
    top_values, top_indices = scores.topk(top_k)
    return [(doc_ids[i], v.item())
            for i, v in zip(top_indices.tolist(), top_values)]


# ---- Metrics ----
def dcg_at_k(rel, k):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel[:k]))

def ndcg_at_k(retrieved, relevant, k):
    rel = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg = dcg_at_k(rel, k)
    n   = min(len(relevant), k)
    idcg = dcg_at_k([1]*n + [0]*(k-n), k)
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


# ---- Run one pass ----
def run_pass(label, augment):
    print(f"\nRunning: {label} (augment={augment})...")
    ndcgs, recalls, latencies, q_lengths = [], [], [], []

    for q in tqdm(queries, ncols=80):
        relevant = qrels[q.query_id]

        t0 = time.perf_counter()
        q_vec = encode_query(q.text, max_len=32, augment=augment)
        results = search(q_vec, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)

        q_lengths.append(q_vec.shape[0])
        retrieved = [d for d, _ in results]
        ndcgs.append(ndcg_at_k(retrieved, relevant, 10))
        recalls.append(recall_at_k(retrieved, relevant, 100))

    return {
        "label":        label,
        "augment":      augment,
        "ndcg_at_10":   statistics.mean(ndcgs),
        "recall_at_100": statistics.mean(recalls),
        "median_lat_ms": statistics.median(latencies),
        "avg_q_tokens": statistics.mean(q_lengths),
    }


# ---- Run both ----
r_no_aug = run_pass("No augmentation", augment=False)
r_aug    = run_pass("With augmentation", augment=True)

# ---- Report ----
print("\n" + "=" * 65)
print("A/B Test: query augmentation on SciFact")
print("=" * 65)
print(f"\n{'System':<25} | {'nDCG@10':>8} | {'Recall':>7} | "
      f"{'Median lat':>11} | {'Avg Q tokens':>13}")
print("-" * 65)
for r in [r_no_aug, r_aug]:
    print(f"{r['label']:<25} | {r['ndcg_at_10']:>8.4f} | "
          f"{r['recall_at_100']:>7.4f} | "
          f"{r['median_lat_ms']:>10.1f}ms | "
          f"{r['avg_q_tokens']:>12.1f}")
print("-" * 65)

delta_ndcg   = r_aug["ndcg_at_10"]    - r_no_aug["ndcg_at_10"]
delta_recall = r_aug["recall_at_100"] - r_no_aug["recall_at_100"]
print(f"\nDelta nDCG@10:    {delta_ndcg:+.4f}")
print(f"Delta Recall@100: {delta_recall:+.4f}")
print(f"Avg query tokens: {r_no_aug['avg_q_tokens']:.1f} → "
      f"{r_aug['avg_q_tokens']:.1f} (with augmentation)")
print()

if abs(delta_ndcg) < 0.005:
    print("Result: NEUTRAL — augmentation has no significant effect on SciFact.")
    print("Expected — SciFact queries are long scientific claims (~18 tokens).")
    print("Augmentation will be tested again on short-query medical datasets.")
elif delta_ndcg > 0.005:
    print(f"Result: POSITIVE — augmentation helps by {delta_ndcg:+.4f} nDCG.")
else:
    print(f"Result: NEGATIVE — augmentation hurts by {delta_ndcg:+.4f} nDCG.")
    print("Check implementation — unexpected direction.")