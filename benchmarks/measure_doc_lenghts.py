# benchmarks/measure_doc_lengths.py
"""
Measure document token length distribution for any BEIR dataset.
Run this before encoding to choose the right chunk_size.

Usage:
    uv run python benchmarks/measure_doc_lengths.py beir/scifact/test
    uv run python benchmarks/measure_doc_lengths.py beir/nfcorpus/test
    uv run python benchmarks/measure_doc_lengths.py beir/trec-covid/test
"""
import sys
import statistics
import ir_datasets
from transformers import AutoTokenizer
from tqdm import tqdm

DATASET = sys.argv[1] if len(sys.argv) > 1 else "beir/scifact/test"
MODEL   = "colbert-ir/colbertv2.0"

print(f"Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)

print(f"Loading {DATASET}...")
dataset = ir_datasets.load(DATASET)
docs    = list(dataset.docs_iter())
print(f"  {len(docs)} documents")

def doc_to_text(doc):
    title = getattr(doc, "title", "") or ""
    text  = getattr(doc, "text",  "") or ""
    return f"{title}. {text}".strip()

print("Measuring token lengths...")
lengths = []
for doc in tqdm(docs, ncols=80):
    ids = tok(doc_to_text(doc), add_special_tokens=False)["input_ids"]
    lengths.append(len(ids))

lengths.sort()
N = len(lengths)

print(f"\n{'='*50}")
print(f"Dataset: {DATASET}")
print(f"{'='*50}")
print(f"  Documents: {N}")
print(f"  min:       {min(lengths)}")
print(f"  median:    {statistics.median(lengths):.0f}")
print(f"  mean:      {statistics.mean(lengths):.0f}")
print(f"  p75:       {lengths[int(N*0.75)]}")
print(f"  p90:       {lengths[int(N*0.90)]}")
print(f"  p95:       {lengths[int(N*0.95)]}")
print(f"  p99:       {lengths[int(N*0.99)]}")
print(f"  max:       {max(lengths)}")
print()

# Recommend chunk_size
p90 = lengths[int(N*0.90)]
recommended = min(p90 + 50, 500)   # cover p90 + buffer, cap at 500
overlap     = min(recommended // 5, 100)

print(f"Recommendation:")
print(f"  chunk_size    = {recommended}  (covers p90={p90} with buffer)")
print(f"  chunk_overlap = {overlap}")
print(f"  BERT limit check: {recommended} + 3 = {recommended+3} "
      f"{'✓ safe' if recommended+3 <= 512 else '✗ TOO LARGE — reduce chunk_size'}")