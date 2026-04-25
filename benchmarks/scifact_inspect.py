"""Quick exploration of the SciFact dataset."""
import ir_datasets

dataset = ir_datasets.load("beir/scifact/test")

# Corpus stats
docs = list(dataset.docs_iter())
print(f"Corpus: {len(docs)} documents")
print(f"Sample doc: {docs[0]}")

# Queries
queries = list(dataset.queries_iter())
print(f"\nQueries: {len(queries)} total")
print(f"Sample query: {queries[0]}")

# Relevance judgments (qrels)
qrels = list(dataset.qrels_iter())
print(f"\nQrels: {len(qrels)} total")
print(f"Sample qrel: {qrels[0]}")

# Per-query qrel count
from collections import defaultdict
qrels_per_query = defaultdict(list)
for qrel in qrels:
    qrels_per_query[qrel.query_id].append(qrel.doc_id)

avg_relevant = sum(len(v) for v in qrels_per_query.values()) / len(qrels_per_query)
print(f"\nAvg relevant docs per query: {avg_relevant:.1f}")
print(f"Queries with at least 1 relevant: {len(qrels_per_query)}")