"""
Ingest SciFact single-vector embeddings into Qdrant.
Verify with a sample query.
"""
import pickle
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff, OptimizersConfigDiff,
)
from sentence_transformers import SentenceTransformer


# ---- Config ----
COLLECTION = "scifact_v1"
QDRANT_URL = "http://localhost:6333"
DATA_PATH = "data/scifact_singlevec.pkl"


# ---- Connect ----
print(f"Connecting to Qdrant at {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL)


# ---- Load encoded corpus ----
print(f"Loading {DATA_PATH}...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

doc_ids = data["doc_ids"]
texts = data["texts"]
embeddings = data["embeddings"]
dim = data["dim"]

print(f"  {len(doc_ids)} docs, dim={dim}")


# ---- Create collection (or recreate) ----
print(f"\nCreating collection '{COLLECTION}'...")
if client.collection_exists(COLLECTION):
    print(f"  Deleting existing collection")
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=dim,
        distance=Distance.COSINE,
    ),
    # Default HNSW params — good baseline
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100,
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=1000,
    ),
)
print("  Created.")


# ---- Ingest ----
# Qdrant requires integer or UUID IDs. SciFact doc_ids are strings, so we
# use the index as the Qdrant ID and store the original doc_id in payload.
print(f"\nIngesting {len(doc_ids)} points...")
t0 = time.perf_counter()

BATCH = 256
for start in range(0, len(doc_ids), BATCH):
    end = min(start + BATCH, len(doc_ids))
    batch_points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "scifact_id": doc_ids[i],
                "text_preview": texts[i][:100],
            },
        )
        for i in range(start, end)
    ]
    client.upsert(collection_name=COLLECTION, points=batch_points)
    print(f"  {end}/{len(doc_ids)}", end="\r")

ingest_time = time.perf_counter() - t0
print(f"\nIngested in {ingest_time:.1f}s ({len(doc_ids)/ingest_time:.0f} docs/sec)")


# ---- Wait for indexing to finish ----
# HNSW indexing happens in the background. Wait until it's done.
print("\nWaiting for indexing to complete...")
while True:
    info = client.get_collection(COLLECTION)
    if info.status == "green" and info.indexed_vectors_count > 0:
        break
    print(f"  status={info.status}, indexed={info.indexed_vectors_count}/{len(doc_ids)}",
          end="\r")
    time.sleep(1)
print(f"\n  Indexed {info.indexed_vectors_count} vectors")


# ---- Sanity check: a query ----
print("\n" + "=" * 60)
print("Sanity test: querying for a SciFact-style claim")
print("=" * 60)

model = SentenceTransformer(data["model"])
query_text = "0-dimensional biomaterials show inductive properties."
print(f"\nQuery: {query_text!r}")

q_vec = model.encode(query_text, normalize_embeddings=True)
results = client.query_points(
    collection_name=COLLECTION,
    query=q_vec.tolist(),
    limit=5,
).points

print("\nTop 5 results:")
for r in results:
    sf_id = r.payload["scifact_id"]
    preview = r.payload["text_preview"]
    print(f"  [{r.score:.4f}] doc_id={sf_id}: {preview[:70]}...")

print("\n[OK] Round-trip working. Day 19 done.")