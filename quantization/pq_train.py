"""
Train a product quantizer's codebooks from a sample of vectors.
This is offline — runs once, saves codebooks to disk.
"""
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans


def train_pq(
    vectors: np.ndarray,
    M: int = 16,
    K: int = 256,
    sample_size: int = 100_000,
    seed: int = 42,
):
    """
    Train product quantizer codebooks.
    
    Args:
        vectors: [N, D] array of training vectors (e.g., your token vectors).
        M: number of subquantizers (chunks). D must be divisible by M.
        K: number of centroids per chunk. Typically 256 (one byte per code).
        sample_size: how many vectors to use for training. More = better
                     codebooks, but slower training. 100K is plenty.
    
    Returns:
        codebooks: [M, K, D/M] array — M codebooks, each with K centroids
                   of dimension D/M.
    """
    N, D = vectors.shape
    assert D % M == 0, f"D={D} must be divisible by M={M}"
    chunk_dim = D // M
    
    # ---- Subsample for training ----
    rng = np.random.default_rng(seed)
    if N > sample_size:
        idx = rng.choice(N, size=sample_size, replace=False)
        train_data = vectors[idx]
    else:
        train_data = vectors
    print(f"Training PQ on {len(train_data)} vectors. M={M}, K={K}, chunk_dim={chunk_dim}")
    
    # ---- Train one k-means PER CHUNK ----
    # Each chunk gets its own independent codebook.
    # The "product" in product quantization comes from the M codebooks
    # combining as a Cartesian product.
    codebooks = np.zeros((M, K, chunk_dim), dtype=np.float32)
    
    for m in range(M):
        # Slice out chunk m from every training vector
        chunk = train_data[:, m * chunk_dim : (m + 1) * chunk_dim]   # [N_train, chunk_dim]
        
        # K-means with K centroids in chunk_dim dimensions
        km = MiniBatchKMeans(
            n_clusters=K,
            batch_size=4096,
            n_init=3,
            max_iter=100,
            random_state=seed + m,   # different seed per chunk
            verbose=0,
        )
        km.fit(chunk)
        codebooks[m] = km.cluster_centers_
        
        # Quick quality check: avg distance from training points to nearest centroid
        labels = km.labels_
        avg_dist = np.linalg.norm(chunk - codebooks[m, labels], axis=1).mean()
        print(f"  chunk {m:2d}: trained, avg quantization error = {avg_dist:.4f}")
    
    return codebooks


def encode_with_pq(vectors: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """
    Compress vectors using trained codebooks.
    
    Args:
        vectors:   [N, D] vectors to compress.
        codebooks: [M, K, D/M] trained codebooks.
    
    Returns:
        codes: [N, M] uint8 array — each vector represented as M bytes.
    """
    N, D = vectors.shape
    M, K, chunk_dim = codebooks.shape
    assert D == M * chunk_dim, f"D={D} doesn't match M*chunk_dim={M * chunk_dim}"
    assert K <= 256, f"K={K} > 256 doesn't fit in uint8"
    
    codes = np.zeros((N, M), dtype=np.uint8)
    
    for m in range(M):
        chunk = vectors[:, m * chunk_dim : (m + 1) * chunk_dim]   # [N, chunk_dim]
        
        # Distance from each vector's chunk to every codebook entry
        # Broadcasting: chunk [N, 1, chunk_dim] - codebook [1, K, chunk_dim]
        # → diffs [N, K, chunk_dim]
        diffs = chunk[:, None, :] - codebooks[m][None, :, :]
        dists = np.linalg.norm(diffs, axis=2)   # [N, K]
        
        codes[:, m] = dists.argmin(axis=1)
    
    return codes


# ---- Run training on SciFact corpus ----
if __name__ == "__main__":
    print("Loading SciFact corpus...")
    with open("data/scifact_corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    
    # Stack all token vectors into one big array
    all_vectors = torch.cat(corpus["vectors"], dim=0).numpy()
    print(f"Total tokens: {len(all_vectors)}, dim: {all_vectors.shape[1]}")
    
    # Train codebooks
    codebooks = train_pq(all_vectors, M=16, K=256)
    
    # Encode the whole corpus with the trained codebooks
    print(f"\nEncoding {len(all_vectors)} vectors with PQ...")
    codes = encode_with_pq(all_vectors, codebooks)
    
    # ---- Save ----
    out_path = Path("data/scifact_pq.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "codebooks": codebooks,           # [M, K, D/M] float32
            "codes": codes,                   # [N, M] uint8
            "M": 16,
            "K": 256,
            "D": 128,
        }, f)
    
    # ---- Stats ----
    original_size = all_vectors.nbytes / 1e6
    codes_size = codes.nbytes / 1e6
    codebook_size = codebooks.nbytes / 1e6
    compression_ratio = all_vectors.nbytes / codes.nbytes
    
    print(f"\nDone.")
    print(f"  Original vectors:   {original_size:.1f} MB")
    print(f"  Compressed codes:   {codes_size:.1f} MB")
    print(f"  Codebook overhead:  {codebook_size:.2f} MB")
    print(f"  Compression ratio:  {compression_ratio:.1f}x")
    print(f"  Saved to: {out_path}")