import pickle
import torch

with open("data/corpus.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]
vectors = data["vectors"]

print(f"Loaded {len(texts)} passages, {len(vectors)} vector matrices.")
print(f"\nFirst passage:")
print(f"  Text:  {texts[0][:120]}...")
print(f"  Shape: {vectors[0].shape}")
print(f"  Norm of first token: {torch.norm(vectors[0][0]).item():.4f}")  # should be ~1.0