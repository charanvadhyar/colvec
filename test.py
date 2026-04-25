import pickle
with open("data/corpus.pkl", "rb") as f:
    data = pickle.load(f)

# Check vector dimensions
print(f"First vector shape: {data['vectors'][0].shape}")
print(f"Second vector shape: {data['vectors'][1].shape}")

# Distribution of token counts
import statistics
token_counts = [v.shape[0] for v in data['vectors']]
print(f"Min tokens: {min(token_counts)}")
print(f"Max tokens: {max(token_counts)}")
print(f"Median tokens: {statistics.median(token_counts)}")
print(f"Mean tokens: {statistics.mean(token_counts):.1f}")