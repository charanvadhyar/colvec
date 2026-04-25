import pickle

with open("data/corpus.pkl", "rb") as f:
    data = pickle.load(f)

texts = data["texts"]

# Check several topics
keywords = ["thunder", "storm", "lightning", "weather",
            "manhattan", "vaccine", "vitamin"]

print(f"Corpus size: {len(texts)} passages\n")
for kw in keywords:
    matches = [(i, t) for i, t in enumerate(texts) if kw.lower() in t.lower()]
    print(f"  '{kw}': {len(matches)} passages")
    if matches:
        print(f"     example: [{matches[0][0]}] {matches[0][1][:120]}...")