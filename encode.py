from transformers import AutoTokenizer, AutoModel
import torch

# Load the ColBERTv2 model and tokenizer
model_name = "colbert-ir/colbertv2.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # inference mode, no gradient tracking

# Encode one sentence
sentence = "the cat chase the mouse"
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Get the token-level embeddings
embeddings = outputs.last_hidden_state  # shape: [batch, num_tokens, hidden_dim]

print(f"Sentence: {sentence}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
print(f"Embedding shape: {embeddings.shape}")
print(f"First token's first 5 dims: {embeddings[0, 0, :5]}")


def maxsim(query_text, doc_text):
    q_inputs = tokenizer(query_text, return_tensors="pt")
    d_inputs = tokenizer(doc_text, return_tensors="pt")
    
    with torch.no_grad():
        Q = model(**q_inputs).last_hidden_state[0]   # [num_q_tokens, 768]
        D = model(**d_inputs).last_hidden_state[0]   # [num_d_tokens, 768]
    
    # Normalize so dot product = cosine similarity
    Q = torch.nn.functional.normalize(Q, dim=1)
    D = torch.nn.functional.normalize(D, dim=1)
    
    # The whole MaxSim algorithm in 3 lines
    sim = Q @ D.T                       # [num_q_tokens, num_d_tokens]
    maxes = sim.max(dim=1).values       # [num_q_tokens]
    return maxes.sum().item()

print(f"cat vs dog: {maxsim('cat', 'dog'):.3f}")
print(f"cat vs the dog chased the cat: {maxsim('cat', 'the dog chased the cat'):.3f}")
print(f"cat vs the dog ran fast: {maxsim('cat', 'the dog ran fast'):.3f}")