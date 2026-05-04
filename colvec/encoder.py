"""
Shared query and document encoder for colvec.
Single source of truth — import this everywhere instead of
duplicating the model loading and encode_query function.

Usage:
    from colvec.encoder import encode_query, encode_doc, load_model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_NAME = "colbert-ir/colbertv2.0"
_tokenizer = None
_model     = None
_linear    = None
_device    = None


def load_model(device: str = "cpu"):
    """Load ColBERTv2 model + projection layer. Call once at startup."""
    global _tokenizer, _model, _linear, _device

    if _model is not None:
        return   # already loaded

    _device    = device
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model     = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)

    weights_path = hf_hub_download(
        repo_id=MODEL_NAME, filename="model.safetensors"
    )
    state   = load_file(weights_path)
    _linear = nn.Linear(768, 128, bias=False)
    _linear.weight.data = state["linear.weight"]
    _linear.eval().to(device)


def encode_query(
    text: str,
    max_len: int = 32,
    augment: bool = False,
) -> np.ndarray:
    """
    Encode a query string into ColBERT token vectors.

    Args:
        text:    Query string.
        max_len: Maximum token budget (32 is ColBERT default).
        augment: If True, pad short queries with [MASK] tokens up to
                 max_len. Helps on short queries (<15 tokens).
                 Neutral or slightly harmful on long queries.

    Returns:
        np.ndarray of shape [num_tokens, 128], L2-normalized.
    """
    assert _model is not None, "Call load_model() before encode_query()"

    if augment:
        return _encode_query_augmented(text, max_len)
    else:
        return _encode_query_plain(text, max_len)


def _encode_query_plain(text: str, max_len: int) -> np.ndarray:
    """Standard query encoding — no padding tricks."""
    inputs = _tokenizer(
        "[Q] " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(_device)

    with torch.no_grad():
        out = _model(**inputs).last_hidden_state[0]
        out = _linear(out)

    out = out[1:-1]   # drop [CLS] and [SEP]
    return F.normalize(out, dim=1).cpu().numpy()


def _encode_query_augmented(text: str, max_len: int) -> np.ndarray:
    """
    Query encoding with [MASK] padding up to max_len.

    The [MASK] tokens attend to real query tokens via BERT's attention
    and produce contextualized expansion vectors — automatic query
    expansion without explicit term generation.

    Key: attention_mask must be 1 for [MASK] tokens (not 0).
    We WANT the model to attend to them, not ignore them.
    """
    # Step 1: tokenize without padding to find natural length
    inputs = _tokenizer(
        "[Q] " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    input_ids      = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]
    n_real         = len(input_ids)

    # Step 2: pad with [MASK] tokens up to max_len
    n_pad = max_len - n_real
    if n_pad > 0:
        mask_id     = _tokenizer.mask_token_id
        mask_pad    = torch.full((n_pad,), mask_id, dtype=input_ids.dtype)
        ones_pad    = torch.ones(n_pad, dtype=attention_mask.dtype)
        input_ids      = torch.cat([input_ids, mask_pad])
        attention_mask = torch.cat([attention_mask, ones_pad])

    # Step 3: forward pass
    inputs_padded = {
        "input_ids":      input_ids.unsqueeze(0).to(_device),
        "attention_mask": attention_mask.unsqueeze(0).to(_device),
    }
    with torch.no_grad():
        out = _model(**inputs_padded).last_hidden_state[0]
        out = _linear(out)

    # Step 4: drop [CLS] and original [SEP], keep [MASK] vectors
    sep_id   = _tokenizer.sep_token_id
    keep     = torch.ones(max_len, dtype=torch.bool)
    keep[0]  = False   # drop [CLS]
    for i in range(n_real):
        if input_ids[i] == sep_id:
            keep[i] = False
            break

    out = out[keep]
    return F.normalize(out, dim=1).cpu().numpy()


def encode_doc(text: str, max_len: int = 180) -> np.ndarray:
    """
    Encode a document string into ColBERT token vectors.

    Args:
        text:    Document string (title + body concatenated).
        max_len: Maximum token budget per chunk.

    Returns:
        np.ndarray of shape [num_tokens, 128], L2-normalized.
    """
    assert _model is not None, "Call load_model() before encode_doc()"

    inputs = _tokenizer(
        "[D] " + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(_device)

    with torch.no_grad():
        out = _model(**inputs).last_hidden_state[0]
        out = _linear(out)

    out = out[1:-1]   # drop [CLS] and [SEP]
    return F.normalize(out, dim=1).cpu().numpy()