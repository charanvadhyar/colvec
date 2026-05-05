"""
colvec/encoder.py — Shared query and document encoder.

Single source of truth for model loading and encoding.
Import this everywhere instead of duplicating model loading code.

Usage:
    from colvec.encoder import load_model, encode_query, encode_doc
    load_model()   # call once at startup
    vec = encode_query("what causes thunderstorms")
    vec = encode_doc("The Manhattan Project was...")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_NAME = "colbert-ir/colbertv2.0"

# Module-level singletons — loaded once, reused everywhere
_tokenizer = None
_model     = None
_linear    = None
_device    = None


def load_model(device: str = "cpu") -> None:
    """
    Load ColBERTv2 model + projection layer.
    Call once at startup. Safe to call multiple times (no-op after first).
    """
    global _tokenizer, _model, _linear, _device

    if _model is not None:
        return   # already loaded

    _device    = device
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model     = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)

    weights_path        = hf_hub_download(repo_id=MODEL_NAME, filename="model.safetensors")
    state               = load_file(weights_path)
    _linear             = nn.Linear(768, 128, bias=False)
    _linear.weight.data = state["linear.weight"]
    _linear.eval().to(device)

    print(f"  Model loaded: {MODEL_NAME} on {device}")


def encode_query(
    text:    str,
    max_len: int  = 32,
    augment: bool = False,
) -> np.ndarray:
    """
    Encode a query string into ColBERT token vectors.

    Args:
        text:    Query string.
        max_len: Token budget (ColBERT default = 32).
        augment: Pad short queries with [MASK] tokens up to max_len.
                 Helps on short queries (<15 tokens).
                 Neutral or slightly harmful on long queries (SciFact).
                 Enable for short-query medical datasets (NFCorpus).

    Returns:
        np.ndarray [num_tokens, 128], L2-normalized.
    """
    assert _model is not None, "Call load_model() before encode_query()"

    if augment:
        return _encode_query_augmented(text, max_len)
    return _encode_query_plain(text, max_len)


def encode_doc(
    text:    str,
    max_len: int = 180,
) -> np.ndarray:
    """
    Encode a document string into ColBERT token vectors (single chunk).
    For long documents, use encode_doc_chunks() instead.

    Args:
        text:    Document string (title + body concatenated).
        max_len: Maximum token budget. Text beyond this is truncated.

    Returns:
        np.ndarray [num_tokens, 128], L2-normalized.
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


def encode_doc_chunks(
    text:          str,
    chunk_size:    int = 500,
    chunk_overlap: int = 100,
) -> list:
    """
    Encode a document as overlapping chunks.

    Uses chunk_size=300 by default so most documents (avg ~175 tokens)
    fit in one chunk. Only genuinely long documents get split.

    Args:
        text:          Document string (title + body).
        chunk_size:    Content tokens per chunk (default 300).
                       chunk_size + 2 must be <= 512 (BERT limit).
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        List of np.ndarray, one per chunk, each [num_tokens, 128] L2-normalized.
    """
    assert _model is not None, "Call load_model() before encode_doc_chunks()"

    cls_id = _tokenizer.cls_token_id
    sep_id = _tokenizer.sep_token_id

    # Tokenize the content WITHOUT special tokens and WITHOUT [D] prefix.
    # We add [CLS], [D] marker, and [SEP] manually per chunk.
    # This gives us clean content token IDs to slide the window over.
    d_marker_ids = _tokenizer(
        "[D]", add_special_tokens=False
    )["input_ids"]   # typically [1041] or similar — the [D] token

    content_ids = _tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=509 - len(d_marker_ids),   # leave room for [CLS], [D], [SEP]
    )["input_ids"]

    # If it fits in one chunk, use the standard encode_doc path
    if len(content_ids) <= chunk_size - len(d_marker_ids):
        return [encode_doc(text, max_len=chunk_size + 2)]

    # Slide window over content_ids
    step   = chunk_size - chunk_overlap
    chunks = []
    start  = 0

    while start < len(content_ids):
        end       = min(start + chunk_size - len(d_marker_ids), len(content_ids))
        chunk_ids = content_ids[start:end]

        # Build: [CLS] [D] <content tokens> [SEP]
        input_ids = torch.tensor(
            [cls_id] + d_marker_ids + chunk_ids + [sep_id]
        ).unsqueeze(0).to(_device)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = _model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[0]
            out = _linear(out)

        # Drop [CLS] and [SEP], keep [D] marker and content vectors
        # [D] marker is at position 1, [SEP] is at the end
        out = out[1:-1]
        chunks.append(F.normalize(out, dim=1).cpu().numpy())

        if end == len(content_ids):
            break
        start += step

    return chunks

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

    [MASK] tokens attend to real query tokens via BERT attention and
    produce contextualized expansion vectors — automatic query expansion.

    Critical: attention_mask must be 1 for [MASK] tokens.
    We WANT the model to attend to them, not ignore them.
    """
    # Tokenize without padding to find natural length
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

    # Pad with [MASK] tokens up to max_len
    n_pad = max_len - n_real
    if n_pad > 0:
        mask_id        = _tokenizer.mask_token_id
        input_ids      = torch.cat([input_ids,      torch.full((n_pad,), mask_id, dtype=input_ids.dtype)])
        attention_mask = torch.cat([attention_mask,  torch.ones(n_pad,  dtype=attention_mask.dtype)])

    # Forward pass
    with torch.no_grad():
        out = _model(
            input_ids=input_ids.unsqueeze(0).to(_device),
            attention_mask=attention_mask.unsqueeze(0).to(_device),
        ).last_hidden_state[0]
        out = _linear(out)

    # Drop [CLS] and original [SEP], keep [MASK] vectors
    sep_id  = _tokenizer.sep_token_id
    keep    = torch.ones(max_len, dtype=torch.bool)
    keep[0] = False   # drop [CLS]
    for i in range(n_real):
        if input_ids[i] == sep_id:
            keep[i] = False
            break

    out = out[keep]
    return F.normalize(out, dim=1).cpu().numpy()