# Author: Bradley R. Kinnard
"""Local embedding wrapper using sentence-transformers. Fully offline.
Model runs on CPU to leave GPU VRAM free for ollama."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

# pin torch to single thread -- prevents OpenMP double-free
# when sharing process with FAISS + PyQt6
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

logger = logging.getLogger(__name__)

# lazy-loaded singleton on CPU -- no VRAM pressure
_model = None
_model_name: str = "all-MiniLM-L6-v2"
_encode_lock = threading.Lock()


def _get_model():
    """lazy-load the sentence-transformer model on CPU."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading embedding model: %s (cpu)", _model_name)
        _model = SentenceTransformer(_model_name, device="cpu")
        logger.info("embedding model loaded on cpu")
    return _model


def set_model(name: str) -> None:
    """swap the embedding model (resets the singleton)."""
    global _model, _model_name
    _model = None
    _model_name = name
    logger.info("embedding model set to: %s", name)


def embed_text(text: str) -> np.ndarray:
    """embed a single string, returns a 1-d float32 vector. thread-safe."""
    if not text.strip():
        raise ValueError("cannot embed empty text")
    model = _get_model()
    with _encode_lock:
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    # owned contiguous copy -- critical for FAISS safety
    return np.array(vec, dtype=np.float32, copy=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """embed a list of strings, returns (n, dim) float32 array. thread-safe."""
    if not texts:
        raise ValueError("cannot embed empty batch")
    cleaned = [t for t in texts if t.strip()]
    if len(cleaned) != len(texts):
        raise ValueError("batch contains empty strings")
    model = _get_model()
    with _encode_lock:
        vecs = model.encode(cleaned, convert_to_numpy=True, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32, copy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cosine similarity between two normalized vectors."""
    # vectors are already L2-normalized from encode, so dot product suffices
    return float(np.dot(a, b))
