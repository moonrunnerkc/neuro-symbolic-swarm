# Author: Bradley R. Kinnard
"""FAISS + JSON shared memory layer. Atomic writes, offline persistence."""

from __future__ import annotations

import base64
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# pin torch intraop threads BEFORE any faiss/torch interaction
# prevents OpenMP double-free when both libs share the C++ runtime
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

logger = logging.getLogger(__name__)

# conditional faiss import -- fall back to cpu if gpu unavailable
try:
    import faiss
    _FAISS_GPU = hasattr(faiss, "index_cpu_to_all_gpus")
except ImportError:
    faiss = None
    _FAISS_GPU = False


@dataclass
class MemoryEntry:
    """a single stored embedding with metadata."""
    key: str
    text: str
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class SharedMemory:
    """persistent vector store backed by FAISS + JSON sidecar.

    thread-safe through a reentrant lock. atomic writes prevent
    corruption during multi-process access.
    """

    # cap total entries to avoid unbounded memory growth
    MAX_ENTRIES = 200

    def __init__(
        self,
        index_path: Path,
        meta_path: Optional[Path] = None,
        dimension: int = 384,
        use_gpu: bool = False,
    ):
        self._index_path = Path(index_path)
        self._meta_path = meta_path or self._index_path.with_suffix(".json")
        self._dimension = dimension
        self._lock = threading.RLock()
        self._entries: dict[str, MemoryEntry] = {}
        self._keys_order: list[str] = []

        if faiss is None:
            raise ImportError("faiss is required -- install faiss-cpu or faiss-gpu")

        # clamp FAISS internal threads to avoid OpenMP conflicts
        faiss.omp_set_num_threads(1)

        # build or load the index
        if self._index_path.exists() and self._meta_path.exists():
            self._load()
            # integrity check: validate the loaded index can actually mutate
            if not self._validate_index():
                logger.warning("index failed integrity check, rebuilding")
                self._rebuild_index()
        else:
            self._index = faiss.IndexFlatIP(dimension)  # inner product on normalized vecs
            if use_gpu and _FAISS_GPU:
                try:
                    self._index = faiss.index_cpu_to_all_gpus(self._index)
                    logger.info("faiss index moved to gpu")
                except Exception:
                    logger.warning("gpu transfer failed, staying on cpu")

    # -- public api --

    def upsert(self, entry: MemoryEntry) -> None:
        """append-only insert. skips duplicates, prunes oldest when full."""
        if entry.embedding is None:
            raise ValueError(f"entry '{entry.key}' has no embedding")
        # contiguous copy -- never hand faiss a view or transient buffer
        vec = np.ascontiguousarray(
            entry.embedding.reshape(1, -1), dtype=np.float32,
        ).copy()
        with self._lock:
            # skip duplicates rather than rebuild (avoids C-level corruption)
            if entry.key in self._entries:
                self._entries[entry.key] = entry
                return
            # prune oldest entries if at capacity
            if len(self._entries) >= self.MAX_ENTRIES:
                self._prune_oldest(keep=self.MAX_ENTRIES // 2)
            self._index.add(vec)
            self._keys_order.append(entry.key)
            self._entries[entry.key] = entry

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        """find closest entries by cosine similarity (inner product on normalized vecs)."""
        qv = np.ascontiguousarray(
            query_vec.reshape(1, -1), dtype=np.float32,
        ).copy()
        with self._lock:
            # ntotal check MUST be inside lock to avoid TOCTOU crash
            if self._index.ntotal == 0:
                return []
            k = min(top_k, self._index.ntotal)
            if k <= 0:
                return []
            scores, indices = self._index.search(qv, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._keys_order):
                    continue
                key = self._keys_order[idx]
                if entry := self._entries.get(key):
                    results.append((entry, float(score)))
            return results

    def get(self, key: str) -> Optional[MemoryEntry]:
        """retrieve an entry by key."""
        with self._lock:
            return self._entries.get(key)

    def delete(self, key: str) -> bool:
        """soft-delete: removes from entries dict. index is stale until next prune/save."""
        with self._lock:
            if key not in self._entries:
                return False
            del self._entries[key]
            # mark as removed; search will skip missing keys via the .get() check
            return True

    @property
    def size(self) -> int:
        with self._lock:
            return self._index.ntotal

    def search_by_thread(
        self, query_vec: np.ndarray, thread_id: str, top_k: int = 5,
    ) -> list[tuple[MemoryEntry, float]]:
        """search filtered to entries from a specific thread."""
        raw = self.search(query_vec, top_k=top_k * 3)
        filtered = [
            (entry, score) for entry, score in raw
            if entry.metadata.get("thread_id") == thread_id
        ]
        return filtered[:top_k]

    def clear(self) -> None:
        """wipe all entries and reset the index."""
        with self._lock:
            self._entries.clear()
            self._keys_order.clear()
            self._index = faiss.IndexFlatIP(self._dimension)
            logger.info("memory cleared")

    def get_stats(self) -> dict:
        """memory statistics for UI and status reporting."""
        with self._lock:
            timestamps = [e.timestamp for e in self._entries.values()]
            return {
                "entry_count": len(self._entries),
                "index_vectors": self._index.ntotal,
                "oldest": min(timestamps) if timestamps else 0,
                "newest": max(timestamps) if timestamps else 0,
            }

    def save(self) -> None:
        """atomic write: index + metadata sidecar.

        clones the faiss index before serialization so the live index's
        C++ internals are never touched from the timer thread -- avoids
        OpenMP segfaults when sentence-transformers runs concurrently.
        """
        with self._lock:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)

            # snapshot: clone the index so write_index operates on an
            # isolated C++ object (faiss is not thread-safe at the C level)
            cpu_index = self._cpu_index()
            cloned = faiss.clone_index(cpu_index)

            # snapshot python-side data while still under lock
            keys_snapshot = list(self._keys_order)
            entries_snapshot = dict(self._entries)

        # write from snapshots OUTSIDE the lock -- no contention
        tmp_idx = self._index_path.with_suffix(".faiss.tmp")
        faiss.write_index(cloned, str(tmp_idx))
        shutil.move(str(tmp_idx), str(self._index_path))

        meta = {
            "dimension": self._dimension,
            "keys_order": keys_snapshot,
            "entries": {},
        }
        for key, entry in entries_snapshot.items():
            entry_data = {
                "text": entry.text,
                "timestamp": entry.timestamp,
                "metadata": entry.metadata,
            }
            if entry.embedding is not None:
                entry_data["embedding_b64"] = base64.b64encode(
                    entry.embedding.astype(np.float32).tobytes()
                ).decode("ascii")
            meta["entries"][key] = entry_data
        tmp_meta = self._meta_path.with_suffix(".json.tmp")
        with open(tmp_meta, "w") as f:
            json.dump(meta, f, indent=2)
        shutil.move(str(tmp_meta), str(self._meta_path))
        logger.info("memory saved: %d entries", len(entries_snapshot))

    # -- internals --

    def _load(self) -> None:
        """restore index + metadata from disk."""
        logger.info("loading memory from %s", self._index_path)
        self._index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "r") as f:
            meta = json.load(f)
        self._dimension = meta.get("dimension", self._dimension)
        self._keys_order = meta.get("keys_order", [])
        for key, info in meta.get("entries", {}).items():
            embedding = None
            if "embedding_b64" in info:
                raw_bytes = base64.b64decode(info["embedding_b64"])
                embedding = np.frombuffer(raw_bytes, dtype=np.float32).copy()
            self._entries[key] = MemoryEntry(
                key=key,
                text=info["text"],
                embedding=embedding,
                timestamp=info.get("timestamp", 0.0),
                metadata=info.get("metadata", {}),
            )
        logger.info("memory loaded: %d entries", len(self._entries))

        # validate index alignment -- rebuild if corrupted
        if self._index.ntotal != len(self._keys_order):
            logger.warning(
                "index/keys mismatch (%d vs %d) -- rebuilding",
                self._index.ntotal, len(self._keys_order),
            )
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """reconstruct FAISS index from stored embeddings."""
        new_index = faiss.IndexFlatIP(self._dimension)
        valid_keys = []
        vecs = []
        for key in self._keys_order:
            entry = self._entries.get(key)
            if entry and entry.embedding is not None:
                vec = np.array(
                    entry.embedding.reshape(1, -1), dtype=np.float32, copy=True,
                )
                vecs.append(vec)
                valid_keys.append(key)
        if vecs:
            stacked = np.ascontiguousarray(np.vstack(vecs), dtype=np.float32)
            new_index.add(stacked)
        self._index = new_index
        self._keys_order = valid_keys
        logger.info("index rebuilt: %d vectors", self._index.ntotal)

    def _validate_index(self) -> bool:
        """smoke-test the loaded index with a dummy add+search cycle.

        catches subtle C-level corruption from partial writes during
        previous segfaults that pass read_index but corrupt on mutation.
        """
        try:
            test_idx = faiss.clone_index(self._index)
            dummy = np.random.randn(1, self._dimension).astype(np.float32)
            test_idx.add(dummy)
            test_idx.search(dummy, 1)
            del test_idx
            return True
        except Exception as exc:
            logger.warning("index validation failed: %s", exc)
            return False

    def _prune_oldest(self, keep: int = 100) -> None:
        """drop oldest entries and rebuild index. called under lock."""
        if len(self._keys_order) <= keep:
            return
        # keep the most recent entries
        pruned_keys = self._keys_order[-keep:]
        vecs = []
        for k in pruned_keys:
            entry = self._entries.get(k)
            if entry and entry.embedding is not None:
                # owned contiguous copy -- never pass views to FAISS
                vec = np.array(
                    entry.embedding.reshape(1, -1), dtype=np.float32, copy=True,
                )
                vecs.append(vec)
        # rebuild on a fresh index
        new_index = faiss.IndexFlatIP(self._dimension)
        if vecs:
            stacked = np.ascontiguousarray(
                np.vstack(vecs), dtype=np.float32,
            )
            new_index.add(stacked)
        self._index = new_index
        # clean up removed entries
        removed = set(self._keys_order) - set(pruned_keys)
        for k in removed:
            self._entries.pop(k, None)
        self._keys_order = pruned_keys
        logger.info("pruned memory: %d -> %d entries", len(removed) + keep, keep)

    def _cpu_index(self):
        """get a cpu copy if the index is on gpu."""
        try:
            return faiss.index_gpu_to_cpu(self._index)
        except Exception:
            return self._index
