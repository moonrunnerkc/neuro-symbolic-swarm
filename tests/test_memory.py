# Author: Bradley R. Kinnard
"""Tests for the FAISS shared memory layer."""

import numpy as np
import pytest

from src.memory import MemoryEntry, SharedMemory


@pytest.fixture
def mem(tmp_path):
    """fresh in-memory SharedMemory instance for each test."""
    index_path = tmp_path / "test.faiss"
    return SharedMemory(index_path=index_path, dimension=4, use_gpu=False)


def _vec(values: list[float]) -> np.ndarray:
    """helper: make a normalized float32 vector."""
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestSharedMemory:
    def test_empty_on_init(self, mem):
        assert mem.size == 0

    def test_upsert_single(self, mem):
        entry = MemoryEntry(key="a", text="hello", embedding=_vec([1, 0, 0, 0]))
        mem.upsert(entry)
        assert mem.size == 1

    def test_upsert_replaces(self, mem):
        v1 = _vec([1, 0, 0, 0])
        v2 = _vec([0, 1, 0, 0])
        mem.upsert(MemoryEntry(key="a", text="v1", embedding=v1))
        mem.upsert(MemoryEntry(key="a", text="v2", embedding=v2))
        assert mem.size == 1
        found = mem.get("a")
        assert found is not None
        assert found.text == "v2"

    def test_search_returns_closest(self, mem):
        mem.upsert(MemoryEntry(key="a", text="a", embedding=_vec([1, 0, 0, 0])))
        mem.upsert(MemoryEntry(key="b", text="b", embedding=_vec([0, 1, 0, 0])))
        mem.upsert(MemoryEntry(key="c", text="c", embedding=_vec([0.9, 0.1, 0, 0])))

        results = mem.search(_vec([1, 0, 0, 0]), top_k=2)
        assert len(results) == 2
        # first result should be "a" (exact match)
        assert results[0][0].key == "a"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_search_empty_index(self, mem):
        results = mem.search(_vec([1, 0, 0, 0]))
        assert results == []

    def test_delete(self, mem):
        mem.upsert(MemoryEntry(key="a", text="a", embedding=_vec([1, 0, 0, 0])))
        assert mem.delete("a") is True
        # soft-delete: entry gone from dict, index stale until next prune/save
        assert mem.get("a") is None

    def test_delete_nonexistent(self, mem):
        assert mem.delete("nope") is False

    def test_upsert_no_embedding_raises(self, mem):
        with pytest.raises(ValueError, match="no embedding"):
            mem.upsert(MemoryEntry(key="bad", text="no vec"))

    def test_save_and_reload(self, tmp_path):
        index_path = tmp_path / "persist.faiss"
        m1 = SharedMemory(index_path=index_path, dimension=4, use_gpu=False)
        m1.upsert(MemoryEntry(key="x", text="saved", embedding=_vec([0, 0, 1, 0])))
        m1.save()

        m2 = SharedMemory(index_path=index_path, dimension=4, use_gpu=False)
        assert m2.size == 1
        entry = m2.get("x")
        assert entry is not None
        assert entry.text == "saved"

    def test_multiple_entries(self, mem):
        for i in range(10):
            v = np.zeros(4, dtype=np.float32)
            v[i % 4] = 1.0
            mem.upsert(MemoryEntry(key=f"k{i}", text=f"text{i}", embedding=v))
        assert mem.size == 10

    def test_get_nonexistent(self, mem):
        assert mem.get("missing") is None
