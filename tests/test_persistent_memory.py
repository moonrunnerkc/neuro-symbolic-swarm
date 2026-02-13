# Author: Bradley R. Kinnard
"""Tests for persistent memory: thread reload, embedding persistence,
context retrieval, thread-aware search, auto-save, clear."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from src.config import AppConfig, load_app_config
from src.agent import SwarmMessage
from src.memory import MemoryEntry, SharedMemory


# -- helpers --

def _vec(dim: int = 4, seed: int = 0) -> np.ndarray:
    """reproducible normalized vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_swarm_dir(tmp_path: Path, agents: list[str] | None = None) -> Path:
    """build a minimal swarm directory for testing."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for role in (agents or ["Parser", "Critic"]):
        data = {
            "role": role,
            "model": "phi3:mini",
            "prompt": f"you are a {role.lower()}",
            "max_tokens": 64,
            "score_threshold": 0.5,
        }
        (agents_dir / f"{role.lower()}.yaml").write_text(yaml.dump(data))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "threads").mkdir()
    config = {
        "theme": "dark",
        "agent_count": 2,
        "score_threshold": 0.5,
        "max_cycles": 2,
        "max_tokens": 64,
        "temperature": 0.7,
        "default_model": "phi3:mini",
        "log_level": "DEBUG",
        "context_window": 3,
        "cross_thread_memory": False,
        "auto_save_interval": 0,  # disabled for tests
    }
    (data_dir / "config.json").write_text(json.dumps(config))
    return tmp_path


def _swap_project_root(swarm_mod, new_root):
    """context-manager-style helper for patching _PROJECT_ROOT."""
    original = swarm_mod._PROJECT_ROOT
    swarm_mod._PROJECT_ROOT = new_root
    return original


# -- embedding persistence --

class TestEmbeddingPersistence:
    """embeddings must survive a save/load round-trip."""

    def test_embeddings_roundtrip(self, tmp_path):
        idx = tmp_path / "test.faiss"
        m1 = SharedMemory(index_path=idx, dimension=4)
        vec = _vec(4, seed=42)
        m1.upsert(MemoryEntry(key="k1", text="hello", embedding=vec))
        m1.save()

        m2 = SharedMemory(index_path=idx, dimension=4)
        entry = m2.get("k1")
        assert entry is not None
        assert entry.embedding is not None
        np.testing.assert_array_almost_equal(entry.embedding, vec, decimal=5)

    def test_multiple_embeddings_roundtrip(self, tmp_path):
        idx = tmp_path / "test.faiss"
        m1 = SharedMemory(index_path=idx, dimension=4)
        vecs = {f"k{i}": _vec(4, seed=i) for i in range(5)}
        for key, vec in vecs.items():
            m1.upsert(MemoryEntry(key=key, text=f"text-{key}", embedding=vec))
        m1.save()

        m2 = SharedMemory(index_path=idx, dimension=4)
        for key, expected_vec in vecs.items():
            entry = m2.get(key)
            assert entry is not None
            assert entry.embedding is not None
            np.testing.assert_array_almost_equal(
                entry.embedding, expected_vec, decimal=5,
            )

    def test_loaded_entries_are_searchable(self, tmp_path):
        idx = tmp_path / "test.faiss"
        m1 = SharedMemory(index_path=idx, dimension=4)
        v1 = _vec(4, seed=1)
        v2 = _vec(4, seed=2)
        m1.upsert(MemoryEntry(key="a", text="alpha", embedding=v1))
        m1.upsert(MemoryEntry(key="b", text="beta", embedding=v2))
        m1.save()

        m2 = SharedMemory(index_path=idx, dimension=4)
        results = m2.search(v1, top_k=2)
        assert len(results) == 2
        assert results[0][0].key == "a"

    def test_rebuild_on_mismatch(self, tmp_path):
        """if index and keys diverge, load should rebuild from embeddings."""
        idx = tmp_path / "test.faiss"
        m1 = SharedMemory(index_path=idx, dimension=4)
        vec = _vec(4, seed=7)
        m1.upsert(MemoryEntry(key="x", text="test", embedding=vec))
        m1.save()

        # corrupt the keys_order in the sidecar
        meta_path = idx.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["keys_order"].append("phantom_key")
        meta_path.write_text(json.dumps(meta))

        m2 = SharedMemory(index_path=idx, dimension=4)
        # should have rebuilt; only "x" has a real embedding
        assert m2.size == 1
        assert m2.get("x") is not None

    def test_backward_compat_no_embeddings(self, tmp_path):
        """old sidecar without embedding_b64 should still load (embeddings=None)."""
        idx = tmp_path / "test.faiss"
        m1 = SharedMemory(index_path=idx, dimension=4)
        vec = _vec(4, seed=3)
        m1.upsert(MemoryEntry(key="old", text="legacy entry", embedding=vec))
        m1.save()

        # strip embedding_b64 from sidecar to simulate old format
        meta_path = idx.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        for entry_data in meta["entries"].values():
            entry_data.pop("embedding_b64", None)
        meta_path.write_text(json.dumps(meta))

        m2 = SharedMemory(index_path=idx, dimension=4)
        entry = m2.get("old")
        assert entry is not None
        assert entry.text == "legacy entry"
        # embedding is None because it wasn't in the sidecar
        assert entry.embedding is None


# -- thread persistence --

class TestThreadPersistence:
    def test_threads_reload_on_init(self, tmp_path):
        root = _make_swarm_dir(tmp_path)
        threads_dir = root / "data" / "threads"

        thread_data = {
            "id": "old-session",
            "messages": [
                {"role": "user", "content": "hey there", "timestamp": 1000.0},
                {"role": "swarm", "content": "hello!", "timestamp": 1001.0},
            ],
        }
        (threads_dir / "old-session.json").write_text(json.dumps(thread_data))

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            assert "old-session" in swarm._active_threads
            history = swarm.get_thread_history("old-session")
            assert len(history) == 2
            assert history[0]["content"] == "hey there"
            assert history[1]["content"] == "hello!"
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_multiple_threads_reload(self, tmp_path):
        root = _make_swarm_dir(tmp_path)
        threads_dir = root / "data" / "threads"

        for name in ["thread-a", "thread-b", "thread-c"]:
            data = {"id": name, "messages": [{"role": "user", "content": name}]}
            (threads_dir / f"{name}.json").write_text(json.dumps(data))

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            assert "thread-a" in swarm._active_threads
            assert "thread-b" in swarm._active_threads
            assert "thread-c" in swarm._active_threads
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_corrupt_thread_file_skipped(self, tmp_path):
        root = _make_swarm_dir(tmp_path)
        threads_dir = root / "data" / "threads"

        # valid thread
        valid = {"id": "good", "messages": []}
        (threads_dir / "good.json").write_text(json.dumps(valid))
        # corrupt thread
        (threads_dir / "bad.json").write_text("{broken json!!")

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            assert "good" in swarm._active_threads
            # "bad" should be skipped, not crash
            assert "bad" not in swarm._active_threads
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_create_thread_idempotent(self, tmp_path):
        root = _make_swarm_dir(tmp_path)
        threads_dir = root / "data" / "threads"

        thread_data = {
            "id": "existing",
            "messages": [{"role": "user", "content": "preserved"}],
        }
        (threads_dir / "existing.json").write_text(json.dumps(thread_data))

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            tid = swarm.create_thread("existing")
            assert tid == "existing"
            # messages must NOT be wiped
            assert len(swarm._active_threads["existing"]) == 1
            assert swarm._active_threads["existing"][0]["content"] == "preserved"
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_create_new_thread_works(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            tid = swarm.create_thread("fresh")
            assert tid == "fresh"
            assert swarm._active_threads["fresh"] == []
            thread_file = root / "data" / "threads" / "fresh.json"
            assert thread_file.exists()
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original


# -- memory search --

class TestMemorySearch:
    def test_search_by_thread(self, tmp_path):
        idx = tmp_path / "test.faiss"
        mem = SharedMemory(index_path=idx, dimension=4)
        v1 = _vec(4, seed=10)
        v2 = _vec(4, seed=11)
        mem.upsert(MemoryEntry(
            key="t1-q", text="thread1 query", embedding=v1,
            metadata={"thread_id": "t1", "type": "query"},
        ))
        mem.upsert(MemoryEntry(
            key="t2-q", text="thread2 query", embedding=v2,
            metadata={"thread_id": "t2", "type": "query"},
        ))

        results = mem.search_by_thread(v1, thread_id="t1", top_k=5)
        assert len(results) >= 1
        assert all(e.metadata["thread_id"] == "t1" for e, _ in results)

    def test_search_by_thread_excludes_other(self, tmp_path):
        idx = tmp_path / "test.faiss"
        mem = SharedMemory(index_path=idx, dimension=4)
        v = _vec(4, seed=0)
        mem.upsert(MemoryEntry(
            key="only-t2", text="only in t2", embedding=v,
            metadata={"thread_id": "t2"},
        ))

        results = mem.search_by_thread(v, thread_id="t1", top_k=5)
        assert len(results) == 0

    def test_clear(self, tmp_path):
        idx = tmp_path / "test.faiss"
        mem = SharedMemory(index_path=idx, dimension=4)
        mem.upsert(MemoryEntry(key="a", text="a", embedding=_vec(4)))
        mem.upsert(MemoryEntry(key="b", text="b", embedding=_vec(4, seed=1)))
        assert mem.size == 2

        mem.clear()
        assert mem.size == 0
        assert mem.get("a") is None
        assert mem.get("b") is None

    def test_get_stats(self, tmp_path):
        idx = tmp_path / "test.faiss"
        mem = SharedMemory(index_path=idx, dimension=4)
        mem.upsert(MemoryEntry(
            key="a", text="a", embedding=_vec(4), timestamp=100.0,
        ))
        mem.upsert(MemoryEntry(
            key="b", text="b", embedding=_vec(4, seed=1), timestamp=200.0,
        ))
        stats = mem.get_stats()
        assert stats["entry_count"] == 2
        assert stats["index_vectors"] == 2
        assert stats["oldest"] == 100.0
        assert stats["newest"] == 200.0

    def test_get_stats_empty(self, tmp_path):
        idx = tmp_path / "test.faiss"
        mem = SharedMemory(index_path=idx, dimension=4)
        stats = mem.get_stats()
        assert stats["entry_count"] == 0
        assert stats["oldest"] == 0
        assert stats["newest"] == 0


# -- context retrieval --

class TestContextRetrieval:
    def test_retrieve_returns_relevant(self, tmp_path):
        """swarm._retrieve_context should return formatted memory hits."""
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            swarm.create_thread("ctx-test")

            # seed memory with a known entry
            vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
            swarm._memory.upsert(MemoryEntry(
                key="seeded",
                text="the capital of France is Paris",
                embedding=vec,
                metadata={"thread_id": "ctx-test", "type": "response"},
            ))

            context = swarm._retrieve_context(vec, "ctx-test")
            assert "Paris" in context
            assert "RELEVANT PAST CONTEXT" in context
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_retrieve_empty_when_disabled(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            swarm._app_config.context_window = 0
            vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
            context = swarm._retrieve_context(vec, "any")
            assert context == ""
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_cross_thread_filtering(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            swarm.create_thread("t1")
            swarm.create_thread("t2")

            vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
            swarm._memory.upsert(MemoryEntry(
                key="t2-data",
                text="only in thread2",
                embedding=vec,
                metadata={"thread_id": "t2", "type": "response"},
            ))

            # cross_thread_memory=False, searching from t1 should miss t2 data
            context = swarm._retrieve_context(vec, "t1")
            assert "only in thread2" not in context

            # enable cross-thread, should find it
            swarm._app_config.cross_thread_memory = True
            context = swarm._retrieve_context(vec, "t1")
            assert "only in thread2" in context
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_retrieve_empty_memory(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            swarm.create_thread("empty")
            vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
            context = swarm._retrieve_context(vec, "empty")
            assert context == ""
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original


# -- config backward compatibility --

class TestAppConfigNewFields:
    def test_defaults_present(self):
        cfg = AppConfig()
        assert cfg.context_window == 5
        assert cfg.cross_thread_memory is False
        assert cfg.auto_save_interval == 60

    def test_backward_compatible_load(self, tmp_path):
        """existing config.json without new fields should load fine."""
        old_config = {
            "theme": "dark",
            "agent_count": 5,
            "score_threshold": 0.8,
            "max_cycles": 10,
            "max_tokens": 256,
            "temperature": 0.7,
            "default_model": "phi3:mini",
            "log_level": "INFO",
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(old_config))

        cfg = load_app_config(path)
        assert cfg.context_window == 5
        assert cfg.cross_thread_memory is False
        assert cfg.auto_save_interval == 60

    def test_custom_values_load(self, tmp_path):
        config = {
            "theme": "dark",
            "agent_count": 3,
            "score_threshold": 0.5,
            "max_cycles": 5,
            "max_tokens": 128,
            "temperature": 0.3,
            "default_model": "phi3:mini",
            "log_level": "DEBUG",
            "context_window": 10,
            "cross_thread_memory": True,
            "auto_save_interval": 120,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))

        cfg = load_app_config(path)
        assert cfg.context_window == 10
        assert cfg.cross_thread_memory is True
        assert cfg.auto_save_interval == 120


# -- SwarmMessage context field --

class TestSwarmMessageContext:
    def test_context_default_empty(self):
        msg = SwarmMessage(content="hello")
        assert msg.context == ""

    def test_context_set(self):
        msg = SwarmMessage(content="hello", context="some past info")
        assert msg.context == "some past info"

    def test_to_dict_unchanged(self):
        """context field shouldn't break serialization."""
        msg = SwarmMessage(content="test", context="ctx", score=0.5)
        d = msg.to_dict()
        assert "content" in d
        assert "score" in d


# -- clear memory --

class TestClearMemory:
    def test_clear_memory_via_swarm(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            # seed some memory
            vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
            swarm._memory.upsert(MemoryEntry(
                key="will-be-cleared", text="temp data", embedding=vec,
            ))
            assert swarm._memory.size == 1

            swarm.clear_memory()
            assert swarm._memory.size == 0
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original


# -- auto-save --

class TestAutoSave:
    def test_auto_save_disabled_when_zero(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            # interval=0 in test config, so timer should be None
            assert swarm._auto_save_timer is None
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original

    def test_auto_save_enabled(self, tmp_path):
        root = _make_swarm_dir(tmp_path)
        # set a nonzero interval
        config_path = root / "data" / "config.json"
        cfg = json.loads(config_path.read_text())
        cfg["auto_save_interval"] = 300
        config_path.write_text(json.dumps(cfg))

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            assert swarm._auto_save_timer is not None
            assert swarm._auto_save_timer.daemon is True
            swarm.close()
            # timer should be cancelled after close
            assert swarm._auto_save_timer is None
        finally:
            swarm_mod._PROJECT_ROOT = original


# -- get_status includes memory stats --

class TestGetStatusMemory:
    def test_status_includes_memory_dict(self, tmp_path):
        root = _make_swarm_dir(tmp_path)

        import src.swarm as swarm_mod
        original = _swap_project_root(swarm_mod, root)
        try:
            swarm = swarm_mod.SwarmNexus(
                config_path="data/config.json", agents_dir="agents",
            )
            status = swarm.get_status()
            assert "memory" in status
            assert "entry_count" in status["memory"]
            assert "index_vectors" in status["memory"]
            assert "oldest" in status["memory"]
            assert "newest" in status["memory"]
            swarm.close()
        finally:
            swarm_mod._PROJECT_ROOT = original
