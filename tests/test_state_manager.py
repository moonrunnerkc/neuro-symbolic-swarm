# Author: Bradley R. Kinnard
"""Tests for the StateManager symbolic ledger."""

import json
import threading
from pathlib import Path

import pytest

from src.state_manager import Fact, StateManager, MAX_FACTS_PER_THREAD


class TestFact:
    def test_key_generation(self):
        f = Fact(subject="User", predicate="name", obj="Brad")
        assert f.key == "user::name"

    def test_strips_whitespace(self):
        f = Fact(subject="  User  ", predicate=" Name ", obj=" Brad ")
        assert f.subject == "user"
        assert f.predicate == "name"
        assert f.obj == "Brad"

    def test_to_dict_roundtrip(self):
        f = Fact(subject="setting", predicate="location", obj="Mars", thread_id="t1")
        d = f.to_dict()
        restored = Fact.from_dict(d)
        assert restored.subject == f.subject
        assert restored.predicate == f.predicate
        assert restored.obj == f.obj
        assert restored.thread_id == f.thread_id
        assert restored.created_at == f.created_at

    def test_repr(self):
        f = Fact(subject="user", predicate="name", obj="Brad")
        assert "user" in repr(f)
        assert "name" in repr(f)


class TestStateManagerBasics:
    def test_init_creates_empty(self, tmp_path):
        path = tmp_path / "world_state.json"
        sm = StateManager(path)
        assert sm.size == 0
        assert sm.thread_count == 0

    def test_upsert_and_query(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        fact = Fact(subject="user", predicate="name", obj="Brad", thread_id="t1")
        sm.upsert(fact)

        results = sm.query("t1")
        assert len(results) == 1
        assert results[0].obj == "Brad"

    def test_upsert_overwrites(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="user", predicate="name", obj="Brad", thread_id="t1"))
        sm.upsert(Fact(subject="user", predicate="name", obj="Bradley", thread_id="t1"))

        results = sm.query("t1")
        assert len(results) == 1
        assert results[0].obj == "Bradley"

    def test_delete(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="user", predicate="name", obj="Brad", thread_id="t1"))
        assert sm.delete("t1", "user", "name") is True
        assert sm.size == 0

    def test_delete_nonexistent(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        assert sm.delete("t1", "ghost", "field") is False

    def test_clear_thread(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="a", predicate="b", obj="c", thread_id="t1"))
        sm.upsert(Fact(subject="x", predicate="y", obj="z", thread_id="t2"))
        sm.clear_thread("t1")
        assert sm.query("t1") == []
        assert len(sm.query("t2")) == 1

    def test_clear_all(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="a", predicate="b", obj="c", thread_id="t1"))
        sm.upsert(Fact(subject="x", predicate="y", obj="z", thread_id="t2"))
        sm.clear_all()
        assert sm.size == 0
        assert sm.thread_count == 0

    def test_query_subject(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="user", predicate="name", obj="Brad", thread_id="t1"))
        sm.upsert(Fact(subject="user", predicate="role", obj="creator", thread_id="t1"))
        sm.upsert(Fact(subject="setting", predicate="planet", obj="Mars", thread_id="t1"))

        user_facts = sm.query_subject("t1", "user")
        assert len(user_facts) == 2
        subjects = {f.predicate for f in user_facts}
        assert subjects == {"name", "role"}


class TestStateManagerPersistence:
    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "ws.json"
        sm = StateManager(path)
        sm.upsert(Fact(subject="dog", predicate="name", obj="Reaper", thread_id="t1"))
        sm.upsert(Fact(subject="dog", predicate="talent", obj="sings", thread_id="t1"))
        sm.save()

        sm2 = StateManager(path)
        assert sm2.size == 2
        facts = sm2.query("t1")
        names = {f.obj for f in facts}
        assert "Reaper" in names
        assert "sings" in names

    def test_save_skips_when_clean(self, tmp_path):
        path = tmp_path / "ws.json"
        sm = StateManager(path)
        sm.save()  # nothing dirty, should not create file
        assert not path.exists()

    def test_corrupt_file_recovers(self, tmp_path):
        path = tmp_path / "ws.json"
        path.write_text("not json at all", encoding="utf-8")
        sm = StateManager(path)
        assert sm.size == 0  # recovered gracefully

    def test_atomic_write(self, tmp_path):
        path = tmp_path / "ws.json"
        sm = StateManager(path)
        sm.upsert(Fact(subject="a", predicate="b", obj="c", thread_id="t1"))
        sm.save()

        # no .tmp files left behind
        assert not (tmp_path / "ws.tmp").exists()
        assert path.exists()

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["version"] == 1
        assert "t1" in raw["threads"]


class TestStateManagerGlobalScope:
    def test_global_facts_not_mixed(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="app", predicate="version", obj="1.0", thread_id="global"))
        sm.upsert(Fact(subject="user", predicate="name", obj="Brad", thread_id="t1"))

        # thread-only query excludes global
        t1_facts = sm.query("t1", include_global=False)
        assert len(t1_facts) == 1

    def test_global_merge(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="app", predicate="version", obj="1.0", thread_id="global"))
        sm.upsert(Fact(subject="user", predicate="name", obj="Brad", thread_id="t1"))

        merged = sm.query("t1", include_global=True)
        assert len(merged) == 2


class TestConstraintBlock:
    def test_empty_returns_empty_string(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        assert sm.constraint_block("nonexistent") == ""

    def test_generates_constraint_text(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="setting", predicate="location", obj="Mars", thread_id="t1"))
        sm.upsert(Fact(subject="timeline", predicate="year", obj="2099", thread_id="t1"))

        block = sm.constraint_block("t1")
        assert "ESTABLISHED FACTS" in block
        assert "Mars" in block
        assert "2099" in block
        assert "must not contradict" in block.lower()


class TestPruning:
    def test_prune_at_limit(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        # stuff more than MAX_FACTS_PER_THREAD facts
        for i in range(MAX_FACTS_PER_THREAD + 10):
            sm.upsert(Fact(
                subject=f"s{i}", predicate="p", obj=f"v{i}", thread_id="t1",
            ))
        # should have been pruned
        assert len(sm.query("t1")) <= MAX_FACTS_PER_THREAD


class TestConcurrency:
    def test_parallel_upserts(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        errors = []

        def worker(thread_num: int):
            try:
                for i in range(50):
                    sm.upsert(Fact(
                        subject=f"t{thread_num}-s{i}",
                        predicate="val",
                        obj=f"v{i}",
                        thread_id=f"thread-{thread_num}",
                    ))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert sm.size == 200  # 4 threads x 50 facts


class TestGetStats:
    def test_stats_reflect_state(self, tmp_path):
        sm = StateManager(tmp_path / "ws.json")
        sm.upsert(Fact(subject="a", predicate="b", obj="c", thread_id="t1"))
        sm.upsert(Fact(subject="x", predicate="y", obj="z", thread_id="t2"))

        stats = sm.get_stats()
        assert stats["total_facts"] == 2
        assert stats["threads"] == 2
        assert stats["facts_per_thread"]["t1"] == 1
        assert stats["facts_per_thread"]["t2"] == 1
