# Author: Bradley R. Kinnard
"""Symbolic state ledger for neuro-symbolic state anchoring.

Maintains per-thread factual triples extracted from conversation.
Provides constraint blocks for injection into synthesis, preventing
hallucination drift."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# max facts per thread before pruning oldest
MAX_FACTS_PER_THREAD = 100
_PRUNE_TARGET = MAX_FACTS_PER_THREAD // 2


class Fact:
    """a single subject-predicate-object triple with metadata."""

    __slots__ = ("subject", "predicate", "obj", "thread_id", "created_at", "updated_at")

    def __init__(
        self,
        subject: str,
        predicate: str,
        obj: str,
        thread_id: str = "global",
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
    ):
        now = time.time()
        self.subject = subject.strip().lower()
        self.predicate = predicate.strip().lower()
        self.obj = obj.strip()
        self.thread_id = thread_id
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @property
    def key(self) -> str:
        """unique identity: subject + predicate within a thread."""
        return f"{self.subject}::{self.predicate}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "obj": self.obj,
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        return cls(
            subject=data["subject"],
            predicate=data["predicate"],
            obj=data["obj"],
            thread_id=data.get("thread_id", "global"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def __repr__(self) -> str:
        return f"Fact({self.subject} -> {self.predicate} -> {self.obj})"


class StateManager:
    """thread-safe symbolic fact ledger with JSON persistence."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        # thread_id -> {fact_key -> Fact}
        self._threads: dict[str, dict[str, Fact]] = {}
        self._dirty = False
        self._load()

    # -- persistence --

    def _load(self) -> None:
        """load state from disk if it exists."""
        if not self._path.exists():
            self._threads = {}
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            threads_raw = raw.get("threads", {})
            for tid, facts_list in threads_raw.items():
                self._threads[tid] = {}
                for fd in facts_list:
                    fact = Fact.from_dict(fd)
                    self._threads[tid][fact.key] = fact
            logger.info("state ledger loaded: %d threads", len(self._threads))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("corrupt world_state.json, starting fresh: %s", exc)
            self._threads = {}

    def save(self) -> None:
        """atomic write to disk."""
        with self._lock:
            if not self._dirty:
                return
            self._write_locked()

    def _write_locked(self) -> None:
        """write current state to disk. caller must hold _lock."""
        payload = {
            "version": 1,
            "threads": {},
        }
        for tid, facts in self._threads.items():
            payload["threads"][tid] = [f.to_dict() for f in facts.values()]

        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)
        self._dirty = False
        logger.debug("state ledger saved: %d threads", len(self._threads))

    # -- mutations --

    # predicates that define the world and cannot be overwritten once set.
    # the extractor often misclassifies scene details (e.g. "underneath ice")
    # as the world "setting", clobbering the real value. lock these after first write.
    PROTECTED_PREDICATES: set[str] = {"setting", "genre", "era", "timeline", "planet"}

    def upsert(self, fact: Fact) -> None:
        """insert or update a fact. overwrites if same subject+predicate exists.

        protected predicates (setting, genre, era, timeline) are immutable
        once established for a thread -- subsequent writes are silently dropped.
        """
        with self._lock:
            tid = fact.thread_id
            if tid not in self._threads:
                self._threads[tid] = {}

            existing = self._threads[tid].get(fact.key)
            if existing:
                # guard world-anchor facts from overwrite
                if fact.predicate.strip().lower() in self.PROTECTED_PREDICATES:
                    logger.debug(
                        "protected predicate '%s' already set, ignoring update "
                        "'%s' -> '%s'",
                        fact.predicate, existing.obj, fact.obj,
                    )
                    return
                existing.obj = fact.obj
                existing.updated_at = time.time()
            else:
                self._threads[tid][fact.key] = fact

            # prune if over limit
            if len(self._threads[tid]) > MAX_FACTS_PER_THREAD:
                self._prune_thread(tid)

            self._dirty = True

    def _prune_thread(self, thread_id: str) -> None:
        """drop oldest facts to stay under limit. caller must hold _lock."""
        facts = self._threads[thread_id]
        sorted_facts = sorted(facts.values(), key=lambda f: f.updated_at)
        keep = sorted_facts[len(sorted_facts) - _PRUNE_TARGET:]
        self._threads[thread_id] = {f.key: f for f in keep}
        logger.info(
            "pruned thread %s: %d -> %d facts",
            thread_id, len(sorted_facts), _PRUNE_TARGET,
        )

    def delete(self, thread_id: str, subject: str, predicate: str) -> bool:
        """remove a specific fact. returns True if it existed."""
        key = f"{subject.strip().lower()}::{predicate.strip().lower()}"
        with self._lock:
            facts = self._threads.get(thread_id, {})
            if key in facts:
                del facts[key]
                self._dirty = True
                return True
            return False

    def clear_thread(self, thread_id: str) -> None:
        """wipe all facts for a thread."""
        with self._lock:
            if thread_id in self._threads:
                del self._threads[thread_id]
                self._dirty = True

    def clear_all(self) -> None:
        """wipe the entire ledger."""
        with self._lock:
            self._threads.clear()
            self._dirty = True

    # -- queries --

    def query(self, thread_id: str, include_global: bool = False) -> list[Fact]:
        """get all facts for a thread, optionally merging global scope."""
        with self._lock:
            results = list(self._threads.get(thread_id, {}).values())
            if include_global and thread_id != "global":
                results.extend(self._threads.get("global", {}).values())
            return sorted(results, key=lambda f: f.updated_at, reverse=True)

    def query_subject(self, thread_id: str, subject: str) -> list[Fact]:
        """get all facts about a specific subject in a thread."""
        subject_lower = subject.strip().lower()
        with self._lock:
            facts = self._threads.get(thread_id, {})
            return [f for f in facts.values() if f.subject == subject_lower]

    def constraint_block(self, thread_id: str, include_global: bool = False) -> str:
        """build a constraint string for injection into synthesis prompts."""
        facts = self.query(thread_id, include_global=include_global)
        if not facts:
            return ""

        lines = [
            "ESTABLISHED FACTS (these are absolute constraints):",
            "You MUST NOT contradict, ignore, or override any of these facts.",
            "If the user's new message conflicts with these facts, point out",
            "the conflict and stay consistent with the established facts.",
            "",
        ]
        for f in facts:
            lines.append(f"  {f.predicate}: {f.obj}")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        """total fact count across all threads."""
        with self._lock:
            return sum(len(fs) for fs in self._threads.values())

    @property
    def thread_count(self) -> int:
        with self._lock:
            return len(self._threads)

    def get_stats(self) -> dict:
        """summary for debug/status panels."""
        with self._lock:
            return {
                "total_facts": sum(len(fs) for fs in self._threads.values()),
                "threads": len(self._threads),
                "facts_per_thread": {
                    tid: len(fs) for tid, fs in self._threads.items()
                },
            }
