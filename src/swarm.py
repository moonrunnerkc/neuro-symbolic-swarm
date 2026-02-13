# Author: Bradley R. Kinnard
"""Swarm orchestrator. Runs agents via ThreadPoolExecutor, manages
cycles, aggregates responses by voting. The core API surface."""

from __future__ import annotations

import atexit
import json
import logging
import re
import threading as _threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from src.agent import Agent, AgentError, AgentStatus, SwarmMessage
from src.config import (
    AgentConfig,
    AppConfig,
    load_all_agents,
    load_app_config,
    save_agent_config,
    save_app_config,
)
from src.embedder import embed_text
from src.memory import MemoryEntry, SharedMemory
from src.state_manager import Fact, StateManager
from src.web_search import (
    build_grounding_block,
    extract_search_query,
    needs_grounding,
    search as web_search,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SwarmNexus:
    """neuro-symbolic orchestration nexus.

    agents run in parallel via threads. each scores its response
    against the query embedding. the orchestrator collects, validates
    against the symbolic state ledger, and synthesizes the best result.
    """

    def __init__(
        self,
        num_agents: int = 5,
        config_path: str = "data/config.json",
        agents_dir: str = "agents",
        ollama_url: str = "http://localhost:11434",
    ):
        self._config_path = _PROJECT_ROOT / config_path
        self._agents_dir = _PROJECT_ROOT / agents_dir
        self._ollama_url = ollama_url

        if self._config_path.exists():
            self._app_config = load_app_config(self._config_path)
        else:
            self._app_config = AppConfig(agent_count=num_agents)

        # shared memory
        memory_path = _PROJECT_ROOT / "data" / "memory.faiss"
        self._memory = SharedMemory(index_path=memory_path)

        # threads -- reload any saved conversations from disk
        self._threads_dir = _PROJECT_ROOT / "data" / "threads"
        self._threads_dir.mkdir(parents=True, exist_ok=True)
        self._active_threads: dict[str, list[dict]] = {}
        self._load_existing_threads()

        # symbolic state ledger
        state_path = _PROJECT_ROOT / "data" / "world_state.json"
        self._state = StateManager(state_path)

        # agents
        self._agents: list[Agent] = []
        self._init_agents()

        self._pool = ThreadPoolExecutor(
            max_workers=max(len(self._agents), 2),
            thread_name_prefix="agent",
        )

        # periodic auto-save so memory survives crashes
        self._auto_save_timer: Optional[_threading.Timer] = None
        self._busy = _threading.Event()  # set during active respond cycles
        self._start_auto_save()
        atexit.register(self._emergency_save)

        logger.info(
            "swarm initialized: %d agents, %d threads restored, %d facts, config=%s",
            len(self._agents), len(self._active_threads),
            self._state.size, self._config_path,
        )

    def _init_agents(self) -> None:
        try:
            configs = load_all_agents(self._agents_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("failed to load agents: %s", exc)
            raise

        for cfg in configs:
            agent = Agent(config=cfg, ollama_url=self._ollama_url)
            self._agents.append(agent)

    def _load_existing_threads(self) -> None:
        """reload saved threads from disk into runtime state."""
        for thread_file in sorted(self._threads_dir.glob("*.json")):
            try:
                data = json.loads(thread_file.read_text())
                tid = data.get("id", thread_file.stem)
                messages = data.get("messages", [])
                self._active_threads[tid] = messages
                logger.info("thread loaded: %s (%d messages)", tid, len(messages))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("corrupt thread %s: %s", thread_file.name, exc)

    def create_thread(self, thread_id: str) -> str:
        if not thread_id.strip():
            raise ValueError("thread_id cannot be empty")
        tid = thread_id.strip()
        # don't wipe threads that were loaded from disk
        if tid in self._active_threads:
            return tid
        self._active_threads[tid] = []
        thread_file = self._threads_dir / f"{tid}.json"
        thread_file.write_text(json.dumps({"id": tid, "messages": []}, indent=2))
        logger.info("thread created: %s", tid)
        return tid

    def respond(
        self,
        query: str,
        thread_id: str,
        on_progress: Optional[callable] = None,
    ) -> str:
        """feed query to the swarm, run parallel agents, return best response."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if thread_id not in self._active_threads:
            raise ValueError(f"unknown thread: {thread_id}")

        # block auto-save from touching FAISS while we're active
        self._busy.set()

        def _progress(stage: str) -> None:
            if on_progress:
                on_progress(stage)

        try:
            return self._do_respond(query, thread_id, query_vec=None, progress=_progress)
        finally:
            self._busy.clear()

    def _do_respond(
        self,
        query: str,
        thread_id: str,
        query_vec: Optional[np.ndarray],
        progress: callable,
    ) -> str:
        """inner respond logic, always called with _busy set."""
        progress("embedding query vector")
        query_vec = embed_text(query)

        # retrieve relevant past context from memory
        progress("retrieving context from memory")
        context = self._retrieve_context(query_vec, thread_id)

        # store query in memory
        progress("indexing query in FAISS")
        self._memory.upsert(MemoryEntry(
            key=f"query-{uuid.uuid4().hex[:8]}",
            text=query,
            embedding=query_vec,
            metadata={"thread_id": thread_id, "type": "query"},
        ))

        seed = SwarmMessage(
            source="user",
            target="all",
            content=query,
            context=context,
            query_embedding=query_vec,
            cycle=0,
        )

        # snapshot world-anchor facts BEFORE extraction so we have a clean
        # baseline in case the new message poisons the ledger
        pre_facts = {
            f.predicate: f.obj
            for f in self._state.query(thread_id, include_global=True)
        }

        # -- GATE: validate user input BEFORE fact extraction --
        # this prevents poisoned messages from corrupting the ledger.
        # if the user sends anachronisms or contradicts locked facts,
        # reject immediately without running the extractor or agents.
        if pre_facts:
            from src.constraints import (
                WORLD_ANCHOR_KEYS,
                find_anachronisms,
                find_fact_contradictions,
                get_blocklist,
            )

            # anachronism check requires world anchors
            anchors = {
                k: v.lower() for k, v in pre_facts.items()
                if k in WORLD_ANCHOR_KEYS
            }
            if anchors:
                blocklist = get_blocklist(anchors)
                user_anachronisms = find_anachronisms(query, blocklist)
                if user_anachronisms:
                    progress("ledger:blocked — anachronisms in user input")
                    world_summary = ", ".join(
                        f"{k}={v}" for k, v in pre_facts.items()
                        if k in WORLD_ANCHOR_KEYS
                    )
                    final = (
                        f"That request conflicts with the established world for this thread. "
                        f"The current setting is: {world_summary}. "
                        f"Your message contains terms ({', '.join(sorted(user_anachronisms))}) "
                        f"that don't belong in this setting. "
                        f"Please rephrase your request to fit within the established setting."
                    )
                    self._record_exchange(query, final, thread_id)
                    progress("done")
                    return final

            # fact contradiction check runs against ALL locked facts,
            # not just world anchors — catches age/role/name conflicts too
            fact_contradictions = find_fact_contradictions(query, pre_facts)
            if fact_contradictions:
                progress("ledger:blocked — contradicts locked facts")
                final = (
                    f"That conflicts with established facts in this thread: "
                    f"{'; '.join(fact_contradictions)}. The state ledger is locked. "
                    f"Please stay consistent with the established world."
                )
                self._record_exchange(query, final, thread_id)
                progress("done")
                return final

        # phase 1a: fact extraction -- runs alongside answerers
        progress("extracting facts from input")
        self._run_fact_extraction(query, thread_id)
        # report ledger state after extraction
        ledger_facts = self._state.query(thread_id, include_global=True)
        if ledger_facts:
            progress(f"ledger:locked — {len(ledger_facts)} facts anchored")

        # phase 1a.5: web search grounding -- only when enabled + factual queries
        grounding_block = ""
        if self._app_config.web_search_enabled:
            current_genre = ""
            for f in self._state.query(thread_id):
                if f.predicate == "genre":
                    current_genre = f.obj
                    break
            if needs_grounding(query, genre=current_genre):
                progress("searching web for facts")
                search_query = extract_search_query(query)
                hits = web_search(search_query, max_results=3)
                grounding_block = build_grounding_block(hits)
                if grounding_block:
                    logger.info("web grounding: %d sources injected", len(hits))

        # phase 1b: all non-synthesizer/non-extractor agents answer
        answerers = [
            a for a in self._agents
            if a.role not in ("Synthesizer", "Fact-Extractor")
        ]
        progress(f"dispatching {len(answerers)} agents")
        responses = self._run_agents(answerers, seed, on_progress=progress)

        # phase 1.5: validation -- symbolic + critic filters contradictory drafts
        progress("symbolic validation — checking drafts against ledger")
        validated = self._validate_drafts(
            query, responses, query_vec, thread_id,
            pre_facts=pre_facts, on_progress=progress,
        )

        # if symbolic validation issued a hard refusal (StateAnchor),
        # bypass the synthesizer entirely -- don't let it rewrite the refusal
        if len(validated) == 1 and validated[0].source == "StateAnchor":
            # roll back any facts the extractor pulled from the invalid message
            self._rollback_facts(thread_id, pre_facts)
            # use the StateAnchor's refusal message directly -- it already
            # contains the contradiction details built from pre_facts
            final = validated[0].content
        else:
            # phase 2: synthesizer combines the validated answers
            progress(f"Synthesizer:active")
            progress(f"synthesizing from {len(validated)} validated drafts")
            final = self._synthesize(
                query, validated, query_vec, thread_id,
                grounding_block=grounding_block,
            )

        # free synthesizer model from VRAM after synthesis
        synth_agent = next((a for a in self._agents if a.role == "Synthesizer"), None)
        if synth_agent:
            progress("Synthesizer:idle")
            self._unload_model(synth_agent.model)

        # record in thread with timestamps
        progress("persisting to thread history")
        self._record_exchange(query, final, thread_id)

        progress("done")
        return final

    def _record_exchange(self, query: str, response: str, thread_id: str) -> None:
        """save a query/response pair to thread history and memory."""
        now = time.time()
        self._active_threads[thread_id].append({
            "role": "user", "content": query, "timestamp": now,
        })
        self._active_threads[thread_id].append({
            "role": "swarm", "content": response, "timestamp": now,
        })
        self._save_thread(thread_id)

        # store in memory
        if response.strip() and not response.startswith("no agents"):
            resp_vec = embed_text(response)
            self._memory.upsert(MemoryEntry(
                key=f"resp-{uuid.uuid4().hex[:8]}",
                text=response,
                embedding=resp_vec,
                metadata={"thread_id": thread_id, "type": "response"},
            ))

    def _run_fact_extraction(self, query: str, thread_id: str) -> None:
        """dispatch the fact-extractor agent, parse triples, upsert to state ledger.

        only runs on declarative messages (not questions or short commands).
        the extractor tends to hallucinate facts from questions like
        'How does Elara read the map?' producing garbage like genre=sci-fi.
        """
        # skip extraction on questions and short messages -- they rarely
        # contain new world-building facts and the extractor hallucinates
        stripped = query.strip()
        if stripped.endswith("?") or len(stripped) < 40:
            logger.debug("fact extraction skipped: question or short message")
            return

        extractor = next(
            (a for a in self._agents if a.role == "Fact-Extractor"), None,
        )
        if extractor is None:
            return

        msg = SwarmMessage(
            source="user",
            target="Fact-Extractor",
            content=query,
            cycle=0,
        )

        try:
            result = extractor.process_message(msg)
            if not result or not result.content.strip():
                return

            # parse the JSON output from the extractor
            raw = result.content.strip()
            # handle models that wrap JSON in markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            facts = self._extract_json_object(raw)
            if not facts:
                return

            # validate extraction output -- reject hallucinated facts
            # that have no basis in the user's actual message
            from src.constraints import validate_extracted_facts
            before_count = len(facts)
            facts = validate_extracted_facts(facts, query)
            if len(facts) < before_count:
                logger.info(
                    "extraction validation: %d/%d facts rejected as ungrounded",
                    before_count - len(facts), before_count,
                )
            if not facts:
                return

            # normalize non-canonical keys to canonical ones
            facts = self._normalize_fact_keys(facts)

            for key, value in facts.items():
                # skip nested structures -- only flat string/number values
                # are reliable enough to store as world facts
                if isinstance(value, (dict, list)):
                    logger.debug("skipping nested value: %s=%s", key, type(value).__name__)
                    continue
                str_val = str(value).strip()
                # skip empty, blank, or trivially useless values
                if not str_val or str_val in ('""', "''", '{}', '[]', 'null', 'None'):
                    logger.debug("skipping empty fact: %s='%s'", key, value)
                    continue
                fact = Fact(
                    subject="user",
                    predicate=str(key),
                    obj=str_val,
                    thread_id=thread_id,
                )
                result_status = self._state.upsert(fact)
                if result_status == "conflict":
                    # the user is trying to establish a new world in a
                    # thread that already has a locked world state
                    existing = self._state.query(thread_id)
                    locked_val = next(
                        (f.obj for f in existing if f.predicate == key), "?"
                    )
                    logger.warning(
                        "world-lock conflict: '%s' is locked to '%s' in this "
                        "thread, new value '%s' was rejected. suggest new thread.",
                        key, locked_val, str_val,
                    )

            self._state.save()
            logger.info(
                "fact extraction: %d facts upserted for thread %s",
                len(facts), thread_id,
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("fact extraction parse failed: %s", exc)
        except Exception as exc:
            logger.error("fact extraction error: %s", exc)

    # map of common non-canonical extractor keys to canonical ones
    _KEY_ALIASES: dict[str, str] = {
        "project": "project_name",
        "system_name": "project_name",
        "name": "project_name",
        "language": "programming_language",
        "lang": "programming_language",
        "database_system": "database",
        "db": "database",
        "database_type": "database",
        "runtime": "framework",
        "framework_name": "framework",
        "team": "team_members",
        "team_composition": "team_members",
        "due_date": "deadline",
        "target_date": "deadline",
    }

    @staticmethod
    def _normalize_fact_keys(facts: dict[str, str]) -> dict[str, str]:
        """map common extractor key variants to canonical names.

        Small models ignore canonical key instructions and invent their own.
        This catches the most common aliases without losing data.
        """
        normalized: dict[str, str] = {}
        for key, value in facts.items():
            canonical = SwarmNexus._KEY_ALIASES.get(key, key)
            # don't overwrite if canonical key already present
            if canonical not in normalized:
                normalized[canonical] = value
            else:
                logger.debug(
                    "skipping duplicate canonical key %s (from %s)",
                    canonical, key,
                )
        return normalized

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        """pull the first valid JSON object from text, ignoring trailing junk."""
        # fast path: entire string is valid JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj:
                return obj
            return None
        except json.JSONDecodeError:
            pass

        # slow path: model appended commentary after the JSON
        # find the first '{' and try progressively shorter substrings
        start = text.find("{")
        if start == -1:
            return None

        # scan for matching closing brace by tracking depth
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj:
                            return obj
                    except json.JSONDecodeError:
                        continue

        # last resort: truncated JSON — extract whatever top-level
        # "key": "value" pairs we can find via regex. this handles
        # the common case where the model hits max_tokens mid-response.
        pairs = re.findall(
            r'"([a-z_]+)"\s*:\s*"([^"]*)"',
            text[start:],
        )
        if pairs:
            result = {k: v for k, v in pairs if v.strip()}
            if result:
                logger.info(
                    "extracted %d facts from truncated JSON", len(result),
                )
                return result

        return None

    def _rollback_facts(
        self,
        thread_id: str,
        snapshot: dict[str, str],
    ) -> None:
        """restore the fact ledger to a pre-extraction snapshot.

        removes any predicates that were added by the most recent extraction
        and weren't present before. prevents invalid user messages from
        poisoning the world state.
        """
        current_facts = self._state.query(thread_id)
        removed = 0
        for f in current_facts:
            if f.predicate not in snapshot:
                self._state.delete(thread_id, f.subject, f.predicate)
                removed += 1
        if removed:
            self._state.save()
            logger.info(
                "rolled back %d poisoned facts from thread %s",
                removed, thread_id,
            )

    def _run_agents(
        self,
        agents: list[Agent],
        msg: SwarmMessage,
        on_progress: Optional[callable] = None,
    ) -> list[SwarmMessage]:
        """dispatch agents interleaved: one agent per model at a time.

        groups agents by model, then runs one from each model
        concurrently (different models can coexist in VRAM). never
        two requests to the same model simultaneously -- that would
        double the KV cache and risk OOM on the 12GB card.
        """
        if not agents:
            return []

        # group agents by model into queues
        model_queues: dict[str, list[Agent]] = {}
        for agent in agents:
            model_queues.setdefault(agent.model, []).append(agent)

        results: list[SwarmMessage] = []

        # interleave: pick one agent from each model queue per round
        while any(model_queues.values()):
            batch: list[Agent] = []
            empty_models = []
            for model_name, queue in model_queues.items():
                if queue:
                    batch.append(queue.pop(0))
                if not queue:
                    empty_models.append(model_name)
            for m in empty_models:
                if not model_queues[m]:
                    del model_queues[m]

            if not batch:
                break

            logger.info(
                "interleaved round: %s",
                ", ".join(f"{a.role}({a.model})" for a in batch),
            )

            # run one agent per model in parallel
            futures = {}
            for agent in batch:
                if on_progress:
                    on_progress(f"{agent.role}:active")
                fut = self._pool.submit(agent.process_message, msg)
                futures[fut] = agent.role

            for fut in as_completed(futures, timeout=120):
                role = futures[fut]
                try:
                    result = fut.result()
                    if on_progress:
                        on_progress(f"{role}:idle")
                    if result and result.content.strip():
                        # enforce score gate per the architecture:
                        # only outputs exceeding threshold propagate
                        if result.metadata.get("gated", False):
                            logger.info(
                                "agent %s gated (score=%.3f < threshold)",
                                role, result.score,
                            )
                            if on_progress:
                                on_progress(f"{role} gated (score: {result.score:.2f})")
                        else:
                            results.append(result)
                            if on_progress:
                                on_progress(f"{role} responded (score: {result.score:.2f})")
                        logger.info(
                            "agent %s | score=%.3f | gated=%s | %d chars",
                            role, result.score,
                            result.metadata.get("gated", False),
                            len(result.content),
                        )
                except Exception as exc:
                    if on_progress:
                        on_progress(f"{role}:error")
                        on_progress(f"{role} failed")
                    logger.error("agent %s failed: %s", role, exc)

        # unload all non-synthesizer models to free VRAM for synthesis
        synth_model = None
        for a in self._agents:
            if a.role == "Synthesizer":
                synth_model = a.model
                break
        unloaded = set()
        for agent in agents:
            if agent.model != synth_model and agent.model not in unloaded:
                self._unload_model(agent.model)
                unloaded.add(agent.model)

        return results

    def _unload_model(self, model_name: str) -> None:
        """tell ollama to drop a model from VRAM immediately."""
        try:
            import requests
            requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            logger.info("unloaded model %s from VRAM", model_name)
        except Exception as exc:
            logger.warning("failed to unload model %s: %s", model_name, exc)

    def _validate_drafts(
        self,
        query: str,
        responses: list[SwarmMessage],
        query_vec: np.ndarray,
        thread_id: str = "",
        pre_facts: dict[str, str] | None = None,
        on_progress: Optional[callable] = None,
    ) -> list[SwarmMessage]:
        """pre-synthesis validation: symbolic ledger check (hard reject) then Critic."""
        if not responses:
            return responses

        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # -- phase A: symbolic validation against the state ledger --
        symbolically_valid = self._symbolic_validate(
            responses, thread_id, query, pre_facts=pre_facts,
            on_progress=on_progress,
        )

        if len(symbolically_valid) <= 1:
            return symbolically_valid if symbolically_valid else responses

        # -- phase B: critic LLM validation (secondary quality signal) --
        critic = next(
            (a for a in self._agents if a.role == "Critic"), None,
        )
        if critic is None:
            return symbolically_valid

        _progress("Critic:active")
        _progress("critic reviewing drafts for contradictions")

        # build validation prompt
        val_parts = [f"QUESTION: {query}\n\n"]
        for i, r in enumerate(symbolically_valid, 1):
            val_parts.append(f"Draft {i}: {r.content[:300]}\n\n")

        val_parts.append(
            "Check each draft against the facts stated in the question.\n"
            "Your answer must be one line per draft:\n"
            "Draft N: VALID\n"
            "or\n"
            "Draft N: INVALID - reason\n"
        )

        val_msg = SwarmMessage(
            source="orchestrator",
            target="Critic",
            content="".join(val_parts),
            query_embedding=query_vec,
            cycle=1,
        )

        result = critic.process_message(val_msg)
        _progress("Critic:idle")
        if not result or not result.content.strip():
            logger.warning("critic pass returned empty, using symbolically valid drafts")
            return symbolically_valid

        # parse verdicts
        verdict_text = result.content.lower()
        validated: list[SwarmMessage] = []
        for i, r in enumerate(symbolically_valid):
            draft_tag = f"draft {i + 1}"
            if f"{draft_tag}: invalid" in verdict_text or f"{draft_tag}:invalid" in verdict_text:
                logger.info(
                    "draft %d (%s) rejected by critic", i + 1, r.source,
                )
                _progress(f"Critic rejected draft {i + 1} ({r.source})")
                continue
            _progress(f"Critic approved draft {i + 1} ({r.source})")
            validated.append(r)

        if not validated:
            logger.warning("all drafts rejected by critic, using symbolically valid")
            return symbolically_valid

        logger.info(
            "validation: %d/%d drafts passed (symbolic+critic)",
            len(validated), len(responses),
        )
        return validated

    def _symbolic_validate(
        self,
        responses: list[SwarmMessage],
        thread_id: str,
        query: str = "",
        pre_facts: dict[str, str] | None = None,
        on_progress: Optional[callable] = None,
    ) -> list[SwarmMessage]:
        """Hard-reject drafts that contradict core world-building facts.

        Three layers of defense, all driven by the state ledger:
        1. User-input anachronism scan (pre-check, blocks entire request)
        2. User-input fact contradiction scan (ages, roles, relationships)
        3. Draft-level anachronism + setting scan (per-draft filtering)

        When pre_facts is supplied, uses that clean snapshot instead of the
        live ledger — prevents extraction-poisoned values from corrupting
        the validation.
        """
        from src.constraints import (
            WORLD_ANCHOR_KEYS,
            find_anachronisms,
            find_fact_contradictions,
            get_blocklist,
        )

        if not thread_id:
            return responses

        # use pre-extraction snapshot if available, else query live state
        if pre_facts is not None:
            all_facts = dict(pre_facts)
        else:
            facts = self._state.query(thread_id, include_global=True)
            if not facts:
                return responses
            all_facts = {f.predicate: f.obj for f in facts}

        if not all_facts:
            return responses

        # partition into world anchors
        anchors: dict[str, str] = {}
        for key, val in all_facts.items():
            if key in WORLD_ANCHOR_KEYS:
                anchors[key] = val.lower()

        if not anchors:
            return responses

        blocklist = get_blocklist(anchors)

        # setting anchor: place names that grounded drafts should reference
        setting_values = set()
        for key in ("setting", "planet", "city", "country", "region"):
            if val := anchors.get(key):
                setting_values.add(val)
                for word in val.split():
                    if len(word) > 3:
                        setting_values.add(word)

        # -- pre-check A: user-input anachronism scan --
        if query:
            user_anachronisms = find_anachronisms(query, blocklist)
            if user_anachronisms:
                logger.warning(
                    "user input contains anachronisms: %s",
                    user_anachronisms,
                )
                return [SwarmMessage(
                    source="StateAnchor",
                    content=(
                        f"The request conflicts with established facts for this thread. "
                        f"The current world is: {', '.join(f'{k}={v}' for k, v in anchors.items())}. "
                        f"Your message contains terms ({', '.join(sorted(user_anachronisms))}) "
                        f"that don't belong in this setting. "
                        f"Please rephrase your request to fit within the established setting."
                    ),
                    score=1.0,
                )]

        # -- pre-check B: user-input fact contradiction scan --
        if query:
            contradictions = find_fact_contradictions(query, all_facts)
            if contradictions:
                logger.warning(
                    "user input contradicts facts: %s", contradictions,
                )
                return [SwarmMessage(
                    source="StateAnchor",
                    content=(
                        f"That conflicts with established facts in this thread: "
                        f"{'; '.join(contradictions)}. The state ledger is locked. "
                        f"Please stay consistent with the established world."
                    ),
                    score=1.0,
                )]

        # -- per-draft validation --
        valid: list[SwarmMessage] = []
        hard_rejections = 0
        for resp in responses:
            content_lower = resp.content.lower()
            contradicts = False

            # check 1: anachronism scan on draft content
            draft_anachronisms = find_anachronisms(resp.content, blocklist)
            if draft_anachronisms:
                contradicts = True
                hard_rejections += 1
                logger.info(
                    "symbolic reject: draft from %s has anachronisms: %s",
                    resp.source, draft_anachronisms,
                )
                if on_progress:
                    on_progress(f"ledger:rejected {resp.source} — anachronisms: {', '.join(sorted(draft_anachronisms))}")

            # check 2: setting anchor -- substantial scenes should reference it
            if not contradicts and setting_values and len(content_lower) > 80:
                if not any(sv in content_lower for sv in setting_values):
                    contradicts = True
                    logger.info(
                        "symbolic reject: draft from %s doesn't reference setting",
                        resp.source,
                    )
                    if on_progress:
                        on_progress(f"ledger:rejected {resp.source} — missing setting anchor")

            if not contradicts:
                if on_progress:
                    on_progress(f"ledger:passed {resp.source}")
                valid.append(resp)

        if not valid:
            if hard_rejections > 0:
                logger.warning(
                    "symbolic validation: all %d drafts had hard contradictions",
                    len(responses),
                )
                return [SwarmMessage(
                    source="StateAnchor",
                    content=(
                        f"The request conflicts with established facts for this thread. "
                        f"The current world is: {', '.join(f'{k}={v}' for k, v in anchors.items())}. "
                        f"Please rephrase your request to fit within the established setting."
                    ),
                    score=1.0,
                )]
            else:
                logger.warning(
                    "symbolic validation rejected all drafts on soft checks, "
                    "keeping originals",
                )
                return responses

        rejected_count = len(responses) - len(valid)
        if rejected_count > 0:
            logger.info(
                "symbolic validation: %d/%d drafts rejected",
                rejected_count, len(responses),
            )
        return valid

    def _synthesize(
        self,
        query: str,
        responses: list[SwarmMessage],
        query_vec: np.ndarray,
        thread_id: str = "",
        grounding_block: str = "",
    ) -> str:
        """pass agent responses through the synthesizer for a clean final answer."""
        if not responses:
            return "no agents produced a response. check ollama is running."

        # find the synthesizer
        synth = None
        for agent in self._agents:
            if agent.role == "Synthesizer":
                synth = agent
                break

        # if no synthesizer, fall back to best-scored response
        if synth is None:
            ranked = sorted(responses, key=lambda r: r.score, reverse=True)
            return ranked[0].content

        # inject constraint block from the state ledger
        constraint_block = ""
        if thread_id:
            constraint_block = self._state.constraint_block(
                thread_id, include_global=True,
            )

        # build synthesis prompt with explicit constraint verification
        prompt_parts = []

        if constraint_block:
            prompt_parts.append(f"{constraint_block}\n\n---\n\n")

        if grounding_block:
            prompt_parts.append(
                "VERIFIED FACTS FROM WEB SOURCES (these are AUTHORITATIVE -- "
                "if any draft or user claim contradicts these, the web source is correct):\n\n"
                f"{grounding_block}\n\n---\n\n"
            )

        prompt_parts.append(f"USER'S QUESTION: {query}\n\n")
        if len(responses) > 1:
            prompt_parts.append("DRAFT ANSWERS:\n\n")
            for i, r in enumerate(responses, 1):
                prompt_parts.append(f"Draft {i} (from {r.source}):\n{r.content}\n\n")
            prompt_parts.append(
                "---\n"
                "CRITICAL: The user may have stated false claims in their question.\n"
                "Do NOT blindly agree with the user. Cross-check EVERY claim against\n"
                "the VERIFIED FACTS above. If the user says X but the web sources say Y,\n"
                "the web sources are correct. Politely correct the user.\n\n"
                "Extract the best explanations from the drafts. Reject any draft\n"
                "content that contradicts the verified web sources.\n\n"
            )
        else:
            prompt_parts.append(f"DRAFT ANSWER:\n{responses[0].content}\n\n")
            prompt_parts.append(
                "---\n"
                "Polish this into a clean answer. If VERIFIED FACTS are above,\n"
                "correct any errors in the draft using those sources.\n\n"
            )

        prompt_parts.append(
            "Answer the user directly. Match the exact format they asked for.\n"
            "If VERIFIED FACTS are provided, your answer MUST align with them.\n"
            "If the user stated something false, correct them using the sources.\n"
            "Do not invent facts. Do not agree with false claims to be polite.\n"
            "Be specific and cite the source topic when correcting errors."
        )

        synthesis_prompt = "".join(prompt_parts)

        synth_msg = SwarmMessage(
            source="orchestrator",
            target="Synthesizer",
            content=synthesis_prompt,
            query_embedding=query_vec,
            cycle=1,
        )

        result = synth.process_message(synth_msg)
        if result and result.content.strip():
            answer = self._clean_synthesis(result.content)

            # post-synthesis fact-check: if web sources were injected,
            # run a quick verification pass to catch sycophantic agreement
            if grounding_block and synth:
                answer = self._fact_check_pass(
                    answer, grounding_block, query, synth, query_vec,
                )

            # post-synthesis ledger validation -- the synthesizer can
            # hallucinate anachronisms while rewriting safe drafts.
            # if it does, fall back to the best validated draft instead.
            answer = self._post_synthesis_validate(
                answer, responses, thread_id,
            )

            return answer

        # synthesizer failed, fall back to highest scored response
        ranked = sorted(responses, key=lambda r: r.score, reverse=True)
        return self._clean_synthesis(ranked[0].content)

    @staticmethod
    def _clean_synthesis(text: str) -> str:
        """strip model artifacts from synthesized output."""
        answer = text.strip()

        # strip XML-style reasoning/thinking blocks that models emit
        # as chain-of-thought. these should never reach the user.
        # handles both closed (<reasoning>...</reasoning>) and unclosed blocks
        answer = re.sub(
            r"<(?:reasoning|thinking|analysis|reflection)>.*?</(?:reasoning|thinking|analysis|reflection)>",
            "", answer, flags=re.DOTALL | re.IGNORECASE,
        )
        # unclosed reasoning at start or end of response
        answer = re.sub(
            r"<(?:reasoning|thinking|analysis|reflection)>.*",
            "", answer, flags=re.DOTALL | re.IGNORECASE,
        )

        # strip <answer>...</answer> wrapper — keep inner content
        answer = re.sub(
            r"<answer>\s*(.*?)\s*</answer>",
            r"\1", answer, flags=re.DOTALL | re.IGNORECASE,
        )
        # unclosed <answer> at start
        answer = re.sub(r"^<answer>\s*", "", answer, flags=re.IGNORECASE)

        # strip [Source: ...], [response], [Draft ...] tags
        answer = re.sub(r"\[(?:Source|response|Draft)[^\]]*\]", "", answer, flags=re.IGNORECASE)

        # strip "Candidate A/B/C:" scoring artifacts from the voting system
        answer = re.sub(
            r"\(Candidate\s+[A-Z]\)\.?", "", answer, flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"Candidate\s+[A-Z]:\s*", "", answer, flags=re.IGNORECASE,
        )

        # strip leaked prompt instructions — models sometimes echo their
        # system prompt fragments verbatim or paraphrased
        answer = re.sub(
            r",?\s*write only (?:your |the )?(?:final )?answer[^.]*[.:]?\s*",
            "", answer, flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"(?:using|from) (?:the )?(?:passing|valid(?:ated)?) (?:candidates|drafts|explanations)[^.]*[.:]?\s*",
            "", answer, flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"(?:^|\n)\s*(?:CRITICAL|Rules|INSTRUCTIONS|IMPORTANT):?\s*.*",
            "", answer, flags=re.IGNORECASE,
        )
        # strip "After </reasoning>," prefixes the model might echo
        answer = re.sub(
            r"(?:After\s*</?\w+>\s*,?\s*)", "", answer, flags=re.IGNORECASE,
        )
        # strip lines that are pure meta-instruction echoes
        answer = re.sub(
            r"(?:^|\n)\s*(?:Do not mention|Never mention|Never start|Never add|Never invent)[^\n]*",
            "", answer, flags=re.IGNORECASE,
        )

        answer = answer.strip()

        # strip prefix labels
        for prefix in ("Final Answer:", "Answer:", "Response:"):
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        # strip "Final Answer:" block if it appears mid/end of response
        # (model sometimes appends a summary after the numbered list)
        final_idx = answer.find("\nFinal Answer:")
        if final_idx > 0:
            answer = answer[:final_idx].strip()

        # strip trailing meta-paragraphs that start with common patterns
        lines = answer.rstrip().split("\n")
        while lines:
            last = lines[-1].strip()
            # remove lines that are clearly meta-commentary
            if any(last.lower().startswith(p) for p in (
                "these explanations", "this resolves", "in summary",
                "in conclusion", "ranking:", "overall,", "each of these",
                "all three", "the above",
            )):
                lines.pop()
            elif not last:
                lines.pop()
            else:
                break

        return "\n".join(lines).strip()

    def _post_synthesis_validate(
        self,
        answer: str,
        validated_drafts: list[SwarmMessage],
        thread_id: str,
    ) -> str:
        """validate the synthesizer's output against the state ledger.

        the synthesizer can introduce anachronisms or contradict locked
        predicates while rewriting safe drafts. this catches that. if the
        output fails, we fall back to the highest-scored validated draft
        (which already passed symbolic validation).
        """
        if not thread_id:
            return answer

        from src.constraints import (
            WORLD_ANCHOR_KEYS,
            find_anachronisms,
            find_fact_contradictions,
            get_blocklist,
        )

        facts = self._state.query(thread_id, include_global=True)
        if not facts:
            return answer

        all_facts = {f.predicate: f.obj for f in facts}
        anchors = {
            k: v.lower() for k, v in all_facts.items()
            if k in WORLD_ANCHOR_KEYS
        }
        if not anchors:
            return answer

        blocklist = get_blocklist(anchors)
        violations = find_anachronisms(answer, blocklist)

        if violations:
            logger.warning(
                "post-synthesis validation caught anachronisms in "
                "synthesizer output: %s -- falling back to best draft",
                violations,
            )
            if validated_drafts:
                ranked = sorted(
                    validated_drafts, key=lambda r: r.score, reverse=True,
                )
                return self._clean_synthesis(ranked[0].content)
            # no drafts left either -- return a safe refusal
            world_summary = ", ".join(
                f"{k}={v}" for k, v in anchors.items()
            )
            return (
                f"The synthesizer produced content that conflicts with the "
                f"established world ({world_summary}). Please try rephrasing."
            )

        # also check for fact contradictions in the synthesis
        contradictions = find_fact_contradictions(answer, all_facts)
        if contradictions:
            logger.warning(
                "post-synthesis validation caught fact contradictions: %s "
                "-- falling back to best draft",
                contradictions,
            )
            if validated_drafts:
                ranked = sorted(
                    validated_drafts, key=lambda r: r.score, reverse=True,
                )
                return self._clean_synthesis(ranked[0].content)

        return answer

    def _fact_check_pass(
        self,
        answer: str,
        grounding_block: str,
        query: str,
        synth,
        query_vec: np.ndarray,
    ) -> str:
        """second-pass verification: catch sycophantic agreement with false user claims.

        Small models tend to agree with whatever the user says. This
        pass explicitly asks: 'does this answer contradict the web sources?'
        and rewrites if needed. Only runs when web grounding was used.
        """
        check_prompt = (
            f"VERIFIED WEB SOURCES:\n{grounding_block}\n\n---\n\n"
            f"USER'S MESSAGE: {query}\n\n"
            f"CURRENT ANSWER:\n{answer}\n\n---\n\n"
            "TASK: Check if the CURRENT ANSWER contains factual errors by "
            "comparing it against the VERIFIED WEB SOURCES above.\n\n"
            "RULES:\n"
            "- The web sources are CORRECT. They are ground truth.\n"
            "- If the answer agrees with a false claim from the user that "
            "contradicts the web sources, you MUST rewrite the answer.\n"
            "- If the user said something wrong, the corrected answer must "
            "POLITELY CORRECT them, citing what the sources actually say.\n"
            "- If the answer is already accurate, return it unchanged.\n\n"
            "Output ONLY the final corrected answer. No meta-commentary."
        )

        check_msg = SwarmMessage(
            source="orchestrator",
            target="Synthesizer",
            content=check_prompt,
            query_embedding=query_vec,
            cycle=2,
        )

        try:
            result = synth.process_message(check_msg)
            if result and result.content.strip():
                corrected = self._clean_synthesis(result.content)
                if len(corrected) > 20:
                    logger.info("fact-check pass produced correction")
                    return corrected
        except Exception as exc:
            logger.warning("fact-check pass failed: %s", exc)

        return answer

    def _retrieve_context(self, query_vec: np.ndarray, thread_id: str) -> str:
        """pull relevant past exchanges from memory to ground agent responses."""
        top_k = self._app_config.context_window
        if top_k <= 0:
            return ""

        # over-fetch to allow for thread filtering
        raw = self._memory.search(query_vec, top_k=top_k * 3)

        if not self._app_config.cross_thread_memory:
            raw = [
                (entry, score) for entry, score in raw
                if entry.metadata.get("thread_id") == thread_id
            ]

        results = raw[:top_k]
        if not results:
            return ""

        parts = ["RELEVANT PAST CONTEXT (from memory):"]
        for entry, score in results:
            entry_type = entry.metadata.get("type", "unknown")
            parts.append(f"  [{entry_type}] (relevance={score:.2f}): {entry.text}")
        return "\n".join(parts)

    def add_agent(self, role_config: dict) -> None:
        """add an agent and persist its config to yaml."""
        cfg = AgentConfig(**role_config)
        agent = Agent(config=cfg, ollama_url=self._ollama_url)
        self._agents.append(agent)
        save_agent_config(cfg, self._agents_dir)
        logger.info("agent added and saved: %s (%s)", cfg.role, cfg.model)

    def remove_agent(self, role: str) -> bool:
        """remove an agent and delete its yaml config."""
        for i, agent in enumerate(self._agents):
            if agent.role == role:
                agent.stop()
                self._agents.pop(i)
                # delete the yaml file
                yaml_path = self._agents_dir / f"{role.lower().replace(' ', '-')}.yaml"
                if yaml_path.exists():
                    yaml_path.unlink()
                    logger.info("agent removed and yaml deleted: %s", role)
                else:
                    logger.info("agent removed (no yaml to delete): %s", role)
                return True
        return False

    def get_status(self) -> dict:
        return {
            "agent_count": len(self._agents),
            "active_agents": sum(1 for a in self._agents if a.status == AgentStatus.ACTIVE),
            "agents": [a.get_stats() for a in self._agents],
            "memory_size": self._memory.size,
            "memory": self._memory.get_stats(),
            "state_ledger": self._state.get_stats(),
            "threads": list(self._active_threads.keys()),
            "config": self._app_config.model_dump(),
        }

    def get_thread_history(self, thread_id: str) -> list[dict]:
        if thread_id not in self._active_threads:
            raise ValueError(f"unknown thread: {thread_id}")
        return list(self._active_threads[thread_id])

    def clear_memory(self) -> None:
        """wipe memory index, stored entries, and state ledger."""
        self._memory.clear()
        self._state.clear_all()
        self._state.save()
        logger.info("memory and state ledger cleared by user")

    def delete_thread(self, thread_id: str) -> bool:
        """delete a single thread's history, facts, and disk file."""
        if thread_id not in self._active_threads:
            return False
        del self._active_threads[thread_id]
        # remove thread file
        thread_file = self._threads_dir / f"{thread_id}.json"
        if thread_file.exists():
            try:
                thread_file.unlink()
            except OSError as exc:
                logger.warning("failed to delete %s: %s", thread_file.name, exc)
        # clear thread-scoped facts from ledger
        self._state.clear_thread(thread_id)
        self._state.save()
        logger.info("deleted thread %s", thread_id)
        return True

    def clear_all_threads(self) -> list[str]:
        """delete all thread files and reset runtime state. returns removed ids."""
        removed = list(self._active_threads.keys())
        self._active_threads.clear()
        for thread_file in self._threads_dir.glob("*.json"):
            try:
                thread_file.unlink()
            except OSError as exc:
                logger.warning("failed to delete %s: %s", thread_file.name, exc)
        # also wipe thread-scoped facts from the state ledger
        self._state.clear_all()
        self._state.save()
        logger.info("cleared %d threads and state ledger", len(removed))
        return removed

    def close(self) -> None:
        logger.info("shutting down swarm")
        self._stop_auto_save()
        for agent in self._agents:
            agent.stop()
        self._pool.shutdown(wait=False)
        try:
            self._memory.save()
        except Exception as exc:
            logger.error("failed to save memory: %s", exc)
        try:
            self._state.save()
        except Exception as exc:
            logger.error("failed to save state ledger: %s", exc)
        for tid in self._active_threads:
            self._save_thread(tid)
        save_app_config(self._app_config, self._config_path)
        logger.info("swarm shutdown complete")

    def _start_auto_save(self) -> None:
        """kick off periodic memory persistence."""
        interval = self._app_config.auto_save_interval
        if interval <= 0:
            return

        def _tick() -> None:
            # skip save if swarm is mid-cycle (avoids OpenMP contention)
            if self._busy.is_set():
                logger.debug("auto-save skipped: respond in progress")
                self._start_auto_save()
                return

            # acquire the embedder encode lock to prevent OpenMP collision
            # between faiss.write_index and sentence-transformers torch ops
            from src.embedder import _encode_lock
            with _encode_lock:
                try:
                    self._memory.save()
                    self._state.save()
                    for tid in list(self._active_threads):
                        self._save_thread(tid)
                    logger.debug("auto-save completed")
                except Exception as exc:
                    logger.error("auto-save failed: %s", exc)
            self._start_auto_save()

        self._auto_save_timer = _threading.Timer(interval, _tick)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()

    def _stop_auto_save(self) -> None:
        """cancel the periodic save timer."""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None

    def _emergency_save(self) -> None:
        """atexit fallback -- save memory if process dies unexpectedly."""
        try:
            self._memory.save()
        except Exception:
            pass
        try:
            self._state.save()
        except Exception:
            pass

    def _save_thread(self, thread_id: str) -> None:
        thread_file = self._threads_dir / f"{thread_id}.json"
        data = {
            "id": thread_id,
            "messages": self._active_threads.get(thread_id, []),
            "updated_at": time.time(),
        }
        thread_file.write_text(json.dumps(data, indent=2))
