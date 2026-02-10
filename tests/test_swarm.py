# Author: Bradley R. Kinnard
"""Tests for the SwarmChatbot orchestrator."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import AppConfig
from src.swarm import SwarmChatbot


@pytest.fixture
def swarm_dir(tmp_path):
    """set up a minimal swarm directory structure for testing."""
    # agents
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    for role in ["Parser", "Critic"]:
        data = {
            "role": role,
            "model": "phi3:mini",
            "prompt": f"you are a {role.lower()}",
            "max_tokens": 64,
            "score_threshold": 0.5,
        }
        (agents_dir / f"{role.lower()}.yaml").write_text(yaml.dump(data))

    # fact-extractor agent
    extractor_data = {
        "role": "Fact-Extractor",
        "model": "phi3:mini",
        "prompt": "extract facts as JSON",
        "max_tokens": 256,
        "temperature": 0.1,
        "score_threshold": 0.0,
    }
    (agents_dir / "fact_extractor.yaml").write_text(yaml.dump(extractor_data))

    # config
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
        "auto_save_interval": 0,
    }
    (data_dir / "config.json").write_text(json.dumps(config))

    # world state ledger
    (data_dir / "world_state.json").write_text(
        json.dumps({"version": 1, "threads": {}})
    )
    return tmp_path


def _swap_root(swarm_mod, new_root):
    original = swarm_mod._PROJECT_ROOT
    swarm_mod._PROJECT_ROOT = new_root
    return original


def test_swarm_init(swarm_dir):
    """swarm should load agents on init (parser + critic + fact-extractor)."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(
            config_path="data/config.json",
            agents_dir="agents",
        )
        assert len(swarm._agents) == 3
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_create_thread(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        tid = swarm.create_thread("test-thread")
        assert tid == "test-thread"
        assert "test-thread" in swarm._active_threads
        thread_file = swarm_dir / "data" / "threads" / "test-thread.json"
        assert thread_file.exists()
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_create_thread_empty_id(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        with pytest.raises(ValueError, match="empty"):
            swarm.create_thread("")
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_respond_unknown_thread(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        with pytest.raises(ValueError, match="unknown thread"):
            swarm.respond("hello", "nonexistent")
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_respond_empty_query(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")
        with pytest.raises(ValueError, match="empty"):
            swarm.respond("", "t1")
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_get_status(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        status = swarm.get_status()
        assert status["agent_count"] == 3
        assert "agents" in status
        assert "memory_size" in status
        assert "state_ledger" in status
        assert status["state_ledger"]["total_facts"] == 0
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_add_agent_dynamic(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.add_agent({
            "role": "Innovator",
            "model": "gemma2:2b",
            "prompt": "be creative",
        })
        assert len(swarm._agents) == 4
        # verify yaml was persisted
        yaml_path = swarm_dir / "agents" / "innovator.yaml"
        assert yaml_path.exists()
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_add_agent_survives_restart(swarm_dir):
    """dynamically added agent should be loaded on next init."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        s1 = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        s1.add_agent({"role": "Tester", "model": "phi3:mini", "prompt": "test"})
        s1.close()

        s2 = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        roles = [a.role for a in s2._agents]
        assert "Tester" in roles
        s2.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_remove_agent(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        removed = swarm.remove_agent("Parser")
        assert removed is True
        assert len(swarm._agents) == 2
        # verify yaml was deleted
        yaml_path = swarm_dir / "agents" / "parser.yaml"
        assert not yaml_path.exists()
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_remove_agent_survives_restart(swarm_dir):
    """removed agent should stay gone on next init."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        s1 = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        s1.remove_agent("Critic")
        s1.close()

        s2 = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        roles = [a.role for a in s2._agents]
        assert "Critic" not in roles
        assert len(s2._agents) == 2
        s2.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_remove_nonexistent_agent(swarm_dir):
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        removed = swarm.remove_agent("FakeAgent")
        assert removed is False
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_fact_extractor_excluded_from_answerers(swarm_dir):
    """fact-extractor should not appear in the answerer list."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        answerers = [
            a for a in swarm._agents
            if a.role not in ("Synthesizer", "Fact-Extractor")
        ]
        roles = [a.role for a in answerers]
        assert "Fact-Extractor" not in roles
        assert "Parser" in roles
        assert "Critic" in roles
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_fact_extractor_agent_loaded(swarm_dir):
    """fact-extractor agent should be present after init."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        roles = [a.role for a in swarm._agents]
        assert "Fact-Extractor" in roles
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_state_ledger_initialized(swarm_dir):
    """state ledger should init empty and be accessible."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        assert swarm._state.size == 0
        assert swarm._state.thread_count == 0
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_clear_memory_also_clears_state(swarm_dir):
    """clear_memory should wipe both FAISS memory and the state ledger."""
    import src.swarm as swarm_mod
    from src.state_manager import Fact
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        # manually insert a fact
        swarm._state.upsert(Fact(
            subject="test", predicate="val", obj="123", thread_id="t1",
        ))
        assert swarm._state.size == 1
        swarm.clear_memory()
        assert swarm._state.size == 0
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_run_fact_extraction_parses_json(swarm_dir):
    """_run_fact_extraction should parse JSON and upsert facts."""
    import src.swarm as swarm_mod
    from unittest.mock import patch, MagicMock
    from src.agent import SwarmMessage
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")

        # mock the fact-extractor's process_message to return valid JSON
        mock_result = SwarmMessage(
            source="Fact-Extractor",
            content='{"setting": "Mars", "timeline": "2099"}',
        )
        extractor = next(a for a in swarm._agents if a.role == "Fact-Extractor")
        with patch.object(extractor, "process_message", return_value=mock_result):
            swarm._run_fact_extraction(
                "I am planning a story set on Mars in the year 2099 with robots",
                "t1",
            )

        assert swarm._state.size == 2
        facts = swarm._state.query("t1")
        predicates = {f.predicate for f in facts}
        assert "setting" in predicates
        assert "timeline" in predicates
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_run_fact_extraction_handles_bad_json(swarm_dir):
    """_run_fact_extraction should not crash on malformed output."""
    import src.swarm as swarm_mod
    from unittest.mock import patch
    from src.agent import SwarmMessage
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")

        mock_result = SwarmMessage(
            source="Fact-Extractor",
            content="this is not json at all",
        )
        extractor = next(a for a in swarm._agents if a.role == "Fact-Extractor")
        with patch.object(extractor, "process_message", return_value=mock_result):
            swarm._run_fact_extraction("hello", "t1")  # should not raise

        assert swarm._state.size == 0
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_run_fact_extraction_handles_markdown_fenced_json(swarm_dir):
    """_run_fact_extraction strips markdown fences around JSON."""
    import src.swarm as swarm_mod
    from unittest.mock import patch
    from src.agent import SwarmMessage
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")

        mock_result = SwarmMessage(
            source="Fact-Extractor",
            content='```json\n{"genre": "sci-fi"}\n```',
        )
        extractor = next(a for a in swarm._agents if a.role == "Fact-Extractor")
        with patch.object(extractor, "process_message", return_value=mock_result):
            swarm._run_fact_extraction(
                "I am writing a sci-fi story about a starship crew exploring deep space",
                "t1",
            )

        assert swarm._state.size == 1
        facts = swarm._state.query("t1")
        assert facts[0].obj == "sci-fi"
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_extract_json_object_clean():
    """_extract_json_object handles clean JSON."""
    result = SwarmChatbot._extract_json_object('{"setting": "Mars"}')
    assert result == {"setting": "Mars"}


def test_extract_json_object_trailing_text():
    """_extract_json_object handles JSON followed by model commentary."""
    raw = '{"setting": "Mars", "year": "2187"}\n\nThe above JSON represents...'
    result = SwarmChatbot._extract_json_object(raw)
    assert result == {"setting": "Mars", "year": "2187"}


def test_extract_json_object_multiline_trailing():
    """_extract_json_object handles multiline JSON with trailing junk."""
    raw = (
        '{\n  "genre": "cyberpunk",\n  "location": "Mars"\n}\n'
        'I extracted the key facts from your message.\n'
        'Here is what I found...'
    )
    result = SwarmChatbot._extract_json_object(raw)
    assert result == {"genre": "cyberpunk", "location": "Mars"}


def test_extract_json_object_no_json():
    """_extract_json_object returns None for non-JSON text."""
    assert SwarmChatbot._extract_json_object("no json here") is None


def test_extract_json_object_empty_dict():
    """_extract_json_object returns None for empty dict."""
    assert SwarmChatbot._extract_json_object("{}") is None


def test_extract_json_object_leading_text():
    """_extract_json_object ignores leading text before JSON."""
    raw = 'Here are the facts: {"name": "Kael"}'
    result = SwarmChatbot._extract_json_object(raw)
    assert result == {"name": "Kael"}


def test_synthesize_includes_constraint_block(swarm_dir):
    """_synthesize should prepend state ledger constraints when facts exist."""
    import src.swarm as swarm_mod
    from unittest.mock import patch, MagicMock
    from src.agent import SwarmMessage
    from src.state_manager import Fact
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")

        # seed the state ledger with facts
        swarm._state.upsert(Fact(
            subject="setting", predicate="location", obj="Mars", thread_id="t1",
        ))
        swarm._state.upsert(Fact(
            subject="timeline", predicate="year", obj="2099", thread_id="t1",
        ))

        # add a fake synthesizer so _synthesize doesn't just return best draft
        from src.config import AgentConfig
        from src.agent import Agent
        synth_cfg = AgentConfig(
            role="Synthesizer", model="phi3:mini",
            prompt="synthesize", max_tokens=64, score_threshold=0.0,
        )
        synth_agent = Agent(config=synth_cfg, ollama_url="http://localhost:11434")
        swarm._agents.append(synth_agent)

        # capture what gets sent to the synthesizer
        captured_prompts = []
        def mock_process(msg):
            captured_prompts.append(msg.content)
            return SwarmMessage(
                source="Synthesizer", content="Mars in 2099", score=0.9,
            )

        with patch.object(synth_agent, "process_message", side_effect=mock_process):
            draft = SwarmMessage(source="Parser", content="draft answer", score=0.8)
            query_vec = np.zeros(384, dtype=np.float32)
            result = swarm._synthesize("tell me about Mars", [draft], query_vec, "t1")

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "ESTABLISHED FACTS" in prompt
        assert "Mars" in prompt
        assert "2099" in prompt
        assert "must not contradict" in prompt.lower()
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_synthesize_no_constraints_when_no_facts(swarm_dir):
    """_synthesize should NOT include constraint block when ledger is empty."""
    import src.swarm as swarm_mod
    from unittest.mock import patch
    from src.agent import SwarmMessage
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")

        from src.config import AgentConfig
        from src.agent import Agent
        synth_cfg = AgentConfig(
            role="Synthesizer", model="phi3:mini",
            prompt="synthesize", max_tokens=64, score_threshold=0.0,
        )
        synth_agent = Agent(config=synth_cfg, ollama_url="http://localhost:11434")
        swarm._agents.append(synth_agent)

        captured_prompts = []
        def mock_process(msg):
            captured_prompts.append(msg.content)
            return SwarmMessage(
                source="Synthesizer", content="clean answer", score=0.9,
            )

        with patch.object(synth_agent, "process_message", side_effect=mock_process):
            draft = SwarmMessage(source="Parser", content="draft", score=0.7)
            query_vec = np.zeros(384, dtype=np.float32)
            result = swarm._synthesize("hello", [draft], query_vec, "t1")

        prompt = captured_prompts[0]
        assert "ESTABLISHED FACTS" not in prompt
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_symbolic_validate_passes_consistent_drafts(swarm_dir):
    """drafts consistent with the ledger should pass symbolic validation."""
    import src.swarm as swarm_mod
    from src.agent import SwarmMessage
    from src.state_manager import Fact
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")
        swarm._state.upsert(Fact(
            subject="user", predicate="setting", obj="Mars", thread_id="t1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="genre", obj="sci-fi", thread_id="t1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="era", obj="futuristic", thread_id="t1",
        ))

        drafts = [
            SwarmMessage(source="Parser", content="The story is set on Mars. The red planet stretches before us with its vast deserts and ancient craters forming a backdrop to our narrative.", score=0.8),
            SwarmMessage(source="Reasoner", content="Mars is the primary location. The Martian landscape provides a unique setting for this cyberpunk tale.", score=0.7),
        ]
        result = swarm._symbolic_validate(drafts, "t1")
        assert len(result) == 2  # both mention Mars, both pass
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_symbolic_validate_rejects_anachronisms(swarm_dir):
    """drafts with modern tech in a medieval setting should be rejected."""
    import src.swarm as swarm_mod
    from src.agent import SwarmMessage
    from src.state_manager import Fact
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")
        swarm._state.upsert(Fact(
            subject="user", predicate="setting", obj="Shattered Reach", thread_id="t1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="era", obj="medieval", thread_id="t1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="genre", obj="fantasy", thread_id="t1",
        ))

        good_draft = SwarmMessage(
            source="Parser",
            content="Elara navigates the frozen streets of the Shattered Reach using her echolocation magic. The harbor is filled with wooden longships and the air smells of salt and pine tar.",
            score=0.8,
        )
        bad_draft = SwarmMessage(
            source="Reasoner",
            content="Elara drives her pickup truck down the highway to the nearest Walmart. She parks in the lot and grabs a shopping cart to load up on supplies for the quest.",
            score=0.7,
        )
        result = swarm._symbolic_validate([good_draft, bad_draft], "t1")
        assert len(result) == 1
        assert result[0].source == "Parser"
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_symbolic_validate_rejects_missing_setting(swarm_dir):
    """drafts that ignore the anchor setting should be rejected."""
    import src.swarm as swarm_mod
    from src.agent import SwarmMessage
    from src.state_manager import Fact
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")
        swarm._state.upsert(Fact(
            subject="user", predicate="setting", obj="Tokyo", thread_id="t1",
        ))

        drafts = [
            SwarmMessage(source="Parser", content="Ren wakes up in his apartment in Tokyo. The neon lights of Shibuya flash outside his window as he stretches and checks his neural implant.", score=0.8),
            SwarmMessage(source="Reasoner", content="Ren starts his day in rural Kansas. The wheat fields sway gently in the morning breeze as he walks to his barn and begins his routine farm chores.", score=0.7),
        ]
        result = swarm._symbolic_validate(drafts, "t1")
        assert len(result) == 1
        assert result[0].source == "Parser"
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_symbolic_validate_returns_all_when_no_facts(swarm_dir):
    """with empty ledger, symbolic validation should pass everything through."""
    import src.swarm as swarm_mod
    from src.agent import SwarmMessage
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")

        drafts = [
            SwarmMessage(source="Parser", content="Some text about Earth", score=0.8),
        ]
        result = swarm._symbolic_validate(drafts, "t1")
        assert len(result) == 1
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_symbolic_validate_no_thread_id(swarm_dir):
    """with no thread_id, symbolic validation should pass everything."""
    import src.swarm as swarm_mod
    from src.agent import SwarmMessage
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        drafts = [
            SwarmMessage(source="Parser", content="anything", score=0.5),
        ]
        result = swarm._symbolic_validate(drafts, "")
        assert len(result) == 1
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


def test_validate_drafts_uses_thread_id(swarm_dir):
    """_validate_drafts should accept and pass thread_id to symbolic validation."""
    import src.swarm as swarm_mod
    from src.agent import SwarmMessage
    from src.state_manager import Fact
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmChatbot(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("t1")
        swarm._state.upsert(Fact(
            subject="user", predicate="setting", obj="Mars", thread_id="t1",
        ))

        drafts = [
            SwarmMessage(source="Parser", content="Mars is a great setting for this story. The red landscape stretches endlessly beneath the dome of the colony, casting long shadows in the dim light.", score=0.8),
            SwarmMessage(source="Reasoner", content="Mars exploration is central to the plot. The colony on Mars provides both isolation and wonder as the characters navigate its harsh terrain.", score=0.7),
        ]
        query_vec = np.zeros(384, dtype=np.float32)
        # no critic available (it needs ollama), so it falls through after symbolic
        result = swarm._validate_drafts("about Mars", drafts, query_vec, "t1")
        # both drafts are consistent with the ledger
        assert len(result) >= 1
        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original
