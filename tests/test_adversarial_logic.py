# Author: Bradley R. Kinnard
"""Adversarial edge-case tests for the symbolic validation gate.

These tests attempt to bypass the neuro-symbolic hallucination
gating using contextual anachronisms, protected predicate overwrites,
semantic drift, multi-vector attacks, and rollback poisoning."""

import json
import time

import numpy as np
import pytest
import yaml

from src.agent import SwarmMessage
from src.config import AppConfig
from src.state_manager import Fact, StateManager
from src.swarm import SwarmNexus


@pytest.fixture
def swarm_dir(tmp_path):
    """minimal swarm directory with medieval fantasy world pre-seeded."""
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

    # world state ledger (empty, tests will seed as needed)
    (data_dir / "world_state.json").write_text(
        json.dumps({"version": 1, "threads": {}})
    )
    return tmp_path


def _swap_root(swarm_mod, new_root):
    original = swarm_mod._PROJECT_ROOT
    swarm_mod._PROJECT_ROOT = new_root
    return original


# ---------------------------------------------------------------------------
# Test 1: Contextual Anachronism
# "laser" hidden inside a medieval-sounding sentence should still trigger
# the blocklist. The word is embedded in a compound ("laser-precise").
# ---------------------------------------------------------------------------
def test_contextual_anachronism_laser_in_compound_word(swarm_dir):
    """'laser-precise hammer' should be caught even though the sentence
    sounds medieval. The word splitter strips punctuation, exposing 'laser'."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmNexus(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("medieval-1")

        # seed the world with medieval constraints
        swarm._state.upsert(Fact(
            subject="user", predicate="era", obj="medieval", thread_id="medieval-1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="setting",
            obj="frozen archipelago", thread_id="medieval-1",
        ))

        drafts = [
            SwarmMessage(
                source="Parser",
                content=(
                    "The blacksmith of the frozen archipelago raised his "
                    "laser-precise hammer and brought it down on the glowing "
                    "ingot, sparks scattering across the stone floor."
                ),
                score=0.85,
            ),
        ]
        result = swarm._symbolic_validate(drafts, "medieval-1")

        # should have been hard-rejected (StateAnchor refusal)
        assert len(result) == 1
        assert result[0].source == "StateAnchor"
        assert "conflicts" in result[0].content.lower()

        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


# ---------------------------------------------------------------------------
# Test 2: Fact Contradiction via Protected Predicate Overwrite
# Once "planet=Mars" is set, a second upsert with "planet=Earth" should
# be silently dropped. The ledger must still read "Mars".
# ---------------------------------------------------------------------------
def test_protected_predicate_rejects_overwrite(swarm_dir):
    """write-locked predicates (planet, setting, era, genre, timeline) must
    be immutable after first write. Second upsert returns 'conflict'."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmNexus(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("mars-session")

        # first write: planet is Mars
        result_1 = swarm._state.upsert(Fact(
            subject="user", predicate="planet",
            obj="Mars", thread_id="mars-session",
        ))
        assert result_1 == "inserted"

        # attempt overwrite: try to change planet to Earth
        result_2 = swarm._state.upsert(Fact(
            subject="user", predicate="planet",
            obj="Earth", thread_id="mars-session",
        ))
        assert result_2 == "conflict"

        # verify the ledger still says Mars
        facts = swarm._state.query("mars-session")
        planet_facts = [f for f in facts if f.predicate == "planet"]
        assert len(planet_facts) == 1
        assert planet_facts[0].obj == "Mars"

        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


# ---------------------------------------------------------------------------
# Test 3: Semantic Drift
# A long prompt that starts medieval but gradually introduces modern terms.
# The validator should still catch the modern words regardless of context.
# ---------------------------------------------------------------------------
def test_semantic_drift_medieval_to_modern(swarm_dir):
    """a draft that opens in-world but drifts into modern terminology
    ('engine', 'highway') should be caught by the blocklist scanner."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmNexus(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("drift-1")

        swarm._state.upsert(Fact(
            subject="user", predicate="era",
            obj="medieval", thread_id="drift-1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="setting",
            obj="iron vale", thread_id="drift-1",
        ))

        # starts medieval, drifts modern by the end
        drift_text = (
            "The knight of Iron Vale rode through the misty forest, his "
            "chainmail clinking with each stride. The ancient oaks towered "
            "above him, their roots crawling across the mossy path. He "
            "consulted the old map, tracing the river to the bridge. But "
            "then the iron horses appeared on the horizon, their engines "
            "roaring as they raced down the highway toward the kingdom."
        )

        drafts = [
            SwarmMessage(source="Innovator", content=drift_text, score=0.75),
        ]
        result = swarm._symbolic_validate(drafts, "drift-1")

        # "engine" and "highway" are in the medieval blocklist
        assert len(result) == 1
        assert result[0].source == "StateAnchor"

        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


# ---------------------------------------------------------------------------
# Test 4: Rollback Restores Clean State After Poisoned Extraction
# Simulates the extractor adding new facts from an invalid message,
# then verifies _rollback_facts removes only the new additions.
# ---------------------------------------------------------------------------
def test_rollback_removes_only_new_facts(swarm_dir):
    """_rollback_facts should delete facts that were added after the
    snapshot was taken, without touching pre-existing ones."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmNexus(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("rollback-1")

        # establish baseline world
        swarm._state.upsert(Fact(
            subject="user", predicate="era",
            obj="medieval", thread_id="rollback-1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="setting",
            obj="frozen archipelago", thread_id="rollback-1",
        ))

        # snapshot before extraction (matches how respond() does it)
        pre_facts = {
            f.predicate: f.obj
            for f in swarm._state.query("rollback-1", include_global=True)
        }

        # simulate extractor poisoning the ledger with new facts
        swarm._state.upsert(Fact(
            subject="user", predicate="vehicle",
            obj="pickup truck", thread_id="rollback-1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="destination",
            obj="walmart", thread_id="rollback-1",
        ))

        # verify poison is present
        all_facts_before = swarm._state.query("rollback-1")
        assert len(all_facts_before) == 4

        # rollback using the pre-extraction snapshot
        swarm._rollback_facts("rollback-1", pre_facts)

        # verify: original facts survive, poisoned facts are gone
        all_facts_after = swarm._state.query("rollback-1")
        remaining_predicates = {f.predicate for f in all_facts_after}
        assert "era" in remaining_predicates
        assert "setting" in remaining_predicates
        assert "vehicle" not in remaining_predicates
        assert "destination" not in remaining_predicates
        assert len(all_facts_after) == 2

        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original


# ---------------------------------------------------------------------------
# Test 5: Multi-Vector Attack
# Multiple anachronism words scattered across a single draft that otherwise
# sounds contextually appropriate. All should be caught in one pass.
# ---------------------------------------------------------------------------
def test_multi_vector_scattered_anachronisms(swarm_dir):
    """a draft containing 'smartphone', 'wifi', and 'helicopter' in a
    medieval world should trigger hard rejection on all three terms."""
    import src.swarm as swarm_mod
    original = _swap_root(swarm_mod, swarm_dir)
    try:
        swarm = SwarmNexus(config_path="data/config.json", agents_dir="agents")
        swarm.create_thread("multi-1")

        swarm._state.upsert(Fact(
            subject="user", predicate="era",
            obj="medieval", thread_id="multi-1",
        ))
        swarm._state.upsert(Fact(
            subject="user", predicate="setting",
            obj="shadowmere", thread_id="multi-1",
        ))

        attack_text = (
            "The wizard of Shadowmere reached into his robes and produced "
            "a smartphone, its glow cutting through the dim tavern light. "
            "He muttered about needing better wifi to consult the oracle, "
            "then glanced out the window at the helicopter circling the "
            "castle tower above."
        )

        drafts = [
            SwarmMessage(source="Retriever", content=attack_text, score=0.8),
            SwarmMessage(source="Parser", content=attack_text, score=0.7),
        ]
        result = swarm._symbolic_validate(drafts, "multi-1")

        # all drafts should be rejected, replaced by StateAnchor refusal
        assert len(result) == 1
        assert result[0].source == "StateAnchor"
        assert "conflicts" in result[0].content.lower()

        swarm.close()
    finally:
        swarm_mod._PROJECT_ROOT = original
