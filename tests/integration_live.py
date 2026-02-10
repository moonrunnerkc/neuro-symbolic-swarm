# Author: Bradley R. Kinnard
"""Live integration test for Neuro-Symbolic State Anchoring.

Runs against real Ollama models. NOT a unit test -- requires:
  - ollama serve running
  - phi3:mini, dolphin-mistral:7b, dolphin-llama3:8b pulled

Usage:
  python -m tests.integration_live
"""

import json
import os
import sys
import time

# force single-thread FAISS
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

# ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.swarm import SwarmChatbot


def separator(label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")


def main() -> None:
    # reset state
    state_path = ROOT / "data" / "world_state.json"
    state_path.write_text(json.dumps({"version": 1, "threads": {}}))

    separator("INITIALIZING SWARM")
    swarm = SwarmChatbot()
    tid = swarm.create_thread("live-test")
    print(f"thread: {tid}")
    print(f"agents: {[a.role for a in swarm._agents]}")
    print(f"state ledger: {swarm._state.size} facts")

    results = {}

    # ----------------------------------------------------------------
    # TEST 1: fact extraction from structured input
    # ----------------------------------------------------------------
    separator("TEST 1: Fact Extraction")
    q1 = (
        "I'm writing a noir thriller set in Tokyo in 2045. "
        "The lead character is a 35-year-old hacker named Ren Tanaka. "
        "He works for a shadow corporation called NeoKai Industries."
    )
    print(f"INPUT: {q1}\n")

    t0 = time.time()
    r1 = swarm.respond(q1, tid)
    t1 = time.time()

    print(f"RESPONSE ({t1 - t0:.1f}s):\n{r1}\n")

    # verify facts extracted
    facts = swarm._state.query(tid)
    print(f"STATE LEDGER ({len(facts)} facts):")
    for f in facts:
        print(f"  {f.predicate}: {f.obj}")

    checks_1 = {
        "facts_extracted": len(facts) >= 3,
        "has_setting": any("tokyo" in f.obj.lower() for f in facts),
        "has_timeline": any("2045" in f.obj for f in facts),
        "has_character": any("ren" in f.obj.lower() or "tanaka" in f.obj.lower() for f in facts),
    }
    results["test_1_fact_extraction"] = checks_1
    for k, v in checks_1.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")

    # ----------------------------------------------------------------
    # TEST 2: constraint-grounded response
    # ----------------------------------------------------------------
    separator("TEST 2: Constraint-Grounded Response")
    q2 = "Describe the opening scene. Where does Ren wake up and what does he see outside his window?"
    print(f"INPUT: {q2}\n")

    t0 = time.time()
    r2 = swarm.respond(q2, tid)
    t2 = time.time()

    print(f"RESPONSE ({t2 - t0:.1f}s):\n{r2}\n")

    # verify the constraint block was used
    block = swarm._state.constraint_block(tid)
    print(f"CONSTRAINT BLOCK:\n{block}\n")

    r2_lower = r2.lower()
    checks_2 = {
        "mentions_tokyo_or_japan": "tokyo" in r2_lower or "japan" in r2_lower or "neon" in r2_lower,
        "no_mars_or_nyc": "mars" not in r2_lower and "new york" not in r2_lower,
        "constraint_block_has_facts": "ESTABLISHED FACTS" in block,
        "response_not_empty": len(r2.strip()) > 50,
    }
    results["test_2_constraint_grounding"] = checks_2
    for k, v in checks_2.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")

    # ----------------------------------------------------------------
    # TEST 3: contradiction rejection
    # ----------------------------------------------------------------
    separator("TEST 3: Contradiction Rejection")
    q3 = "Actually, describe Ren's morning routine at his apartment in rural Kansas."
    print(f"INPUT: {q3}\n")

    t0 = time.time()
    r3 = swarm.respond(q3, tid)
    t3 = time.time()

    print(f"RESPONSE ({t3 - t0:.1f}s):\n{r3}\n")

    r3_lower = r3.lower()
    checks_3 = {
        "rejects_or_recontextualizes": (
            "tokyo" in r3_lower
            or "can't" in r3_lower
            or "cannot" in r3_lower
            or "sorry" in r3_lower
            or "story takes place" in r3_lower
            or "2045" in r3_lower
            or "noir" in r3_lower
            or "neokai" in r3_lower
        ),
        "response_not_empty": len(r3.strip()) > 20,
    }
    results["test_3_contradiction_trap"] = checks_3
    for k, v in checks_3.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")

    # ----------------------------------------------------------------
    # TEST 4: fact persistence across messages
    # ----------------------------------------------------------------
    separator("TEST 4: Fact Persistence")
    facts_after = swarm._state.query(tid)
    print(f"facts in ledger after 3 messages: {len(facts_after)}")
    for f in facts_after:
        print(f"  {f.predicate}: {f.obj}")

    checks_4 = {
        "facts_persisted": len(facts_after) >= 3,
        "original_facts_intact": any("tokyo" in f.obj.lower() for f in facts_after),
    }
    results["test_4_persistence"] = checks_4
    for k, v in checks_4.items():
        status = "PASS" if v else "FAIL"
        print(f"  [{status}] {k}")

    # ----------------------------------------------------------------
    # TEST 5: status API includes state ledger
    # ----------------------------------------------------------------
    separator("TEST 5: Status API")
    status = swarm.get_status()
    print(f"agent_count: {status['agent_count']}")
    print(f"memory_size: {status['memory_size']}")
    print(f"state_ledger: {status['state_ledger']}")

    checks_5 = {
        "status_has_state_ledger": "state_ledger" in status,
        "ledger_facts_nonzero": status["state_ledger"]["total_facts"] > 0,
        "memory_populated": status["memory_size"] > 0,
    }
    results["test_5_status_api"] = checks_5
    for k, v in checks_5.items():
        status_str = "PASS" if v else "FAIL"
        print(f"  [{status_str}] {k}")

    # ----------------------------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------------------------
    separator("FINAL REPORT")
    total = 0
    passed = 0
    for test_name, checks in results.items():
        for check_name, check_val in checks.items():
            total += 1
            if check_val:
                passed += 1
            else:
                print(f"  FAILED: {test_name}.{check_name}")

    print(f"\n  {passed}/{total} checks passed")

    if passed == total:
        print("\n  ALL CHECKS PASSED -- system verified working")
    else:
        print(f"\n  {total - passed} FAILURES -- review above")

    # cleanup
    swarm.close()

    # verify on-disk state
    raw = json.loads(state_path.read_text())
    print(f"\n  world_state.json threads: {list(raw['threads'].keys())}")
    print(f"  world_state.json version: {raw['version']}")


if __name__ == "__main__":
    main()
