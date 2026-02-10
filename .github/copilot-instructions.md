# Offline Decentralized Multi-Agent Swarm Chatbot with Grok-Style UI

> **Author:** Bradley R. Kinnard
> **License:** MIT
> **Platform:** Ubuntu 22.04 · RTX 5070 (Blackwell) · Fully Offline

---

## Overview

A paradigm-shifting **Decentralized Multi-Agent Swarm Architecture** for an offline chatbot. Runs entirely local on an RTX 5070 with Ubuntu 22.04, leveraging multiple lightweight LLM agents that collaborate via message-passing queues to enable **emergent intelligence**. No cloud, internet, or external APIs—everything stays private and offline.

The chatbot distributes user queries across specialized agents (parsing, retrieval, synthesis, critique, innovation), enabling collective decision-making and responses that evolve without a central controller. The UI mimics a **Grok-style interface**: dark theme (deep blacks, oranges, grays), left sidebar for agent status/thread management, right sidebar for settings/tools, and a central chat area.

---

## Deterring LLM Weaknesses Through Architecture

The swarm's decentralized design inherently promotes reliability against common LLM pitfalls:

| Weakness | Mitigation |
|---|---|
| Verbosity | Concise prompts + max token caps (256) |
| Hallucinations | Cross-verification via Critic agent + FAISS context grounding |
| Cascading errors | Voting/scoring thresholds gate propagation |
| Repetitive phrasing | Multi-model diversity across agents |
| Over-engineering | Simplicity-biased prompts + collective critiquing |
| Prompt dependency | Decentralized debate produces emergent, robust outputs |

---

## Key Principles

- **Modularity** — Agents defined via YAML configs for easy swapping/expansion.
- **Offline-first** — All models, embeddings, and storage are local.
- **Emergence** — Arises from decentralized interactions; agents "debate" via scored messages.
- **Hardware Optimization** — GPU MIG/MPS for parallel agent execution, minimal latency.
- **Reliability** — Swarm mechanics enforce concise outputs, error self-detection, and simplified reasoning.

---

## Prerequisites

### Hardware

- **GPU:** RTX 5070 (12GB VRAM minimum; Blackwell tensor cores for parallel inference)
- **Disk:** ≥ 20GB free (models + indexes)

### Software

- Ubuntu 22.04, CUDA 12.5+, NVIDIA drivers, Ollama (pre-installed)
- Git, Python 3.10+ (with `venv`)



### Integration Testing

```bash
python run.py
```

- Input queries, verify emergence and reliability in logs.
- Stress-test with 8 agents to saturate GPU; monitor for deterred issues.

### Debugging

- Built-in structured logs → check `data/logs.json`.
- Monitor VRAM: `watch -n1 nvidia-smi`.

### Optimization Checklist

- [ ] Latency < 10s per response? If not → tune VRAM allocation.
- [ ] VRAM headroom? If tight → quantize models further.
- [ ] Output quality? Adjust score thresholds for stricter deterrence.
- [ ] Agent count vs. GPU capacity balanced?

---

## Upgrade Path: Neuro-Symbolic State Anchoring ("World Model")

This section is a **living implementation guide** for upgrading the existing swarm system. It describes the Neuro-Symbolic "State Anchoring" paradigm, which transforms the chatbot from a probabilistic text generator into a reasoning engine that maintains a consistent internal reality.

> This guide targets the system **as already built**. All changes are incremental, backward-compatible, and designed to slot into the existing YAML-driven agent config and `swarm.py` orchestration without a total rewrite.

### Rationale: Why State Anchoring

Current LLMs operate on probability. Tell a chatbot "The door is locked," and 50 messages later ask "Can I walk through the door?" -- the LLM may hallucinate "Yes" because that token sequence is statistically likely. A State Anchor forces the system to check a Symbolic Ledger first (`door_status: locked`) and overrides the neural guess with a symbolic fact.

This is the **System 1 vs. System 2** distinction (Bengio, NeurIPS):
- **System 1 (Neural):** Fast, intuitive, error-prone. What LLMs do natively.
- **System 2 (Symbolic):** Slow, logical, deliberate. What the State Anchor provides.

Hybrid Neuro-Symbolic systems require less data and smaller models to reach higher accuracy than pure neural approaches (MIT/IBM Watson AI Lab, ICLR; Marcus, "Rebooting AI").

### The Tri-Partite Memory Model

The upgrade extends memory from a single FAISS vector store to three distinct layers:

| Layer | Type | Purpose | Storage |
|---|---|---|---|
| **Episodic** | Neural | Raw chat history per thread | `data/threads/*.json` (existing) |
| **Semantic** | Vector | RAG-based context retrieval | `data/memory.faiss` (existing) |
| **Procedural/Logical** | Symbolic | Hard facts, variables, constraints | `data/world_state.json` (new) |

### A. The "Fact-Extractor" Agent

Add a specialized agent (using `phi3:mini` for speed) that runs **in parallel with the Parser**. Its sole job is to output Subject-Predicate-Object triples from user input.

**Example:**

- **Input:** `"I am planning a sci-fi story set on Mars in the year 2099."`
- **Output:** `{"setting": "Mars", "timeline": "2099", "genre": "Sci-Fi"}`

**Implementation:**

1. Create `agents/fact_extractor.yaml` with a prompt focused on **extraction, not conversation**.
2. Wire the agent into `swarm.py` so it runs alongside the Parser in the parallel dispatch group.
3. Its output feeds directly into the State Manager (below), not into the synthesis pipeline.

### B. The State Ledger (The Anchor)

Create `src/state_manager.py` that maintains a persistent JSON object at `data/world_state.json`.

**Key behaviors:**

- **Upsert facts** extracted by the Fact-Extractor agent (thread-scoped by default, optionally global).
- **Constraint Enforcement:** Before the Synthesizer generates a response, the State Manager injects a "Constraint Block" into the system prompt:

  ```
  CURRENT CONTEXT: Location=Mars, Year=2099. YOU MUST NOT contradict these facts.
  ```

- **Persistence:** Atomic JSON writes with thread-safety (same pattern as `SharedMemory`).
- **Pruning:** Stale or superseded facts get overwritten by newer extractions for the same key.

### C. Logic-Based Validation (Critic Upgrade)

The Critic agent evolves from embedding-similarity scoring to **Symbolic Validation**:

1. Parse the Synthesizer's proposed response for factual claims.
2. Compare extracted facts against the State Ledger.
3. If a claim contradicts a ledger entry (e.g., response says "trees on Earth" but ledger says `location: Mars`), the Critic issues a **Hard Rejection** -- not a low score, but a binary gate.
4. Embedding similarity remains as a secondary quality signal, but no longer the primary validation mechanism.

### Implementation Steps (Phased)

**Phase 1: Schema and State Manager**
- Define `data/world_state.json` schema.
- Implement `src/state_manager.py` with upsert, query, prune, and constraint-block generation.
- Tests for state round-trips, thread scoping, and concurrent access.

**Phase 2: Fact-Extractor Agent**
- Create `agents/fact_extractor.yaml` targeting `phi3:mini`.
- Add extraction prompt tuned for triple output.
- Wire into `swarm.py` parallel dispatch alongside Parser.
- Tests for extraction accuracy on known inputs.

**Phase 3: Constraint Injection**
- Modify `_synthesize` in `swarm.py` to prepend the State Ledger constraint block.
- Verify constraint adherence in integration tests.

**Phase 4: Critic Symbolic Validation**
- Upgrade Critic's validation logic from cosine-only to ledger-aware hard rejection.
- Embedding similarity demoted to secondary signal.
- Tests for contradiction detection and rejection gating.

> **Do not advance phases unless the current phase passes all tests and verification.** Each phase must end with green tests, working integration, and updated README.

### Academic Grounding

- **Bengio et al.** -- "From System 1 Deep Learning to System 2 Deep Learning" (NeurIPS). The case for symbolic reasoning layers atop neural networks.
- **Gary Marcus** -- "The Algebraic Mind" / "Rebooting AI". Evidence that LLMs cannot generalize rules without a symbolic backbone.
- **MIT/IBM Watson AI Lab** -- "The Neuro-Symbolic Concept Learner" (ICLR). Proof that hybrid systems reach higher accuracy with smaller models and less data.

---

## Coding and Content Production Rules

These rules are **binding** for all code generation, documentation, and content produced by Copilot in this project. No exceptions.

### A. Truth and Verification Gates

1. **No hallucinations.** If something is uncertain, flag it as uncertain. Never guess.
2. **Correct false assumptions first.** If context is missing or something is mislabeled, fix that before generating any solution.
3. **Evidence-backed claims.** Words like "verified," "proven," "counterproof," "accurate," or "exact" require support: tests, screenshots, references, or direct source evidence.
4. **No "trust me" descriptions.** Provide explicit validation steps: commands to run, tests to pass, measurable outputs.
5. **State boundaries clearly.** If a system is deterministic within the software boundary but not bit-perfect across CPUs, say so plainly.

### B. Implementation Quality Standards

1. **No placeholders, no pseudocode, no mock/demo crap.** Code must be real, runnable, and directly applicable.
2. **Production-grade by default.** Concrete code, real file paths, real commands, real config structure.
3. **100% working after each phase.** Every phase ends with verification: tests, lint, type checks, or reproducible run output.
4. **Compatibility matters.** Changes must remain compatible with the existing system and constraints.
5. **DRY and SOLID.** Avoid duplication, keep responsibilities tight, don't grow a hairball.
6. **Determinism controls.** Seed RNGs, record run IDs/state digests, keep execution paths reproducible where intended.
7. **Offline-first assumptions.** Avoid "phone home" dependencies. Prefer designs that work air-gapped.
8. **Security and auditability are first-class.** Logs, digests, verification artifacts, and clear trust boundaries.

### C. Testing, Coverage, and Proof

1. **Tests are not optional.** Every meaningful behavior change requires tests.
2. **Coverage is a quality gate.** Push hard toward full coverage, not "some tests."
3. **Property tests when appropriate.** Especially for invariants and state transitions.
4. **Proof outputs must be inspectable.** Screenshots, test summaries, or verifiable command output are expected when claiming success.

### D. Build Guide Structure and Workflow

1. **Step-by-step, phase-based execution.** Phased plans with explicit deliverables and stop conditions.
2. **Do not advance phases unless asked.** Strict scope control across multi-day or multi-phase plans.
3. **Update docs as you go.** README and other docs updated alongside changes, not after the fact.
4. **README tracks upgrades.** When any upgrade (including Neuro-Symbolic State Anchoring phases) is successfully tested and verified working, the README **must** be updated to reflect the new capability, architecture change, and any new files or agents before the phase is considered complete.
5. **Single source of truth.** Ontology, API contracts, and build rules stay consistent and drive implementation.
6. **Explicit configuration and interfaces.** Clear CLI flags, config files, and stable contracts.

### E. Repo and Contribution Conventions

1. **PEP 8 import ordering.**
2. **Lowercase comments.**
3. **Docstring conventions:** no argument descriptions.
4. **Strict consistency with `CONTRIBUTING.md` rules** (treated as binding).

### F. Tone and Wording Constraints

1. **Serious, direct, minimal fluff.**
2. **Literal language.** No "poetic" or interpretive wording unless explicitly requested.
3. **No em dashes.** Not in prose, not in docs.
4. **No "AI tells."** Avoid the polished, generic, templated feel.

### G. Accuracy and Integrity Constraints

1. **Do not invent claims. Ever.**
2. **Do not remove claims when rewriting** unless explicitly approved.
3. **Reordering is allowed** only if accuracy is preserved.
4. **Make it "human written" when requested:** natural contractions, slightly imperfect rhythm, not over-edited, not corporate-gloss.

### H. README and Documentation Expectations

1. **Modern layout is fine; content integrity is non-negotiable.**
2. **Use common, expected repo sections.** Roadmap is acceptable; weird invented sections are not.
3. **Professional viewer-facing voice.** Don't make the README personal except for author attribution.
4. **Badges should be relevant and real.** No fake flex.
5. **Evidence sections are expected.** Surface "verification evidence" prominently when the repo makes reliability claims.
6. **Upgrade changelog in README.** Every verified upgrade (new agents, architecture changes, memory model expansions) must be documented in the README with a brief description, date, and verification status. The README is the public-facing proof that the system works.

### I. Copilot Session Rules

1. **Follow this instructions file exactly** by its real filename, not a made-up alias.
2. **Prove understanding with evidence** before changing things.
3. **Run to completion across phases** while staying inside these rules, updating docs, and verifying after each phase.

---

## Quick Reference

```bash
# activate environment
source ~/swarm-chatbot/venv/bin/activate

# start ollama backend
ollama serve

# launch the chatbot
python run.py
```
