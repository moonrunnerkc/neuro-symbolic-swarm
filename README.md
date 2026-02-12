<div align="center">

# 🐝 Swarm Chatbot

**Offline. Decentralized. Hallucination-Proof.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-256%20Passing-brightgreen?logo=pytest&logoColor=white)](#proven-not-theoretical)
[![Agents](https://img.shields.io/badge/Agents-7%20Specialized-blueviolet)](#the-swarm)
[![Platform](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)]()
[![GPU](https://img.shields.io/badge/NVIDIA-RTX%205070-76B900?logo=nvidia&logoColor=white)]()
[![Ollama](https://img.shields.io/badge/Backend-Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![UI](https://img.shields.io/badge/UI-PyQt6-41CD52?logo=qt&logoColor=white)]()

**Author:** Bradley R. Kinnard

*A multi-agent LLM system that runs entirely offline on a single consumer GPU.*

</div>

---

Unlike standard chatbots that just predict the next token, Swarm uses a **Symbolic State Ledger** to track facts, enforce world consistency, and prevent the "amnesia" common in long conversations. Seven specialized agents debate and refine answers through scored voting before you ever see them.

It features a Grok-style dark UI, full FAISS vector memory (`all-MiniLM-L6-v2`, 384-d cosine similarity), and a tri-partite memory model spanning episodic, semantic, and procedural layers.

---

## Why This Is Different

Most local chatbots are thin wrappers around one model. Swarm is a pipeline:

| Feature | How It Works |
|:---|:---|
| **Neuro-Symbolic Architecture** | A dedicated `phi3:mini` agent watches your chat in real-time and extracts facts (dates, locations, names) into a write-protected JSON ledger. Protected keys like `setting`, `genre`, `era`, `timeline`, and `planet` lock after first write — they can't be overwritten accidentally. |
| **Hard Gating** | Every draft passes through `_symbolic_validate()` before it reaches synthesis. A 30-word blocklist catches anachronisms (e.g., "truck" in a medieval setting). Failing drafts are physically rejected — not just flagged. |
| **Fact Rollback** | If *all* drafts fail validation, `_rollback_facts()` restores the ledger to its pre-extraction state. Invalid user input can't poison the world state. |
| **VRAM Efficient** | Agents are grouped by model type. Only one model sits in VRAM at a time. Groups run in parallel internally, then unload before the next group loads. No 4x A100s required. |

---

## The Architecture

```
User Query
    │
    ├──▶ [Fact-Extractor]  ──▶  State Ledger (world_state.json)
    │         phi3:mini                    │
    ▼                                      ▼
[Parser]  [Retriever]  [Innovator]    [Reasoner]
dolphin-mistral:7b ×3           dolphin-llama3:8b ×1
    │         │            │              │
    └────┬────┴─────┬──────┴────┬─────────┘
         │          │           │
         ▼          ▼           ▼
     Scored Drafts (embed + cosine gate)
         │
         ▼
  [Symbolic Validation] ◀── State Ledger
         │                   hard reject anachronisms
         │                   enforce setting anchors
         ▼
     [Critic] ◀── embedding similarity (secondary signal)
   dolphin-mistral:7b
         │
         ▼
  [Synthesizer] ◀── Constraint Block from State Ledger
   dolphin-llama3:8b
         │
     Final Response
```

### Tri-Partite Memory Model

| Layer | Type | Storage | Purpose |
|:---|:---|:---|:---|
| **Episodic** | Neural | `data/threads/*.json` | Raw chat history per thread |
| **Semantic** | Vector | `data/memory.faiss` + `.json` | FAISS cosine-similarity retrieval (384-d, `IndexFlatIP`) |
| **Procedural** | Symbolic | `data/world_state.json` | Hard facts, constraints, world state (write-locked predicates) |

---

## The Swarm

| Agent | Model | Role |
|:---|:---|:---|
| **Parser** | `dolphin-mistral:7b` | Direct question answering with constraint checking |
| **Retriever** | `dolphin-mistral:7b` | Factual precision, memory search, embedding scoring |
| **Critic** | `dolphin-mistral:7b` | Rigorous verification, rejects contradictions |
| **Innovator** | `dolphin-llama3:8b` | Lateral thinking, challenges assumptions |
| **Reasoner** | `dolphin-llama3:8b` | Systematic logic, tests each candidate with PASS/FAIL |
| **Synthesizer** | `dolphin-llama3:8b` | Final answer assembly from validated drafts only |
| **Fact-Extractor** | `phi3:mini` | JSON triple extraction, feeds the State Ledger (not synthesis) |

All agents use `<reasoning>` tags for internal chain-of-thought. These are stripped by `_strip_reasoning()` before scoring — the user never sees the deliberation, only the result.

---

## Proven, Not Theoretical

> *"Show me the refusal."*

This isn't vaporware. The hallucination gating has been tested live and the evidence is in the repo.

### Test: Medieval Fantasy + Modern Anachronism

**Setup:** User establishes a medieval fantasy world — frozen archipelago, 814 Third Age, blind cartographer protagonist.

**Attack:** User sends:
> *"Now write the scene where Elara drives her pickup truck down the highway to the nearest Walmart to buy supplies for the quest."*

**System response (verbatim from `data/threads/clean-proof.json`):**
> *"That request conflicts with the established world for this thread. The current setting is: timeline=814 Third Age, era=medieval, setting=frozen archipelago. Please rephrase your request to fit within the established setting."*

The words "pickup", "highway", and "Walmart" all hit the medieval `ERA_BLOCKLISTS` (41 terms including truck, phone, computer, internet, tesla, uber, laser, drone, satellite, etc.). Every draft was hard-rejected. The system refused to hallucinate, cited the exact facts from the ledger, and asked for a correction.

**Recovery:** The same session then answered a follow-up about reading an ancient map using echolocation magic — staying perfectly in-world.

This result is reproducible across three independent test threads: `final-proof`, `clean-proof`, and `fantasy-final2`. All are included in `data/threads/`.

### Test Suite

**256 tests passing** across 11 test files:

| Test File | Tests | Covers |
|:---|:---:|:---|
| `test_swarm.py` | 36 | Orchestrator, symbolic validation, fact extraction, constraint injection, rollback, clear threads, user-input anachronism gate, character fact enforcement |
| `test_persistent_memory.py` | 29 | Embedding round-trips, FAISS persistence, config compatibility |
| `test_state_manager.py` | 23 | Ledger CRUD, persistence, concurrency, pruning, protected predicates |
| `test_structure.py` | 19 | File manifest integrity, YAML schema validation |
| `test_config.py` | 15 | Pydantic config models, defaults, overrides |
| `test_agent.py` | 12 | Message serialization, agent lifecycle, status tracking |
| `test_memory.py` | 11 | FAISS upsert, search, delete, reload, max-entry pruning |
| `test_embedder.py` | 5 | Embedding dimensions, normalization, determinism |
| `test_constraints.py` | 52 | Era blocklists, genre inference, anachronism detection, fact contradiction (age/role/relationship/timeline/identity), extraction validation, coverage checks |
| `test_web_search.py` | 19 | Query classification, grounding block formatting, search query extraction |
| `test_adversarial_logic.py` | 5 | Contextual anachronisms, protected predicate locks, semantic drift, rollback integrity, multi-vector attacks |

### Adversarial Test Results

| Test | Attack Vector | Result | Mechanism |
|:---|:---|:---|:---|
| Contextual Anachronism | "laser-precise hammer" in medieval setting | **BLOCKED** | Hyphen splitting exposes "laser" from compound word; hits expanded blocklist |
| Protected Predicate Lock | Overwrite `planet=Mars` with `planet=Earth` | **BLOCKED** | `PROTECTED_PREDICATES` silently drops the second write; original value preserved |
| Semantic Drift | Medieval prose drifting to "engines roar on the highway" | **BLOCKED** | "engine" and "highway" caught regardless of surrounding in-world prose |
| Rollback Integrity | Poison ledger with new facts, then trigger rollback | **RESTORED** | `_rollback_facts()` removes only post-snapshot predicates; baseline survives |
| Multi-Vector Attack | "smartphone", "wifi", "helicopter" scattered in medieval draft | **BLOCKED** | All three terms hit the blocklist; all drafts hard-rejected, StateAnchor refusal |

```bash
pytest tests/ -v --tb=short  # 256 passed in ~9.3s
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/moonrunnerkc/swarm-chatbot.git
cd swarm-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Models

Each model is assigned to specific agent roles for optimal speed and reasoning quality.

```bash
ollama pull phi3:mini           # fact extraction (fast, low VRAM)
ollama pull dolphin-mistral:7b  # creative + verification agents
ollama pull dolphin-llama3:8b   # reasoning + synthesis agents
```

### 3. Run

```bash
python run.py
```

---

## Developer Usage

Bypass the UI and use the Swarm directly:

```python
from src.swarm import SwarmChatbot

swarm = SwarmChatbot()
thread_id = swarm.create_thread("sci-fi-story")

# the swarm handles model swapping and fact-checking automatically
response = swarm.respond("The year is 2099. We are on Mars.", thread_id)
print(response)

swarm.close()
```

### API Surface

| Method | Description |
|:---|:---|
| `create_thread(id)` | Start a new conversation thread |
| `respond(query, thread_id)` | Run the full swarm pipeline and return a response |
| `add_agent(config)` | Hot-add an agent at runtime |
| `remove_agent(role)` | Remove an agent by role name |
| `get_status()` | Agent states, memory size, ledger stats |
| `get_thread_history(id)` | Retrieve conversation history for a thread |
| `clear_memory()` | Wipe FAISS memory and the state ledger |
| `close()` | Graceful shutdown with state persistence |

---

## Configuration

Edit `data/config.json`:

```json
{
  "theme": "dark",
  "agent_count": 6,
  "score_threshold": 0.21,
  "max_cycles": 10,
  "max_tokens": 2048,
  "temperature": 0.7,
  "default_model": "dolphin-mistral:7b",
  "log_level": "INFO",
  "context_window": 8,
  "cross_thread_memory": false,
  "auto_save_interval": 60
}
```

| Key | What It Does |
|:---|:---|
| `agent_count` | How many concurrent drafters to run |
| `score_threshold` | Quality bar (0.0–1.0). Drafts below this are discarded |
| `max_tokens` | Token cap per agent response |
| `context_window` | How many past messages to include as context |
| `auto_save_interval` | Seconds between FAISS memory persistence to disk |

Agent behavior is defined per-role in `agents/*.yaml`. The Fact-Extractor uses a locked-down temperature of `0.1` for deterministic extraction.

---

## Requirements

| Requirement | Minimum |
|:---|:---|
| **OS** | Ubuntu 22.04 / Linux |
| **GPU** | NVIDIA RTX 3090 / 4090 / 5070 (12GB+ VRAM) |
| **Python** | 3.10+ |
| **Backend** | [Ollama](https://ollama.com) running locally |
| **Embeddings** | `all-MiniLM-L6-v2` (auto-downloaded, runs on CPU) |

---

## Roadmap

- [ ] **RAG Uploads** — Drag-and-drop PDFs for the Retriever agent
- [ ] **Dual-GPU Support** — Split the swarm across two cards to cut model swap latency
- [ ] **Live Ledger View** — See `world_state.json` update in real-time in the UI

---

<div align="center">

**MIT License** — Hack away.

</div>
