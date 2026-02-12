<div align="center">

<img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License" />
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
<img src="https://img.shields.io/badge/Tests-256_Passing-brightgreen?logo=pytest&logoColor=white" alt="256 Tests Passing" />
<img src="https://img.shields.io/badge/Agents-7_Specialized-7B68EE" alt="7 Agents" />
<img src="https://img.shields.io/badge/Backend-Ollama-000000?logo=ollama&logoColor=white" alt="Ollama Backend" />
<img src="https://img.shields.io/badge/Inference-100%25_Offline-2E8B57" alt="100% Offline" />

# Swarm Chatbot

**Seven LLM agents. One GPU. Zero cloud.**

*Author: Bradley R. Kinnard*

</div>

<br/>

> **What you're looking at:** This is a research prototype, not a product. The chatbot is just one skin on top of a general-purpose multi-agent orchestration engine. That same engine -- the agent dispatch loop, the scored voting, the symbolic validation, the FAISS memory -- could be wired up to a code reviewer, a document QA system, or anything else where you want multiple models to argue about an answer before the user ever sees it. The chatbot is the test harness.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture Diagram](#architecture-diagram)
- [Using Your Own Models](#using-your-own-models)
- [Quick Start](#quick-start)
- [Programmatic API](#programmatic-api)
- [Configuration](#configuration)
- [Test Results](#test-results)
- [Project Layout](#project-layout)
- [Requirements](#requirements)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## How It Works

Seven specialized agents each take a crack at your input. Their drafts are scored against your original query using cosine similarity on 384-dimensional sentence embeddings (`all-MiniLM-L6-v2`, runs on CPU). Anything below the quality threshold gets cut. The survivors go through a symbolic validation layer that cross-checks against a persistent fact ledger. A synthesizer agent then folds the validated drafts into one coherent answer.

Three things make this different from a typical single-model chatbot wrapper:

### 1. The State Ledger

A small dedicated model (`phi3:mini`) watches every message and pulls out concrete facts (names, dates, locations, roles, relationships) as flat JSON. Those facts go into `data/world_state.json`. Eight predicates are write-locked after first contact:

```
setting  genre  era  timeline  planet  project_name  programming_language  database
```

Once the ledger records `era=medieval`, that value is sealed. No later message can overwrite it. This is what prevents the slow drift you get in long conversations where the model gradually forgets what it established three pages ago.

*Source: `PROTECTED_PREDICATES` in [src/state_manager.py, line 136](src/state_manager.py#L136)*

### 2. Hard Gating

Every draft passes through `_symbolic_validate()` before synthesis. Three layers of defense:

| Layer | What It Catches | Example |
|:------|:----------------|:--------|
| Anachronism scan | Terms that violate the era blocklist | "truck" in a medieval setting |
| Fact contradiction check | Claims that conflict with locked predicates | Saying the database is MySQL when the ledger says CockroachDB |
| Setting anchor enforcement | Drafts that ignore the established world | A long scene set in space when the ledger says `planet=Earth` |

Failing drafts are physically rejected and never reach the synthesizer. If every draft fails, the engine rolls back any facts the extractor pulled from that message, restoring the ledger to its pre-message state.

The medieval era blocklist alone has **85 terms** -- everything from "truck" and "computer" to "tesla", "drone", and "satellite".

*Source: `ERA_BLOCKLISTS["medieval"]` in [src/constraints.py, line 21](src/constraints.py#L21)*

### 3. VRAM-Safe Execution

Agents are grouped by model name. Each round runs one agent per model in parallel -- different models can sit in VRAM simultaneously. The engine never fires two requests at the same model at once, which would double the KV cache and risk OOM on a 12 GB card. After drafting finishes, all non-synthesizer models are unloaded via Ollama's `keep_alive: 0` flag before the synthesizer runs.

*Source: `_run_agents()` at [src/swarm.py, line 553](src/swarm.py#L553); `_unload_model()` at [src/swarm.py, line 637](src/swarm.py#L637)*

---

## Architecture Diagram

```
User Input
    |
    +-----> [Fact-Extractor]  ------>  State Ledger (world_state.json)
    |          phi3:mini                       |
    v                                          v
[Parser]  [Retriever]  [Innovator]        [Reasoner]
   dolphin-mistral:7b x3              dolphin-llama3:8b x1
    |         |            |               |
    +----+----+------+-----+------+--------+
         |           |            |
         v           v            v
     Scored Drafts (384-d cosine similarity gate)
         |
         v
  [Symbolic Validation] <--- State Ledger
         |                     hard reject anachronisms
         |                     enforce locked predicates
         v
     [Critic] <--- embedding similarity (secondary signal)
    dolphin-mistral:7b
         |
         v
  [Synthesizer] <--- Constraint Block from Ledger
    dolphin-llama3:8b
         |
     Final Answer
```

### Memory Architecture

The engine maintains three distinct memory layers:

| Layer | Kind | Backing Store | What It Holds |
|:------|:-----|:--------------|:--------------|
| Episodic | Thread-local | `data/threads/*.json` | Raw message history, per conversation |
| Semantic | Vector | `data/memory.faiss` + `memory.json` | 384-d embeddings via `IndexFlatIP`, cosine retrieval |
| Procedural | Symbolic | `data/world_state.json` | Extracted facts, locked predicates, world constraints |

The FAISS index stores up to 200 entries (`MAX_ENTRIES` in [src/memory.py, line 60](src/memory.py#L60)). The embedding model (`all-MiniLM-L6-v2`) auto-downloads on first launch and runs entirely on CPU.

---

## Using Your Own Models

The engine talks to Ollama over HTTP at `localhost:11434`. The call goes to `POST /api/generate` with a simple JSON payload: model name, prompt, system message, temperature, and token cap. Any model Ollama can serve will work. You control which model each agent uses by editing one line in a YAML file.

### Step 1 -- Pull a model

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:8b
```

Full catalog at [ollama.com/library](https://ollama.com/library). Under 8B fits comfortably on 12 GB cards. 13B+ needs 16 GB or a quantized build (Q4_K_M, Q5_K_M).

### Step 2 -- Edit the YAML

Every agent lives in its own file under `agents/`. Open it and change the `model:` line:

```yaml
# agents/reasoner.yaml
role: Reasoner
model: dolphin-llama3:8b    # change this to your model
prompt: >
  You are a reasoning engine...
max_tokens: 2048
temperature: 0.2
score_threshold: 0.1
```

That one field is the only thing you need to touch. Swap it to whatever you pulled:

```yaml
model: llama3.1:8b
```

### Step 3 -- Restart

Kill the app, run `python run.py` again. Done.

### The agents and their defaults

| Agent | Ships With | Temp | Tokens | Threshold | What It Does |
|:------|:-----------|:----:|:------:|:---------:|:-------------|
| Parser | `dolphin-mistral:7b` | 0.3 | 2048 | 0.15 | Direct answers, constraint checking |
| Retriever | `dolphin-mistral:7b` | 0.2 | 2048 | 0.15 | Factual recall, memory search |
| Critic | `dolphin-mistral:7b` | 0.15 | 2048 | 0.15 | Verification, contradiction detection |
| Innovator | `dolphin-llama3:8b` | 0.5 | 2048 | 0.1 | Lateral thinking, creative angles |
| Reasoner | `dolphin-llama3:8b` | 0.2 | 2048 | 0.1 | Step-by-step chain-of-thought |
| Synthesizer | `dolphin-llama3:8b` | 0.2 | 2048 | 0.1 | Combines validated drafts into final output |
| Fact-Extractor | `phi3:mini` | 0.1 | 512 | 0.0 | JSON fact extraction for the state ledger |

*Source: each `agents/*.yaml` file. Values verified 2026-02-11.*

### Practical notes

- **VRAM math.** The engine runs one agent per unique model at the same time. Three agents sharing `dolphin-mistral:7b` only load it once. But four agents on four different 8B models means all four are in VRAM concurrently. Plan accordingly.
- **Fact-Extractor needs a small, fast model.** It fires on every single message and only outputs JSON. Big creative models tend to ramble instead of producing valid JSON. `phi3:mini`, `qwen2.5:1.5b`, or `gemma2:2b` are good picks.
- **Temperature controls personality.** 0.1 to 0.3 for precision work (extraction, verification). 0.4 to 0.7 for creative agents. The Innovator ships at 0.5 for a reason.
- **GGUF files work.** Write an Ollama Modelfile (there is a working example in the repo root that builds from `dolphin-llama3:8b` with an 8192-token context window), run `ollama create my-model -f Modelfile`, reference `my-model` in the YAML.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/moonrunnerkc/swarm-chatbot.git
cd swarm-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull models

```bash
ollama pull phi3:mini           # fact extraction
ollama pull dolphin-mistral:7b  # parser, retriever, critic
ollama pull dolphin-llama3:8b   # innovator, reasoner, synthesizer
```

Or swap in any models you want. See [Using Your Own Models](#using-your-own-models).

### 3. Launch

```bash
python run.py
```

---

## Programmatic API

Skip the UI entirely:

```python
from src.swarm import SwarmChatbot

swarm = SwarmChatbot()
thread = swarm.create_thread("test-run")

reply = swarm.respond("The year is 2099. We are on Mars.", thread)
print(reply)

swarm.close()
```

### Public methods

| Method | What It Does |
|:-------|:-------------|
| `create_thread(id)` | Start a new conversation thread |
| `respond(query, thread_id)` | Run the full agent pipeline, return a single answer |
| `add_agent(config)` | Hot-add an agent at runtime |
| `remove_agent(role)` | Drop an agent by role name |
| `get_status()` | Agent states, memory stats, ledger summary |
| `get_thread_history(id)` | Pull the raw message log for a thread |
| `clear_memory()` | Wipe the FAISS index and the state ledger |
| `clear_all_threads()` | Delete every thread's history |
| `close()` | Graceful shutdown, persists all state to disk |

*Source: all methods confirmed in [src/swarm.py](src/swarm.py). `clear_all_threads()` at line 1159.*

---

## Configuration

All runtime settings live in `data/config.json`:

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
  "auto_save_interval": 60,
  "web_search_enabled": false
}
```

| Key | What It Controls |
|:----|:-----------------|
| `score_threshold` | Minimum cosine similarity (0.0 to 1.0) a draft needs to survive into synthesis. Ships at 0.21. |
| `max_tokens` | Per-agent response cap. Overridden per-agent in the YAML if set there. |
| `context_window` | How many past messages get injected as context for each agent call. |
| `default_model` | Fallback if an agent YAML omits its `model:` field. |
| `auto_save_interval` | Seconds between automatic FAISS persistence to disk. |
| `web_search_enabled` | Enables Tavily-powered web grounding. Requires a `TAVILY_API_KEY` in `.env`. Off by default. |

*Source: `AppConfig` in [src/config.py, line 42](src/config.py#L42). 12 fields total.*

---

## Test Results

256 tests across 11 files. Every test runs offline. No Ollama instance, no GPU, no network access needed.

```
$ pytest tests/ -v --tb=short

256 passed in 7.66s
```

### Breakdown by file

| File | Count | Coverage |
|:-----|:-----:|:---------|
| `test_constraints.py` | 52 | Era blocklists, genre inference, anachronism detection, fact contradiction patterns (age, role, relationship, timeline, identity), extraction validation |
| `test_structure.py` | 49 | File manifest integrity, YAML schema validation, module presence checks |
| `test_swarm.py` | 36 | Orchestrator lifecycle, symbolic validation, fact extraction, constraint injection, ledger rollback, user-input anachronism gate |
| `test_persistent_memory.py` | 29 | Embedding round-trips, FAISS persistence across restarts, config compatibility |
| `test_state_manager.py` | 23 | Ledger CRUD, disk persistence, thread-safe concurrency, pruning, protected predicate enforcement |
| `test_web_search.py` | 19 | Query classification, grounding block formatting, search query extraction |
| `test_config.py` | 15 | Pydantic model validation, defaults, override behavior |
| `test_agent.py` | 12 | Message serialization, agent lifecycle, status tracking |
| `test_memory.py` | 11 | FAISS upsert, search, delete, reload, max-entry pruning at 200 cap |
| `test_embedder.py` | 5 | 384-d output, L2 normalization, deterministic repeat calls |
| `test_adversarial_logic.py` | 5 | Compound-word anachronisms, protected predicate locks, semantic drift detection, rollback integrity, multi-vector attack blocking |

### Adversarial test details

These five tests simulate deliberate attempts to break the validation layer:

| Test | Attack | Result | Mechanism |
|:-----|:-------|:------:|:----------|
| Compound word bypass | "laser-precise hammer" in medieval context | Blocked | Hyphen splitting exposes "laser"; hits the 85-term blocklist |
| Predicate overwrite | Set `planet=Mars`, then try `planet=Earth` | Blocked | `PROTECTED_PREDICATES` silently drops the second write |
| Semantic drift | Medieval prose that slips in "engines roar on the highway" | Blocked | "engine" and "highway" both caught regardless of prose wrapper |
| Ledger poisoning | Inject bad facts, trigger rollback | Restored | `_rollback_facts()` strips post-snapshot predicates; baseline intact |
| Multi-vector scatter | "smartphone", "wifi", "helicopter" spread across a medieval draft | Blocked | All three hit the blocklist; every draft hard-rejected |

*Source: [tests/test_adversarial_logic.py](tests/test_adversarial_logic.py)*

---

## Project Layout

```
agents/
  parser.yaml             # model, prompt, temp, threshold per agent
  retriever.yaml
  critic.yaml
  innovator.yaml
  reasoner.yaml
  synthesizer.yaml
  fact_extractor.yaml

src/
  agent.py                # Agent class, Ollama HTTP calls, cosine scoring
  swarm.py                # Orchestrator: dispatch, voting, validation, synthesis
  config.py               # Pydantic models, YAML loader, AppConfig
  embedder.py             # all-MiniLM-L6-v2, 384-d, CPU-only
  memory.py               # FAISS IndexFlatIP, 200-entry cap, disk persistence
  state_manager.py        # Symbolic ledger, protected predicates, thread scoping
  constraints.py          # Blocklists (85 medieval terms), contradiction patterns
  web_search.py           # Optional Tavily grounding, disabled by default
  ui/                     # PyQt6 dark theme interface
    controller.py
    theme.py
    widgets/

data/
  config.json             # Runtime settings (checked in)
  threads/                # Per-thread history (gitignored)
  memory.faiss            # FAISS index (gitignored, rebuilt at runtime)
  memory.json             # Embedding metadata (gitignored)
  world_state.json        # Fact ledger (gitignored, rebuilt at runtime)

tests/                    # 256 tests, all offline, ~8 seconds
```

---

## Requirements

| What | Minimum |
|:-----|:--------|
| OS | Linux (tested on Ubuntu 22.04) |
| GPU | NVIDIA, 12 GB+ VRAM |
| Python | 3.10+ |
| Inference | [Ollama](https://ollama.com) running locally |

The embedding model downloads automatically on first run (~90 MB) and runs on CPU. No GPU needed for embeddings.

### Python dependencies

```
sentence-transformers>=2.2.0    faiss-cpu>=1.7.4       numpy>=1.24.0
pydantic>=2.0.0                 pydantic-settings>=2.0.0   PyYAML>=6.0
PyQt6==6.7.0                    requests>=2.31.0
tavily-python>=0.5.0            python-dotenv>=1.0.0
pytest>=7.4.0                   pytest-cov>=4.1.0      pytest-qt>=4.2.0
```

---

## Known Limitations

This is a research prototype. It is not hardened for production use.

- **Latency.** A full 7-agent round takes 10 to 40 seconds on an RTX 5070 depending on input length and model sizes. Every agent generates a complete draft before scoring begins.
- **Domain tuning.** The symbolic validation layer ships pre-tuned for narrative/creative and technical project conversations. Other domains will need custom blocklists and contradiction patterns in `src/constraints.py`.
- **Storage.** Thread history and the state ledger are unencrypted JSON files on disk. There is no authentication and no multi-user support.
- **Platform.** Tested on Ubuntu 22.04 with NVIDIA GPUs. Other Linux distros and macOS may work through Ollama but have not been verified. Windows is untested.

---

## Roadmap

- [ ] RAG document uploads for the Retriever agent
- [ ] Dual-GPU support to run model groups in parallel across cards
- [ ] Live ledger viewer in the UI (watch `world_state.json` update in real time)

---

<div align="center">

MIT License

</div>
