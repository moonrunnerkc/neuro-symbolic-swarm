<div align="center">

# Swarm Chatbot

**A multi-agent LLM system that runs offline on a single consumer GPU.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-256%20Passing-brightgreen?logo=pytest&logoColor=white)](#test-suite)
[![Agents](https://img.shields.io/badge/Agents-7%20Specialized-blueviolet)](#the-swarm)
[![Ollama](https://img.shields.io/badge/Backend-Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com)

**Author:** Bradley R. Kinnard

</div>

---

## What Is This

Swarm Chatbot is a research prototype built on top of a general-purpose multi-agent orchestration engine. The chatbot is one interface for that engine. The same underlying pipeline -- agent dispatch, scored voting, symbolic validation, FAISS memory -- could drive a coding assistant, a document analyzer, or anything else that benefits from multiple models debating an answer before the user sees it.

This repo exists for testing and research. It is not a product.

The engine works like this: seven specialized LLM agents each generate a candidate response to your input. Each candidate is scored against the original query using cosine similarity on sentence embeddings. Candidates that fall below a quality threshold are dropped. The survivors pass through a symbolic validation layer that checks for factual contradictions and setting violations against a persistent state ledger. A synthesizer agent then combines the validated candidates into one final answer.

All of this runs locally through [Ollama](https://ollama.com). No API keys, no cloud, no telemetry.

---

## How the Engine Works

```
User Query
    |
    +---> [Fact-Extractor]  --->  State Ledger (world_state.json)
    |         phi3:mini                    |
    v                                      v
[Parser]  [Retriever]  [Innovator]    [Reasoner]
dolphin-mistral:7b x3           dolphin-llama3:8b x1
    |         |            |              |
    +----+----+-----+------+----+---------+
         |          |           |
         v          v           v
     Scored Drafts (embed + cosine gate)
         |
         v
  [Symbolic Validation] <-- State Ledger
         |                   hard reject contradictions
         |                   enforce setting anchors
         v
     [Critic] <-- embedding similarity (secondary signal)
   dolphin-mistral:7b
         |
         v
  [Synthesizer] <-- Constraint Block from State Ledger
   dolphin-llama3:8b
         |
     Final Response
```

### Symbolic State Ledger

A dedicated `phi3:mini` agent watches every message and extracts concrete facts (names, dates, locations, relationships) into `data/world_state.json`. Certain predicates lock after first write. Once the ledger records `era=medieval`, no subsequent message can overwrite it. This prevents the drift that happens in long conversations where models gradually forget what was established earlier.

The protected predicates are defined in `src/state_manager.py` under `PROTECTED_PREDICATES`. As of this release, the locked set includes: `setting`, `genre`, `era`, `timeline`, `planet`, `project_name`, `programming_language`, and `database`.

### VRAM Management

Agents are grouped by model name. During each round, the engine runs one agent per model concurrently (different models can coexist in VRAM). It never sends two requests to the same model at the same time. Doing so would double the KV cache and risk OOM on a 12GB card. After the drafting phase, non-synthesizer models are unloaded via Ollama's `/api/generate` keepalive=0 endpoint to free VRAM for synthesis.

The relevant code is in `src/swarm.py`, methods `_run_agents()` and `_unload_model()`.

### Memory Model

| Layer | Type | Storage | Purpose |
|:---|:---|:---|:---|
| **Episodic** | Neural | `data/threads/*.json` | Raw chat history per thread |
| **Semantic** | Vector | `data/memory.faiss` + `.json` | FAISS cosine similarity retrieval (384-d, `IndexFlatIP`) |
| **Procedural** | Symbolic | `data/world_state.json` | Hard facts and constraints, write-locked predicates |

Embeddings use `all-MiniLM-L6-v2` (384 dimensions, cosine similarity). The model auto-downloads on first run and executes on CPU.

---

## Using Your Own Models

The system calls Ollama's HTTP API at `http://localhost:11434/api/generate`. Any model that Ollama can serve will work. You pick which model each agent uses by editing its YAML config file in the `agents/` directory.

### Step 1: Pull a model through Ollama

```bash
# examples -- pick whatever fits your GPU
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:8b
```

The full list of available models is at [ollama.com/library](https://ollama.com/library). Models under 8B parameters run well on 12GB cards. Larger models (13B+) need 16GB+ VRAM or quantized variants (Q4_K_M, Q5_K_M).

### Step 2: Edit the agent YAML

Each agent has a YAML file in `agents/`. Open the one you want to change. Here is `agents/reasoner.yaml` as an example:

```yaml
role: Reasoner
model: dolphin-llama3:8b    # <-- change this line
prompt: >
  You are a reasoning engine...
max_tokens: 2048
temperature: 0.2
score_threshold: 0.1
```

Change the `model:` field to whatever you pulled. That is the only required change.

```yaml
model: llama3.1:8b
```

### Step 3: Restart

Stop and re-run `python run.py`. The agent now uses your chosen model.

### Things to keep in mind

- **VRAM budget.** The engine runs one agent per unique model concurrently. If you assign four agents to four different 8B models, all four need to fit in VRAM at the same time. Assigning multiple agents to the same model avoids this, since Ollama shares the loaded weights.
- **The Fact-Extractor needs a fast, structured model.** It runs on every single message and outputs only JSON. Small models like `phi3:mini`, `qwen2.5:1.5b`, or `gemma2:2b` work best here. Large creative models tend to output prose instead of valid JSON.
- **Temperature matters.** Low values (0.1 to 0.3) produce more consistent extraction and verification. Higher values (0.4 to 0.7) give more varied creative output for agents like Innovator.
- **GGUF models work too.** Create an Ollama Modelfile (there is an example in the repo root), run `ollama create my-model -f Modelfile`, then reference `my-model` in the agent YAML.

### Which agent does what

| Agent | Default Model | Job | Good model traits |
|:---|:---|:---|:---|
| Parser | `dolphin-mistral:7b` | Direct answers, constraint checking | Instruction-following, concise |
| Retriever | `dolphin-mistral:7b` | Factual recall, memory search | Precision, low hallucination |
| Critic | `dolphin-mistral:7b` | Verification, contradiction detection | Analytical, skeptical |
| Innovator | `dolphin-llama3:8b` | Creative angles, lateral thinking | Creative, varied output |
| Reasoner | `dolphin-llama3:8b` | Step-by-step logic | Chain-of-thought, systematic |
| Synthesizer | `dolphin-llama3:8b` | Combines validated drafts | Coherent, good at summarizing |
| Fact-Extractor | `phi3:mini` | JSON fact extraction | Fast, reliable structured output |

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
ollama pull phi3:mini           # fact extraction (fast, low VRAM)
ollama pull dolphin-mistral:7b  # parser, retriever, critic
ollama pull dolphin-llama3:8b   # innovator, reasoner, synthesizer
```

Or substitute any models you want. See [Using Your Own Models](#using-your-own-models).

### 3. Run

```bash
python run.py
```

---

## Programmatic API

You can skip the UI and use the engine directly:

```python
from src.swarm import SwarmChatbot

swarm = SwarmChatbot()
thread_id = swarm.create_thread("my-thread")

response = swarm.respond("The year is 2099. We are on Mars.", thread_id)
print(response)

swarm.close()
```

| Method | Description |
|:---|:---|
| `create_thread(id)` | Start a new conversation thread |
| `respond(query, thread_id)` | Run the full pipeline and return a response |
| `add_agent(config)` | Add an agent at runtime |
| `remove_agent(role)` | Remove an agent by role name |
| `get_status()` | Agent states, memory size, ledger stats |
| `get_thread_history(id)` | Retrieve conversation history for a thread |
| `clear_memory()` | Wipe FAISS memory and the state ledger |
| `close()` | Shutdown with state persistence |

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

| Key | What it does |
|:---|:---|
| `score_threshold` | Quality floor (0.0 to 1.0). Drafts scoring below this are cut before synthesis |
| `max_tokens` | Token cap per agent response |
| `context_window` | Number of past messages included as context for each agent |
| `auto_save_interval` | Seconds between FAISS memory persistence to disk |
| `default_model` | Fallback model if an agent YAML does not specify one |

---

## Test Suite

256 tests across 11 files. All run offline -- no Ollama or GPU needed.

| Test File | Tests | Covers |
|:---|:---:|:---|
| `test_swarm.py` | 36 | Orchestrator, symbolic validation, fact extraction, constraint injection, rollback, anachronism gate |
| `test_persistent_memory.py` | 29 | Embedding round-trips, FAISS persistence, config compatibility |
| `test_state_manager.py` | 23 | Ledger CRUD, persistence, concurrency, pruning, protected predicates |
| `test_structure.py` | 19 | File manifest integrity, YAML schema validation |
| `test_config.py` | 15 | Pydantic config models, defaults, overrides |
| `test_agent.py` | 12 | Message serialization, agent lifecycle, status tracking |
| `test_memory.py` | 11 | FAISS upsert, search, delete, reload, max-entry pruning |
| `test_embedder.py` | 5 | Embedding dimensions, normalization, determinism |
| `test_constraints.py` | 52 | Era blocklists, genre inference, anachronism detection, fact contradictions, extraction validation |
| `test_web_search.py` | 19 | Query classification, grounding block formatting |
| `test_adversarial_logic.py` | 5 | Contextual anachronisms, protected predicate locks, drift, rollback integrity |

```bash
pytest tests/ -v --tb=short
```

---

## Requirements

| Requirement | Minimum |
|:---|:---|
| **OS** | Linux (tested on Ubuntu 22.04) |
| **GPU** | NVIDIA with 12GB+ VRAM |
| **Python** | 3.10+ |
| **Backend** | [Ollama](https://ollama.com) running locally |

The embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run and runs on CPU.

---

## Project Structure

```
agents/             # YAML configs: model, prompt, temperature per agent
src/
  agent.py          # Agent class, Ollama HTTP calls, cosine scoring
  swarm.py          # Orchestrator, voting, symbolic validation, synthesis
  config.py         # Pydantic config models, YAML loader
  embedder.py       # Sentence-transformer embeddings
  memory.py         # FAISS vector memory
  state_manager.py  # Symbolic state ledger, protected predicates
  constraints.py    # Blocklists, anachronism detection, contradiction logic
  ui/               # PyQt6 interface
data/
  config.json       # App settings
  threads/          # Per-thread chat history (gitignored)
  memory.faiss      # FAISS index (gitignored, rebuilt at runtime)
  world_state.json  # Fact ledger (gitignored, rebuilt at runtime)
tests/              # 256 tests, all offline
```

---

## Limitations

This is a research prototype. Known constraints:

- Response latency scales with model count and size. A full 7-agent round on an RTX 5070 takes 10 to 40 seconds depending on input complexity.
- The symbolic validation layer is tuned for narrative/creative and technical project contexts. Other domains may need custom blocklists and contradiction patterns in `src/constraints.py`.
- Thread history is stored as unencrypted JSON on disk. No multi-user support.
- Tested on Ubuntu 22.04 with NVIDIA GPUs. Other platforms may work through Ollama but are not verified.

---

## Roadmap

- [ ] RAG document uploads for the Retriever agent
- [ ] Dual-GPU support for parallel model hosting
- [ ] Live ledger view in the UI

---

<div align="center">

**MIT License**

</div>
