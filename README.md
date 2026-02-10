# Swarm Chatbot

**Offline. Decentralized. Hallucination-Proof.**

A multi-agent LLM system designed to run entirely offline on a single consumer GPU (RTX 5070+ recommended). Unlike standard chatbots that just predict the next token, Swarm uses a Symbolic State Ledger to track facts, enforce world consistency, and prevent the "amnesia" common in long conversations.

It features a Grok-style dark UI, full FAISS vector memory, and a swarm of 7 specialized agents that debate and refine answers before you see them.

---

## Why this is different

Most local chatbots are just wrappers around one model. Swarm is a pipeline:

- **Neuro-Symbolic Architecture:** A dedicated agent (`phi3:mini`) watches your chat and extracts facts (dates, locations, names) into a protected JSON ledger.

- **Hard Gating:** If you try to drive a truck in a medieval setting, the system physically rejects the draft before it reaches the synthesis stage.

- **VRAM Efficient:** Agents are grouped by model type. The system automatically loads and unloads models from VRAM as needed, so you can run a massive multi-agent swarm without needing 4x A100s.

---

## The Architecture

The flow is designed to catch hallucinations before they happen.

1. **Fact Extraction:** `phi3` pulls facts -> updates `world_state.json`.
2. **Drafting:** Parser, Retriever, and Innovator agents generate options.
3. **Validation:** Drafts are checked against the State Ledger. Anachronisms or contradictions trigger a hard reject.
4. **Synthesis:** The Synthesizer compiles the surviving drafts into the final answer.

---

## Requirements

- **OS:** Ubuntu 22.04 / Linux
- **GPU:** NVIDIA RTX 3090 / 4090 / 5070 (12GB+ VRAM required)
- **Backend:** Ollama running locally

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/swarm-chatbot.git
cd swarm-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Models

We use specific models for specific roles to optimize speed and logic.

```bash
ollama pull phi3:mini           # for fact extraction (fast)
ollama pull dolphin-mistral:7b  # for creative agents
ollama pull dolphin-llama3:8b   # for reasoning and synthesis
```

### 3. Run

```bash
# start the UI
python run.py
```

---

## Developer Usage

You can bypass the UI and use the Swarm directly in your Python scripts:

```python
from src.swarm import SwarmChatbot

swarm = SwarmChatbot()
thread_id = swarm.create_thread("sci-fi-story")

# the swarm will handle the model swapping and fact-checking automatically
response = swarm.respond("The year is 2099. We are on Mars.", thread_id)
print(response)

swarm.close()
```

---

## Configuration

Check `data/config.json` to tweak the swarm's strictness:

- **`agent_count`** -- How many concurrent drafters to run.
- **`score_threshold`** -- Quality bar (0.0 - 1.0). If drafts score below this, they are discarded.
- **`auto_save_interval`** -- How often to persist the vector memory to disk.

---

## Roadmap

- [ ] RAG Uploads: Drag-and-drop PDFs for the Retriever agent.
- [ ] Dual-GPU Support: Split the swarm across two cards to reduce model swapping latency.
- [ ] Live Ledger View: See the `world_state.json` update in real-time in the UI.

---

## License

MIT. Hack away.
