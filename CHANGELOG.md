# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-02-11

### Added
- Multi-agent orchestration engine with 7 specialized agents
- Agents: Parser, Retriever, Critic, Innovator, Reasoner, Synthesizer, Fact-Extractor
- Symbolic State Ledger with write-locked predicates (`world_state.json`)
- Symbolic validation with hard rejection gating for anachronisms and fact contradictions
- Protected predicate set: setting, genre, era, timeline, planet, project_name, programming_language, database
- Fact rollback on invalid input to prevent state poisoning
- Identity contradiction detection for project-level facts
- Key normalization for extractor output (maps common aliases to canonical keys)
- Truncated JSON recovery in fact extraction pipeline
- FAISS vector memory with `all-MiniLM-L6-v2` embeddings (384-d, cosine similarity)
- Tri-partite memory model: episodic (threads), semantic (FAISS), procedural (state ledger)
- VRAM-safe execution with model grouping and automatic GPU memory management
- Constraint block injection into Synthesizer prompts
- Reasoning tag stripping and scoring artifact cleanup in synthesis output
- PyQt6 dark UI with thread management, agent status cards, and memory controls
- YAML-driven agent configuration with per-agent model, temperature, and threshold
- Any Ollama-compatible model can be assigned to any agent via YAML config
- Programmatic API via `SwarmChatbot` class
- Optional web search grounding via Tavily
- 256 tests across 11 test files, all running offline
- Offline-first design, no cloud dependencies
