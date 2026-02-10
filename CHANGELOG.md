# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-02-10

### Added
- Initial release of the decentralized multi-agent swarm chatbot
- 7 specialized agents: Parser, Retriever, Critic, Innovator, Reasoner, Synthesizer, Fact-Extractor
- Neuro-Symbolic State Anchoring with persistent fact ledger (`world_state.json`)
- Symbolic validation with hard rejection gating for anachronisms and setting contradictions
- Fact rollback on invalid input to prevent world state poisoning
- Constraint block injection into Synthesizer prompts
- FAISS vector memory with sentence-transformers embeddings
- Tri-partite memory model (episodic, semantic, procedural)
- Grok-style dark UI built with PyQt6
- YAML-driven agent configuration
- VRAM-safe execution with model grouping and automatic GPU memory management
- Programmatic API via `SwarmChatbot` class
- Full test suite: unit tests, state manager tests, integration tests
- Offline-first design, no cloud dependencies
