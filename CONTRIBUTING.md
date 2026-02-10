# Contributing to Swarm Chatbot

Thanks for wanting to contribute. Here's how to do it without breaking things.

## Ground Rules

- This project runs **fully offline**. Don't introduce dependencies that phone home.
- All code targets **Python 3.10+** on Ubuntu 22.04.
- GPU assumptions: RTX 5070 (12GB VRAM). Don't add features that assume datacenter hardware.

## Getting Started

```bash
git clone git@github.com:moonrunnerkc/swarm-chatbot.git
cd swarm-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pull the models if you plan to run integration tests:

```bash
ollama pull phi3:mini
ollama pull dolphin-mistral:7b
ollama pull dolphin-llama3:8b
```

## Code Style

- **PEP 8** with `ruff` or `black` formatting.
- **Snake_case** for functions and variables, **PascalCase** for classes.
- **Type hints** on all public functions.
- **Docstrings** should be short and casual. Explain *why*, not *what*.
- Comments: lowercase, pragmatic. No filler.
- Imports: standard library first, third-party second, local third. No unused imports.
- Constants: `ALL_CAPS`.

## Agent Configs

Agent behavior lives in `agents/*.yaml`. If you're adding or modifying an agent:

- Keep the YAML structure consistent with existing configs.
- Set a reasonable `max_tokens` cap (256 default for most agents).
- Test that the agent loads and responds before opening a PR.

## Testing

Every behavior change needs a test. No exceptions.

```bash
# unit tests
pytest tests/ -v --tb=short

# live integration (requires ollama + models)
python -m tests.integration_live
```

Don't submit PRs with failing tests.

## Pull Requests

1. Fork the repo and create a feature branch off `main`.
2. Keep commits focused. One logical change per commit.
3. Write a clear PR description: what changed, why, how to test it.
4. Make sure `pytest tests/ -v` passes locally before pushing.
5. No stubs, no `pass` placeholders, no `TODO` comments in submitted code.

## What We're Looking For

Check the README roadmap for ideas. High-value contributions right now:

- RAG document upload for the Retriever agent
- Dual-GPU support for parallel model hosting
- Live State Ledger view in the UI

## What We Don't Want

- Cloud/API dependencies
- UI framework swaps (we're committed to PyQt6)
- "Improvements" that add complexity without measurable benefit
- Generated boilerplate or AI-typical code patterns

## Reporting Bugs

Open an issue with:

- What you did
- What you expected
- What actually happened
- Your GPU, OS, and Python version
- Relevant logs from `data/logs.json` if applicable

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
