# Contributing to Neuro-Symbolic Swarm

Thanks for your interest. Here is how to contribute without breaking things.

## Ground Rules

- This project runs fully offline. Do not introduce dependencies that phone home.
- All code targets Python 3.10+ on Linux.
- GPU budget: 12GB VRAM. Do not add features that assume datacenter hardware.

## Setup

```bash
git clone git@github.com:moonrunnerkc/neuro-symbolic-swarm.git
cd neuro-symbolic-swarm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pull models if you plan to run integration tests:

```bash
ollama pull phi3:mini
ollama pull dolphin-mistral:7b
ollama pull dolphin-llama3:8b
```

## Code Style

- PEP 8 formatting. Use `ruff` or `black`.
- `snake_case` for functions and variables, `PascalCase` for classes, `ALL_CAPS` for constants.
- Type hints on all public functions.
- Docstrings: short, casual, explain why not what.
- Comments: lowercase, pragmatic. No filler.
- Imports: standard library first, third-party second, local third. No unused imports.

## Agent Configs

Agent behavior lives in `agents/*.yaml`. If you add or modify an agent:

- Keep the YAML structure consistent with existing configs.
- Set a reasonable `max_tokens` cap.
- Test that the agent loads and responds before opening a PR.

## Testing

Every behavior change needs a test.

```bash
# unit tests (no GPU or Ollama needed)
pytest tests/ -v --tb=short

# live integration (requires Ollama + models running)
python -m tests.integration_live
```

Do not submit PRs with failing tests.

## Pull Requests

1. Fork the repo and create a feature branch off `main`.
2. Keep commits focused. One logical change per commit.
3. Write a clear PR description: what changed, why, how to test it.
4. Make sure `pytest tests/ -v` passes locally before pushing.
5. No stubs, no `pass` placeholders, no `TODO` comments in submitted code.

## Good Contributions

Check the README roadmap. High-value work right now:

- RAG document uploads for the Retriever agent
- Dual-GPU support for parallel model hosting
- Live State Ledger view in the UI

## What We Do Not Want

- Cloud or API dependencies
- UI framework changes (committed to PyQt6)
- Complexity without measurable benefit

## Reporting Bugs

Open an issue with:

- What you did
- What you expected
- What actually happened
- Your GPU, OS, and Python version

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
