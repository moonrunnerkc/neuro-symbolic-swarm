# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you find a security issue, do not open a public issue. Use GitHub's private vulnerability reporting feature or contact the maintainer directly.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix if you have one

Expect a response within 48 hours. The issue will be addressed before any public disclosure.

## Design Notes

This project runs fully offline on a single machine. There is no network-facing server, no authentication layer, and no remote API surface. The only network activity is between the Python process and the local Ollama server at `localhost:11434`.

The primary attack surface is local:

- Malicious YAML agent configs in `agents/` (arbitrary prompt injection)
- Crafted FAISS indexes or tampered thread JSON files in `data/`
- Prompt injection through user input (mitigated by the symbolic validation layer, but not fully hardened)

If you run this on a shared machine, treat the `data/` directory as sensitive. Thread history and the state ledger contain full conversation text in plaintext JSON.
