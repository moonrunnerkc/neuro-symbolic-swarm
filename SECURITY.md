# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you find a security issue, **do not open a public issue**. Instead, email the maintainer directly or use GitHub's private vulnerability reporting feature.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

You should get a response within 48 hours. We'll work with you to understand and address the issue before any public disclosure.

## Design Notes

This project is designed to run fully offline on a single machine. There is no network-facing server, no authentication layer, and no remote API surface. The primary attack surface is local: malicious YAML agent configs, crafted FAISS indexes, or tampered thread JSON files.

If you're running this on a shared machine, treat the `data/` directory as sensitive.
