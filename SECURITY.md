# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

**Email:** [alex@alexchernysh.com](mailto:alex@alexchernysh.com)

Please do **not** open a public issue for security vulnerabilities.

## What to Include

When reporting, please provide:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive an acknowledgment within 48 hours and a detailed response within 5 business days.

## Scope

This system processes legal documents via a RAG pipeline that runs locally or inside Docker. By design:

- The FastAPI server binds to `localhost:8000` (not exposed externally by default)
- Qdrant runs locally on `localhost:6333` with no authentication by default
- API keys for external services (OpenAI, Cohere) are loaded from `.env` / `.env.local`
- The eval harness and submission tooling connect to external APIs using `EVAL_API_KEY`

**Out of scope:**

- Vulnerabilities in upstream dependencies (report those to the respective maintainers)
- Issues requiring physical access to the host machine
- Social engineering attacks

## Sensitive Files

The following files contain secrets and must never be committed:

- `.env.local` (workstation-specific secrets)
- Any file containing `EVAL_API_KEY`, `OPENAI_API_KEY`, or `COHERE_API_KEY`

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
