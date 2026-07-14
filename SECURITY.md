<div align="center">

# Omni-Vera

### Security Policy

![Public Edition](https://img.shields.io/badge/Edition-Public-2547A0?style=flat-square)
![Version](https://img.shields.io/badge/Version-1.0-2547A0?style=flat-square)

**Last updated:** 14 July 2026 · **Related docs:** [PRD](./PRD.md) · [Architecture](./ARCHITECTURE.md) · [Technical Documentation](./TECHNICAL_DOCUMENTATION.md)

</div>

---

## Data Handling & Privacy

Omni-Vera is a **local-first** tool. In v1:

- All evidence — uploaded transcripts, Reddit exports, deep-research PDFs — is parsed, chunked, embedded, and stored **on the user's own machine**. Nothing is uploaded to a Omni-Vera-operated server, because no such server exists in v1.
- The only outbound network traffic is to the **Anthropic API** (synthesis, chat, vision) and the **OpenAI API** (embeddings). Model calls send only the retrieved evidence chunks relevant to a given query — never entire raw transcripts.
- Synthesised ICP documents, chat history, and session exports are written to the local `data/` directory and stay there unless the user explicitly shares or exports a file.

### Known limitation

v1 has **no PII-specific handling** beyond local file storage — no redaction, no encryption at rest, no automatic retention limits. If your transcripts contain personally identifiable information about vulnerable populations, or you're working under a data-handling agreement that requires more than "stays on my laptop," v1 does not yet meet that bar. This is called out deliberately in the [PRD's anti-personas section](./PRD.md#43-anti-personas) rather than left implicit.

## Secrets & Credentials

- API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are loaded exclusively via a local `.env` file, never hardcoded.
- Secret values are never printed to logs, stdout, or written into any file under `data/`.
- `.gitignore` excludes `.env` from version control. If you fork or clone this repo, copy `.env.example` to `.env` and fill in your own keys — never commit the real file.
- If a `.env` file is ever accidentally committed or shared (e.g. in a screen share, a branch push, or a support request), **rotate both API keys immediately** at the Anthropic Console and OpenAI Platform, rather than assuming exposure was limited.

## Supported Versions

| Version | Supported |
|---|---|
| 1.0.x (current) | ✅ |
| Pre-release / internal builds | ❌ |

This is a single-maintainer portfolio project, not a maintained library with a long support tail — security attention is focused on the current released version.

## Reporting a Vulnerability

If you find a security issue — a way to exfiltrate another user's local data, a prompt-injection path that leaks credentials, a dependency with a known CVE, or anything else — please report it privately rather than opening a public issue.

**Contact:** `[email protected]` — replace with your preferred contact before publishing.

Please include:
- A description of the issue and its potential impact
- Steps to reproduce, if possible
- Any relevant logs (with secrets redacted)

This is a solo-maintained project, so response times are best-effort rather than a formal SLA — expect an initial reply within a few days. Please allow time to investigate and patch before any public disclosure.

## Scope

This policy covers the Omni-Vera codebase in this repository. It does not cover the security practices of Anthropic or OpenAI's APIs themselves — see their respective security and trust documentation for that.

---

<div align="center">

*Part of the Omni-Vera documentation set — see [PRD.md](./PRD.md), [ARCHITECTURE.md](./ARCHITECTURE.md), and [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md).*

</div>
