# Omni-Vera v1 — Project Instructions

## What this project is

Omni-Vera (codename for ICP Agent v1) takes qualitative customer evidence — interview transcripts, Reddit threads, deep-research PDFs — and produces a structured, citation-bearing Ideal Customer Profile. The same ICP then powers a streaming chat assistant that answers in character as the synthesised persona.

This working directory (`vera_v1/`) is a fork of the shipped v1 codebase. The shipped v1 is treated as stable infrastructure. New work happens additively on top of it.

## Current focus

**Building an agentic orchestrator.** A single CLI entry point (`scripts/run_agent.py`) that wraps Layer 2 (Processing) and Layer 3 (Synthesis & Activation) as a Claude tool-use loop. The user runs one command, the agent inspects state, runs the index build, runs synthesis, and hands off to the chat web UI in the same process.

Out of scope right now: Layer 1 (Ingestion) automation, multi-user support, prompt versioning, evals harness, deployed hosting.

## Architecture — as built

The shipped v1 is a three-layer model. The boundary between layers is the file system. Each layer writes its outputs to disk before the next layer reads them.

1. **Ingestion** (Layer 1) — captures raw evidence (intake form, transcript uploads, Reddit CSVs, deep-research PDFs). I/O-only, no LLM calls.
   - Modules: `web.py`, `intake.py`, `scripts/ingest_reddit.py`, `scripts/ingest_deep_research.py`
   - Outputs: `data/inputs/{intake_id}/`, `data/inputs/registry.json`, `data/raw/`

2. **Processing** (Layer 2) — parses, chunks, embeds, and indexes evidence into a single trust-tagged ChromaDB collection.
   - Modules: `transcripts.py`, `rag.py`, `scripts/build_index.py`
   - Outputs: `data/inputs/{intake_id}/parsed_transcripts.json`, `data/processed/chroma/`

3. **Synthesis & Activation** (Layer 3) — plans retrieval queries, fetches trust-weighted evidence, synthesises an ICP document with citations, and serves a persona-aware streaming chat UI.
   - Modules: `pipeline.py`, `models.py`, `chat.py`, `chat_web.py`, `scripts/run_pipeline.py`, `scripts/run_chat_server.py`
   - Outputs: `data/icp/{intake_id}/icp_*.json`, `data/chat/{intake_id}/`, `data/logs/`

The orchestrator currently being built sits *on top* of these layers as `src/icp_agent/agent/` plus `scripts/run_agent.py`. It does not replace them.

## Frozen modules — do not modify

These modules ship as-is in v1. Treat them as read-only infrastructure. If you think one of them needs to change to complete a task, stop and ask first — the answer is usually "wrap it, don't edit it."

- `src/icp_agent/rag.py`
- `src/icp_agent/pipeline.py`
- `src/icp_agent/chat.py`
- `src/icp_agent/chat_web.py`
- `src/icp_agent/web.py`
- `src/icp_agent/intake.py`
- `src/icp_agent/transcripts.py`
- `src/icp_agent/models.py`
- `scripts/build_index.py`
- `scripts/run_pipeline.py`
- `scripts/run_chat_server.py`
- `scripts/run_web_intake.py`
- `scripts/ingest_reddit.py`
- `scripts/ingest_deep_research.py`

The one allowed exception: if a frozen module's `main()` does work that needs to be reachable from new code (e.g. `build_index.py`'s `main()` does `sys.exit` on error), you may extract that work into a new callable function inside the same module, and have `main()` call into it. That's a refactor, not a behaviour change. Always propose this kind of change before making it.

## Code style

- Python 3.11+. Type hints on all public functions. Docstrings explaining purpose, args, and returns.
- Small functions, one responsibility each.
- Prefer `pathlib` over `os.path`.
- Structured logging via the `logging` module, not `print()`. The exception is CLI entry points where user-facing output is intended — the existing scripts establish the pattern (banner prints, status lines).
- All config via environment variables loaded with `python-dotenv` from `.env` at the project root.
- New modules go under `src/icp_agent/`. Orchestrator-specific code lives in `src/icp_agent/agent/`.

## Secrets and safety

- Never read or print the contents of `.env`. Use `os.environ` or `python-dotenv` to load secrets. To verify a key is set, check presence only — never print the value.
- Never hardcode API keys, subreddit names, search queries, or file paths that should be config.
- Never commit files in `data/raw/`, `data/processed/`, `data/icp/`, `data/chat/`, or `data/logs/` — they contain user data and run artefacts.
- The `.env` is already in `.gitignore`. Keep it that way.

## Testing

- Every module in `src/icp_agent/` has a matching test file in `tests/`.
- Tests cover: happy path, at least one edge case, at least one failure case.
- External APIs (Anthropic, OpenAI, Reddit, ChromaDB network calls) must be mocked in tests — no real API calls in the test suite.
- Use `pytest` fixtures for shared test data and `tmp_path` for filesystem tests.
- Frozen v1 modules already have their own tests; do not rewrite them. Add new tests for new code only.

## Definition of done

A task is not complete until:

1. Code written and follows the style rules above.
2. Code executed at least once on real input with output shown to me.
3. Tests written and passing.
4. Summary of changes in 2–3 bullets.
5. Suggested git commit message.

If blocked, say so explicitly. Never silently skip a step.

## Working rhythm

- Always plan before coding for anything touching 2+ files. Show the plan, wait for approval.
- Run scripts after writing or modifying them. Show the output.
- Commit after each completed module.
- When I paste an error, read the full traceback before guessing — the error usually says what's wrong.
- When in doubt about whether a change touches a frozen module, stop and ask.

## Storage layout reference

```
data/
├── inputs/
│   ├── registry.json                    # IntakeRegistry, atomic-write
│   └── {intake_id}/
│       ├── intake.json
│       ├── transcripts/
│       └── parsed_transcripts.json
├── raw/                                 # User-managed Reddit CSVs, deep-research PDFs
├── processed/
│   ├── transcripts.json                 # Combined parsed transcripts
│   └── chroma/                          # ChromaDB persistent store
├── icp/{intake_id}/
│   ├── icp_partial.json                 # Synthesis checkpoint
│   └── icp_{icp_id}.json                # Final ICPDocument
├── chat/{intake_id}/
│   ├── session_{ts}.json                # Saved conversation history
│   └── export_{ts}.pdf                  # Exported PDF
└── logs/
    ├── pipeline_{ts}.log
    ├── chat_{ts}.log
    └── agent_{ts}.log                   # New, written by the orchestrator
```

## External services

- **Anthropic** (Claude Sonnet 4.5) — synthesis, chat, query decomposition. Auth via `ANTHROPIC_API_KEY`.
- **OpenAI** (`text-embedding-3-small`) — all embedding traffic. Auth via `OPENAI_API_KEY`.
- **ChromaDB** — local persistent store at `data/processed/chroma/`. No network.

Model names are constants in code, not env vars. Promoting them to config is a v2 task.
