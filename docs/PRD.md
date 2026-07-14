<div align="center">

# Omni-Vera

### Product Requirements Document

*v1 MVP — As-Built*

![Public Edition](https://img.shields.io/badge/Edition-Public-2547A0?style=flat-square)
![Version](https://img.shields.io/badge/Version-1.0-2547A0?style=flat-square)
![Status](https://img.shields.io/badge/Status-Released-2547A0?style=flat-square)

**Author:** Meghana Rangarajan
**Last updated:** 14 July 2026 · **Related docs:** [Architecture](./ARCHITECTURE.md) · [Technical Documentation](./TECHNICAL_DOCUMENTATION.md)

</div>

---

## Contents

1. [TL;DR](#1-tldr)
2. [Background & Problem Statement](#2-background--problem-statement)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Target Users & Personas](#4-target-users--personas)
5. [Use Cases & User Stories](#5-use-cases--user-stories)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Out of Scope (v1)](#8-out-of-scope-v1)
9. [Risks, Assumptions & Open Questions](#9-risks-assumptions--open-questions)
10. [Release Plan & Milestones](#10-release-plan--milestones)
11. [Appendix](#11-appendix)

---

## 1. TL;DR

**Omni-Vera** helps founders, marketers, and product researchers turn a pile of raw qualitative evidence — customer interview transcripts, Reddit threads, deep-research reports — into a structured, citation-bearing Ideal Customer Profile, then converse with that profile as a chat assistant.

The user submits an intake (product description, ICP hypothesis, competitors, transcripts), the system synthesises an ICP document with evidence-grounded claims, and a persona-aware chat UI lets the user pressure-test messaging, ad concepts, and product ideas as if they were talking to the synthesised customer.

> **Why this exists** — qualitative customer evidence is hard to convert into a usable ICP. Today it's a manual, days-long exercise: read transcripts, tag quotes in a spreadsheet, write a persona doc nobody updates. Omni-Vera compresses that work into a single intake, a reproducible synthesis run, and an interactive persona that grounds every answer in the underlying evidence.

## 2. Background & Problem Statement

### 2.1 What we observed

Every time a marketer or PM begins a positioning, ad-copy, or messaging exercise, they ask the same question: who is the customer, in their own words, and what do they actually care about? The artefacts that exist to answer that question — interview transcripts, Reddit threads, analyst reports — are too long to read end-to-end, scattered across formats and tools, and rarely cited when they show up in the marketing brief. The result is ICPs that drift into stereotype, messaging that doesn't reflect customer language, and ad creative that has to be re-tested from scratch.

### 2.2 Why now

Modern LLMs can read long-form transcripts and produce structured output. Vector databases make trust-weighted retrieval cheap. Streaming chat UIs make persona interaction feel natural. The gap isn't models — it's workflow: a tool that captures the right inputs, indexes them, synthesises an ICP that cites its sources, and lets the user converse with the result without leaving the artefact behind.

### 2.3 Why existing approaches fall short

- **Traditional user research firms** — expensive ($20k+), slow (4–8 weeks), produce static deliverables, no ongoing use.
- **Generic "customer persona" AI tools** — generate plausible-sounding personas from nothing: no real evidence, no citations, no grounding.
- **Raw LLM chats** — start from zero context every time; can't integrate real interviews and research; hallucinate customer opinions.
- **DIY (spreadsheets + manual coding)** — requires strategist-level skill, produces inconsistent quality, and is almost impossible to refresh.

## 3. Goals & Non-Goals

### 3.1 Goals (v1)

- Produce a structured, citable ICP document from 5+ customer interviews in under one hour of active user time.
- Support an interactive chat experience where users can probe the persona, test messaging, and get evidence-grounded responses.
- Every claim in the output must cite a specific source — interview quote, Reddit post, or web article.
- Treat customer interview transcripts as the highest-trust source and weight them accordingly in retrieval.
- Let the user review and edit the research plan before execution, and the ICP document before chatting.

### 3.2 Non-goals (v1)

- **Not a replacement for primary research** — Omni-Vera synthesises existing research; it doesn't run new interviews.
- **Not a persona hallucination engine** — if there's no evidence for a claim, none is invented.
- **Not multi-user collaboration** — a single-user tool in v1.
- **Not a CRM integration**, forecasting tool, or mass-consumer-scale ("upload 500 Amazon reviews") product.

### 3.3 Explicitly deferred to v2+

- Multi-user workspaces and sharing; scheduled ICP refresh as new data arrives
- Ad-creative generation (v1 tests creatives; it doesn't write them)
- OCR support for scanned PDFs; non-English language support
- A public, hosted web application (CLI/local web UI only in v1)

## 4. Target Users & Personas

v1 targets users who already have qualitative evidence and a working ICP hypothesis, and who need a faster way to formalise both. The system isn't designed for users doing primary research from scratch.

### 4.1 Primary user — the founding marketer

- Owns positioning, messaging, and ad copy for an early-stage product.
- Has access to 5–20 customer interview transcripts and informal Reddit research.
- Is comfortable in a web form but isn't a developer, and won't edit JSON by hand.
- Cares about being able to point to the evidence when defending a positioning choice.

### 4.2 Secondary user — the product researcher / PM

- Runs research sprints and needs an artefact that summarises a stack of interviews.
- Wants to interrogate the persona on edge cases — "what would the cautious buyer say to this objection?"

### 4.3 Anti-personas

- Enterprise marketing teams that need shared editing, governance, and SSO — v1 has none of these.
- Researchers studying populations where transcript metadata (age, location) is sensitive — v1 has no PII handling beyond local file storage.

## 5. Use Cases & User Stories

| ID | User story | Acceptance criterion |
|---|---|---|
| UC-1 | As a marketer, I can submit my product info and a folder of transcripts in one form, so I can kick off ICP synthesis without writing code. | Intake form accepts 1+ transcript and 1+ research goal; intake and parsed-transcript records appear under the intake's data folder. |
| UC-2 | As a marketer, I can run the synthesis pipeline against the latest intake and get a structured ICP document I can read. | The pipeline produces an ICP document with seven sections and 3–5 sub-personas; every claim has at least one citation. |
| UC-3 | As a marketer, I can chat with the synthesised persona and get evidence-grounded answers in real time. | Chat UI streams tokens; every answer cites the source type inline; responses are capped at 120 words and end with "what would change your mind?" |
| UC-4 | As a marketer, I can switch between the composite ICP and any sub-persona mid-conversation. | Persona switch happens without resetting the conversation history. |
| UC-5 | As a marketer, I can drop an ad image or one-pager PDF into the chat and ask the persona to react. | File upload is extracted via vision (image) or text extraction (document); extracted text is injected into the next turn. |
| UC-6 | As a marketer, I can export the conversation as a PDF for sharing. | Export returns a styled PDF with a cover page and "Page X of Y" footers. |
| UC-7 | As a marketer, I can supplement transcripts with Reddit threads and analyst reports. | Ingestion scripts add chunks into the same evidence index with lower trust weights. |
| UC-8 | As a marketer, when the synthesis pipeline crashes, I can resume without losing progress. | A checkpoint file allows the next run to skip already-synthesised sections. |

## 6. Functional Requirements

### 6.1 Intake

- The intake form accepts product name, product description, optional company name, ICP hypothesis, a deduplicated competitor list, research goals (multi-select), optional target geography, and one or more transcript files.
- Supported transcript formats: `.txt`, `.md`, `.pdf`, `.docx`, `.vtt`, `.srt`. Unsupported uploads are rejected at submit time with a clear error message.
- Maximum upload size: 200 MB per submission.
- Each submission produces a unique intake ID, a per-intake data folder, an intake record, and a registry update — written atomically.
- Submission triggers immediate transcript parsing; parsed output appears before the success confirmation.

### 6.2 Transcript parsing

- A single uploaded file may contain multiple interviews; the parser splits on recognised interview headers when two or more are detected.
- For each interview, the parser extracts age, generation, location, occupation, device, family, and interviewer metadata where present.
- Speaker turns are extracted via pattern matching; metadata lines are never misclassified as speaker turns.
- Section headings are normalised and used to tag each turn.

### 6.3 Indexing

- Chunking respects section boundaries; chunks target 400 tokens (soft) and never exceed 600 tokens (hard ceiling).
- Embeddings use OpenAI's `text-embedding-3-small` model, in batches of 100.
- All chunks across all source types live in one vector collection, using HNSW indexing with cosine distance.
- Each chunk carries source type and **trust-weight** metadata: transcripts 1.0, deep research 0.8, Reddit 0.6.
- Reddit and deep-research ingestion deduplicate against existing chunks before embedding.

### 6.4 Synthesis pipeline

- The pipeline verifies the registry, intake, and transcript chunks all exist before any model call.
- For each of the seven ICP sections (demographics, jobs-to-be-done, pains, gains, objections, vocabulary, watering holes), the model generates 3–5 product-specific semantic queries, with a hardcoded fallback pool if generation fails.
- Each query runs through trust-weighted retrieval; results across all lanes are merged and ranked.
- Per-section synthesis rejects any claim whose citations aren't present in the retrieved evidence — the **citation guard**.
- The pipeline produces 3–5 sub-personas; if more are generated, they're truncated by evidence coverage.
- Confidence is re-graded deterministically after generation (2+ transcripts → high; 1 transcript → medium; 2+ non-transcript sources → medium; otherwise low), regardless of what the model self-reported.
- A checkpoint is written after each section and deleted on successful completion, so a crash mid-run doesn't lose progress.

### 6.5 Chat assistant

- The chat server supports text, image, and document inputs in the same turn.
- Each turn decomposes the user's question into 3–5 atomic queries, retrieves evidence via the same trust-weighted search, and injects the top 10 chunks into context.
- The system prompt includes the active persona's traits, motivations, objections, top pains and gains, and top vocabulary terms, and enforces a 120-word response cap.
- Responses stream token by token.
- When conversation history reaches 14 messages, older messages are summarised into a short placeholder; the most recent six are kept verbatim.
- Users can switch active personas mid-conversation without resetting history, and can export a session as a styled PDF.

## 7. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | Synthesis completes in under 6 minutes for ~10 transcripts (~500 chunks); chat first-token latency under 5s (p50); embedding cost under $0.05 per intake. |
| Reliability | Atomic registry writes; idempotent re-ingestion; resumable synthesis via a checkpoint file. |
| Security | Secrets are loaded only via environment configuration; no secret values are printed or logged; uploaded transcripts are stored on local disk only; no outbound traffic beyond the Anthropic and OpenAI APIs. |
| Privacy | All evidence stays on the user's machine. Model calls send only retrieved chunks, never whole transcripts. |
| Observability | A detailed pipeline log per run, and a separate error-level chat log, for debugging without exposing evidence in the console. |
| Compatibility | Python 3.11+; macOS/Linux first; a single configuration file; any modern browser for the chat UI. |
| Cost ceiling | An entire intake → ICP → 30-turn chat conversation costs under $1 in API spend. |

## 8. Out of Scope (v1)

- Multi-user, multi-tenant deployment with authentication.
- Built-in transcript collection integrations (Zoom, Otter, Fireflies).
- Live Reddit scraping — v1 takes a CSV export.
- A manual ICP-edit interface — the underlying data model supports edits, but no UI exposes it yet.
- An A/B testing harness or multivariate messaging tests.
- Image generation in chat replies — the persona describes images, but never produces them.
- Localisation — v1 ships English-only.

## 9. Risks, Assumptions & Open Questions

### 9.1 Risks

| Risk | Impact | Mitigation |
|---|---|---|
| A hallucinated citation slips past the citation guard | Synthesis credibility collapses | The citation guard rejects claims with unrecognised source IDs; rejected counts are logged; a QA spot-check happens before sharing an ICP. |
| Trust weighting biases against Reddit even when it holds the truest signal | The ICP misses informal, real-world vocabulary | Three-lane retrieval still surfaces Reddit chunks; the vocabulary section is deliberately encouraged to draw on them. |
| A single-process server limits concurrency | Multiple users can't use chat at once | v1 is explicitly single-user; a hosted, multi-session version is planned for v2. |
| Credential leakage via local configuration | API charges could be hijacked | Configuration files are excluded from version control; keys are rotated after any repository sharing. |
| The multi-interview splitter mis-segments unusual transcript layouts | Some interviews collapse into one record | Test fixtures cover known layouts; an unrecognised layout falls back to single-interview parsing. |

### 9.2 Assumptions

- Users have at least 5 interview transcripts — below that, synthesis confidence will mostly read as "low."
- Users provide an ICP hypothesis; the system refines rather than discovers from zero.
- The Anthropic and OpenAI APIs are reachable from the user's machine.
- Input is English-language; non-English transcripts may parse, but synthesis quality is untested.

### 9.3 Open questions

- Should chat be allowed to refuse a question if retrieved evidence is too thin, instead of answering at low confidence?
- Should manual edits to the ICP overwrite synthesised claims, or live alongside them with a "user-edited" tag?
- Should the underlying evidence text be exposed in the chat UI on hover, or only the citation tag?
- How should transcript versioning work when the same interview is re-uploaded with corrections?

## 10. Release Plan & Milestones

v1 has shipped as the as-built MVP described in this document. The roadmap below sketches the most likely next two iterations; nothing here is committed.

| Milestone | Status | Target | Scope |
|---|---|---|---|
| v1.0 MVP | Released | April 2026 | End-to-end intake → indexing → synthesis → chat → PDF export, single-user, local-only. |
| v1.1 Polish | Planned | Q3 2026 | Manual ICP edit UI; richer chat citation hover-cards; audit log for synthesis runs. |
| v2.0 Hosted | Planned | Q4 2026 | Hosted, multi-user web app; managed vector database; live Reddit integration; transcript-provider integrations. |

## 11. Appendix

### 11.1 Glossary

| Term | Definition |
|---|---|
| ICP | Ideal Customer Profile — the structured persona document the system synthesises. |
| Intake | A single submission of product info and transcripts, producing a unique intake ID. |
| Chunk | A retrievable slice of evidence — transcript turns, a Reddit thread, or a deep-research excerpt — stored with metadata. |
| Trust weight | A per-chunk metadata field (1.0 / 0.8 / 0.6) that downstream synthesis uses to grade confidence. |
| Evidenced claim | A synthesised statement with citations, source types, and a confidence rating. |
| Sub-persona | An archetype within the ICP, with distinct traits, motivations, and objections, grounded in evidence. |
| Confidence re-grading | The deterministic step at the end of synthesis that overrides the model's self-reported confidence, based on evidence-source counts. |

### 11.2 Ideas captured for later

- Auto-detection of saturation — when does adding more transcripts stop changing the synthesised ICP?
- Persona-versus-persona dialogues, for adversarial messaging testing.
- Multi-language support with translation at the chunk level.
- One-click export connectors for Slack and Notion.

---

<div align="center">

*Part of the Omni-Vera documentation set — see [ARCHITECTURE.md](./ARCHITECTURE.md) and [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md).*

</div>
