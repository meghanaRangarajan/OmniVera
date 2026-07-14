# Vera — System Architecture

*v1, three-layer model. As-built description.*

Author: Meghana Rangarajan

---

## 1. Executive summary

Vera takes qualitative evidence — customer interview transcripts, Reddit threads, deep-research PDFs — and turns it into a structured Ideal Customer Profile. The same ICP then powers a streaming chat assistant that answers in character, reacts to messaging and ad concepts, and lets the user export the conversation as a PDF.

v1 is split into three layers, each bounded by the file system. Every layer has one job, a fixed input and output shape, and writes its results to disk before handing off. That structure is why any layer can be re-run, debugged, or swapped without touching the others — and why a crash mid-pipeline doesn't destroy upstream work.

| Layer | Responsibility | Primary modules | On-disk output |
|---|---|---|---|
| **1. Ingestion** | Capture raw evidence from the user (intake form, transcript uploads) and from external sources (Reddit CSVs, deep-research PDFs). Validate formats, copy files, register the intake. | `web.py`, `intake.py`, `scripts/ingest_reddit.py`, `scripts/ingest_deep_research.py` | `data/inputs/{intake_id}/`, `data/inputs/registry.json`, `data/raw/` |
| **2. Processing** | Parse multi-format evidence into structured turns, chunk respecting semantic boundaries, embed via OpenAI, persist into a single trust-tagged ChromaDB collection. | `transcripts.py`, `rag.py`, `scripts/build_index.py` | `data/inputs/{intake_id}/parsed_transcripts.json`, `data/processed/chroma/` |
| **3. Synthesis & Activation** | Plan retrieval queries, fetch trust-weighted evidence, synthesize an ICP with citations, activate it as a persona-aware streaming chat assistant. | `pipeline.py`, `models.py`, `chat.py`, `chat_web.py` | `data/icp/{intake_id}/icp_*.json`, `data/chat/{intake_id}/`, `data/logs/` |

The user only ever touches two parts of the system: the intake form at the top of Layer 1, and the chat window at the bottom of Layer 3.

---

## 2. Layer 1 — Ingestion

**Purpose:** capture every piece of raw evidence, normalize file formats, and register the intake so downstream layers can find it. I/O-heavy, CPU-light. No embeddings, no LLM calls.

| Module | Responsibility | Outputs |
|---|---|---|
| `web.py` | Flask intake-form server. Renders `templates/intake.html`, validates `POST /api/intake`, builds an `Intake` object, kicks off transcript parsing, then auto-shuts down. | `intake.json`, registry update |
| `intake.py` | Pydantic models (`Intake`, `IntakeRegistry`, `IntakeStatus`, `ResearchGoal`). Registry load/save with atomic writes. Transcript ingestion with a format whitelist. | `intake.json`, `registry.json` (atomic tmp-then-replace) |
| `scripts/ingest_reddit.py` | Reads a Reddit CSV, turns each row into a `Chunk` with `source_type="reddit"`, `trust_weight=0.6`, deduplicates, upserts only new chunks. | Upsert into ChromaDB `icp_evidence` |
| `scripts/ingest_deep_research.py` | Reads PDFs, extracts text via `pdfplumber`, splits on `##` headers, chunks to ~400 tokens, tags `source_type="deep_research"`, `trust_weight=0.8`. | Upsert into ChromaDB `icp_evidence` |

**Why it's its own layer.** User input has different failure modes than indexing does. A malformed PDF, an oversized upload, or a non-whitelisted extension should fail at upload time — not three minutes later inside a synthesis run. The atomic registry write means a crash during intake can't corrupt the master registry.

---

## 3. Layer 2 — Processing

**Purpose:** turn raw evidence into a single, queryable, trust-tagged knowledge base. Once Layer 2 finishes, Layer 3 never reads from disk except through ChromaDB.

| Module | Responsibility |
|---|---|
| `transcripts.py` | Multi-format parser (`.txt` `.md` `.pdf` `.docx` `.vtt` `.srt`). Detects and splits multi-interview files. Extracts interviewee metadata (age, generation, location, occupation, device, family). Regex-parses speaker turns and tags each with its section heading. |
| `rag.chunk_transcripts()` | Section-aware chunker. Groups turns into chunks with a 400-token soft target and 600-token hard ceiling, breaking only at interviewer-question boundaries or section changes. Each chunk inherits its interviewee's metadata. |
| `rag.build_index()` | Embeds chunks via OpenAI `text-embedding-3-small` in batches of 100, upserts into ChromaDB (cosine distance, HNSW) under collection `icp_evidence`. |
| `scripts/build_index.py` | Driver: load parsed transcripts → chunk → build index → run a validation query panel. |

### The Chunk schema

Every piece of retrievable evidence — transcript turn group, Reddit thread, deep-research excerpt — lives in ChromaDB as one `Chunk`. That unification is what lets a query about "the customer's pains" return interview testimony, Reddit complaints, and analyst observations side by side, trust weights attached.

```
Chunk(
  id, text, transcript_id, interviewee_name, turn_ids,
  metadata={age, generation, location, occupation, device,
            section, source_type, trust_weight, ...},
  char_count, token_count
)
```

### Trust weights

Source type is per-chunk metadata, not a separate collection — so one embedding lookup returns chunks across all source types, and the synthesis layer re-ranks by weight.

| Source type | Weight | Rationale |
|---|---|---|
| `transcript` | **1.0** | Primary evidence: customers in their own words. |
| `deep_research` | **0.8** | Curated analyst output. High signal, but secondary to direct customer voice. |
| `reddit` | **0.6** | Public, unsolicited, sometimes anonymous. Useful for corroboration and vocabulary. |

**Why it's its own layer.** Processing is expensive (PDF parsing, tokenization, embedding calls) and produces a large on-disk artifact. Isolating it means a synthesis re-run never re-embeds, and an intake change never invalidates the index unless it adds new transcripts. The single ChromaDB collection *is* the contract: anything Layer 3 needs to know about Layer 1 must be expressible as chunk metadata.

---

## 4. Layer 3 — Synthesis & Activation

**Purpose:** turn indexed evidence into (a) a structured, citation-bearing ICP document and (b) a live persona the user can converse with. The only layer that calls Claude.

### 4.1 Synthesis sub-pipeline

| Function | Responsibility |
|---|---|
| `verify_pipeline_inputs()` | Pre-flight gate. Registry exists, `intake_id` resolves, `intake.json` on disk, ChromaDB has ≥1 transcript chunk (mandatory). Warns if Reddit or deep-research are missing. |
| `build_synthesis_queries()` | Query planner. Claude generates 3–5 product-specific semantic queries for each of the seven ICP sections. Falls back to a hardcoded query pool on invalid JSON. |
| `retrieve_evidence_for_synthesis()` | Runs each query through `search_with_trust_priority()`, merges the three trust lanes, ranks by `matched_query_count` then score. |
| `synthesize_section()` | One Claude call per section. **Validates that every cited `chunk_id` exists in the retrieved evidence** — hallucinated citations are rejected. Retries once on invalid JSON. |
| `synthesize_sub_personas()` | One Claude call across the union of all evidence and claims. Emits 3–5 archetypes; truncates to 5 by evidence coverage. |
| `assemble_and_save_icp()` | Re-grades every claim's confidence with deterministic rules (2+ transcript chunks → high; 1 transcript → medium; 2+ non-transcript → medium; else low). Writes `ICPDocument`, clears the checkpoint, flips registry status to `ICP_SYNTHESIZED`. |

### 4.2 Chat / activation sub-pipeline

| Function | Responsibility |
|---|---|
| `chat.process_input()` | Normalizes text, images (Claude Vision) and documents (extracted text) into one user message. |
| `chat.decompose_and_retrieve()` | Breaks the question into 3–5 atomic queries, retrieves per-lane, deduplicates, ranks, renders the top 10 chunks as an evidence block. |
| `chat.build_chat_system_prompt()` | Assembles the persona prompt: traits, motivations, objections, top pains and gains, top vocabulary terms — plus enforcement rules (cite inline, ≤120 words, no formatting, end with "what would change your mind?"). |
| `chat.stream_response()` | Token-by-token streaming. Compresses history at ≥14 messages by summarizing older turns. |
| `chat_web.py` | Flask SSE server. Session ids, ICP/persona caching, file-upload extraction, persona switching, session save/load, ReportLab PDF export. |

**Why it's its own layer.** Synthesis is the only layer needing LLM credentials, the only one producing an artifact a non-engineer can read, and the only one holding open user sessions. Putting it last means the cost-bearing operations only run after the cheap ones have already succeeded.

---

## 5. Cross-cutting concerns

**Storage layout**

```
data/
├── inputs/
│   ├── registry.json                 # IntakeRegistry, atomic-write
│   └── {intake_id}/
│       ├── intake.json
│       ├── transcripts/
│       └── parsed_transcripts.json
├── raw/                              # User-managed Reddit CSVs, deep-research PDFs
├── processed/
│   ├── transcripts.json
│   └── chroma/                       # ChromaDB persistent store
├── icp/{intake_id}/
│   ├── icp_partial.json              # Synthesis checkpoint
│   └── icp_{icp_id}.json             # Final ICPDocument
├── chat/{intake_id}/
│   ├── session_{ts}.json
│   └── export_{ts}.pdf
└── logs/
```

**Secrets and config.** All secrets load from `.env` via `python-dotenv`. Source code never reads or prints `.env` contents and never logs secret values. Model names and the collection name are constants in code; promoting them to config is a v2 task.

**Atomicity and resumability.** Two patterns guarantee a crash never leaves corrupt state. The registry is written tmp-then-replace, so partial writes are invisible to readers. The synthesis pipeline checkpoints to `icp_partial.json` after each section — re-running picks up where it left off.

**Trust-weighted retrieval.** `rag.search_with_trust_priority()` embeds the query once and reuses it for three filtered lookups (transcript / deep-research / Reddit). The lanes are returned *separately*, so callers can reweight them, render them with different visual cues, or reject low-trust evidence outright.

**Observability.** Long-running scripts log to stdout (INFO) and a timestamped file under `data/logs/` (DEBUG). Synthesis emits a per-section line with evidence-chunk count and rejected hallucinations. No metrics or traces in v1.

---

## 6. Layer interaction contract

The contract between layers is the file system, not function calls.

| From → To | Mechanism | Note |
|---|---|---|
| 1 → 2 | Read `parsed_transcripts.json` / `load_transcripts()` | Multiple intakes can be combined before indexing. |
| 2 → 3 (synthesis) | `search_with_trust_priority()` with metadata filters | Synthesis only queries ChromaDB. It never touches raw transcripts. |
| 2 → 3 (chat) | Same retrieval API, but per-turn | Chat reuses the index synthesis built; it never re-indexes. |
| 3 → user | Flask + SSE, ReportLab for export | The user never sees a layer boundary. |

---

## 7. Evolution

The three-layer model is built to be replaced layer by layer, not re-architected wholesale. Likely v2 changes preserve the boundaries and swap implementations:

- **Ingestion** — deployed web app + object storage for transcripts; a real PRAW scraper replacing the user-supplied CSV.
- **Processing** — managed vector store; multilingual embeddings; re-index triggers keyed on transcript hash.
- **Synthesis** — versioned prompt registry; an evals harness scoring claims against held-out transcript spans; a manual-edit UI for ICP review and freeze.
- **Activation** — chat sessions out of in-process dicts and into Redis or Postgres; streaming attachments.

None of these require touching the contract. As long as `intake.json`, the `icp_evidence` collection, and `icp_*.json` keep their schemas, each layer evolves independently.

---

## Appendix — module-to-layer index

| Module | Layer | Role |
|---|---|---|
| `src/icp_agent/web.py` | 1 | Flask intake-form server |
| `src/icp_agent/intake.py` | 1 | Pydantic models, registry, transcript ingestion |
| `scripts/ingest_reddit.py` | 1 | Reddit CSV → ChromaDB |
| `scripts/ingest_deep_research.py` | 1 | Deep-research PDF → ChromaDB |
| `src/icp_agent/transcripts.py` | 2 | Multi-format parser + splitter |
| `src/icp_agent/rag.py` | 2 | Chunking, embedding, indexing, trust-weighted search |
| `scripts/build_index.py` | 2 | Index builder + smoke queries |
| `src/icp_agent/models.py` | 3 | `ICPDocument` / `EvidencedClaim` / `SubPersona` schemas |
| `src/icp_agent/pipeline.py` | 3 | Five-step synthesis pipeline |
| `src/icp_agent/chat.py` | 3 | Per-turn RAG, persona prompt, streaming |
| `src/icp_agent/chat_web.py` | 3 | Flask SSE server + PDF export |
| `src/icp_agent/agent/` | — | Orchestrator: tool-use loop over Layers 2–3 |
| `scripts/run_agent.py` | — | Single-command entry point |
