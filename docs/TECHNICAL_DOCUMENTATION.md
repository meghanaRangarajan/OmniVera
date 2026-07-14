<div align="center">

# Omni-Vera

### Technical Documentation

*v1 MVP — Engineer Reference*

![Public Edition](https://img.shields.io/badge/Edition-Public-2547A0?style=flat-square)
![Version](https://img.shields.io/badge/Version-1.0-2547A0?style=flat-square)
![Status](https://img.shields.io/badge/Status-Released-2547A0?style=flat-square)

**Audience:** engineers picking up the codebase
**Scope:** every public function, signature, side effect, and on-disk artefact
**Last updated:** 14 July 2026 · **Related docs:** [PRD](./PRD.md) · [Architecture](./ARCHITECTURE.md)

</div>

---

## Contents

1. [System Overview](#1-system-overview)
2. [Tech Stack](#2-tech-stack)
3. [Repository Layout](#3-repository-layout)
4. [Configuration & Environment](#4-configuration--environment)
5. [Module: `models.py`](#5-module-srcicp_agentmodelspy)
6. [Module: `intake.py`](#6-module-srcicp_agentintakepy)
7. [Module: `transcripts.py`](#7-module-srcicp_agenttranscriptspy)
8. [Module: `rag.py`](#8-module-srcicp_agentragpy)
9. [Module: `pipeline.py`](#9-module-srcicp_agentpipelinepy)
10. [Module: `chat.py`](#10-module-srcicp_agentchatpy)
11. [Module: `web.py`](#11-module-srcicp_agentwebpy)
12. [Module: `chat_web.py`](#12-module-srcicp_agentchat_webpy)
13. [Scripts](#13-scripts)

---

## 1. System Overview

Omni-Vera is a Python 3.11+ application that takes an intake form (product description, ICP hypothesis, transcripts) and produces **(a)** a structured ICP document with citation-bearing claims, and **(b)** a streaming chat assistant that answers as the synthesised persona.

It runs locally, persists everything to disk, and depends on three external services:

- **Anthropic** (Claude Sonnet 4.5) — synthesis, chat, and vision
- **OpenAI** (`text-embedding-3-small`) — embeddings
- **ChromaDB** — on-disk vector store

There are **8 Python modules** under `src/icp_agent/`, **9 runnable scripts** under `scripts/`, **8 test modules** under `tests/`, and 2 HTML templates. The total Python surface is roughly **4,000 LOC**. The system is single-user and stateless across process restarts (chat sessions excepted — those live in process memory until explicitly saved). The `data/` directory it produces is self-contained and can be archived or moved verbatim.

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Modern type-hint syntax (`X \| Y`), match statements, performance |
| LLM — synthesis & chat | Claude Sonnet 4.5 (`anthropic` SDK) | Long context, strong at structured JSON output |
| LLM — vision | Claude Sonnet 4.5 (multimodal) | Single vendor, image-to-text in chat |
| Embeddings | OpenAI `text-embedding-3-small` | Cheapest large-scale embedding model with strong recall |
| Vector DB | ChromaDB (`PersistentClient`) | Local, zero infra, HNSW + cosine |
| Tokenizer | `tiktoken` (word-count fallback) | Token-accurate chunk sizing |
| Validation | Pydantic v2 | Round-trips to JSON, readable validation errors |
| Web framework | Flask | Minimal, single-process, SSE-friendly |
| Streaming | Server-Sent Events | One-way token stream, no client library needed |
| PDF export | ReportLab | Pixel-level control over the styled export |
| File parsing | `pdfplumber`, `python-docx`, native Python | Permissive — handles messy real-world customer files |
| Settings | `python-dotenv` | Single `.env` file, no settings framework |
| Logging | stdlib `logging` + custom file handler | DEBUG to file, INFO to stdout |
| Tests | `pytest`, `pytest-mock`, `tmp_path` fixtures | Mocked LLM and vector-store calls — no real network in CI |

**Declared dependencies** (`pyproject.toml`):

```
anthropic, praw, pydantic>=2.0, python-dotenv, chromadb,
trafilatura, pytest, pytest-mock, python-docx, pdfplumber,
reportlab, openai, tiktoken, flask, rich
```

> `praw` and `trafilatura` are declared but unused in v1 — placeholders for a future live-Reddit and web-scraping path.

## 3. Repository Layout

```
omni-vera/
├── src/icp_agent/       # 8 modules, ~4,066 LOC
│   ├── __init__.py
│   ├── models.py         # ICPDocument, EvidencedClaim, SubPersona
│   ├── intake.py         # Intake, Registry, transcript ingestion
│   ├── transcripts.py    # Multi-format parser, splitter, section tagger
│   ├── rag.py            # Chunk, build_index, search_with_trust_priority
│   ├── pipeline.py       # 5-step synthesis orchestrator
│   ├── chat.py           # Per-turn RAG, persona prompt, streaming
│   ├── web.py             # Flask intake form server (port 5000)
│   └── chat_web.py        # Flask SSE chat server (port 5001)
├── scripts/              # 9 runnable entry points
├── tests/                # 8 pytest modules
├── templates/             # intake.html, chat.html
├── data/                  # inputs/, processed/, icp/, chat/, logs/, raw/
├── pyproject.toml
├── .env                   # local secrets (git-ignored — see .env.example)
└── README.md
```

## 4. Configuration & Environment

### 4.1 Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Sonnet 4.5 — synthesis, chat, vision |
| `OPENAI_API_KEY` | `text-embedding-3-small` embedding calls |
| `REDDIT_CLIENT_ID` | Reserved for future PRAW use — unused in v1 |
| `REDDIT_CLIENT_SECRET` | Reserved for future PRAW use — unused in v1 |
| `REDDIT_USER_AGENT` | Reserved for future PRAW use — unused in v1 |

### 4.2 Code-level constants

```python
# pipeline.py
COLLECTION_NAME = "icp_evidence"
SYNTHESIS_MODEL = "claude-sonnet-4-5"
MAX_CHUNKS_PER_SECTION = 8
LOW_CONFIDENCE_THRESHOLD = 3
MAX_SUB_PERSONAS = 5

# chat.py
CHAT_MODEL = "claude-sonnet-4-5"
IMAGE_MODEL = "claude-sonnet-4-5"
_HISTORY_COMPRESSION_THRESHOLD = 14
_HISTORY_KEEP_TAIL = 6

# rag.py — defaults
target_tokens = 400
max_tokens = 600
embedding_model = "text-embedding-3-small"
```

## 5. Module: `src/icp_agent/models.py`

Pydantic v2 schemas for the synthesised ICP. Round-trips losslessly to JSON; `field_validator`s enforce non-empty critical fields.

### 5.1 `EvidencedClaim`

```python
class EvidencedClaim(BaseModel):
    claim: str
    chunk_ids: list[str]          # validator: must be non-empty
    source_types: list[str]
    confidence: Literal["high", "medium", "low"]
```

### 5.2 `SubPersona`

```python
class SubPersona(BaseModel):
    name: str
    description: str
    key_traits: list[str]         # validator: non-empty
    motivations: list[str]
    objections: list[str]
    evidence_chunk_ids: list[str] # validator: non-empty
```

### 5.3 `ICPDocument`

```python
class ICPDocument(BaseModel):
    intake_id: str
    product_name: str
    icp_id: str = Field(default_factory=_default_icp_id)   # timestamp-based
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    demographics: list[EvidencedClaim] = []
    jobs_to_be_done: list[EvidencedClaim] = []
    pains: list[EvidencedClaim] = []
    gains: list[EvidencedClaim] = []
    objections: list[EvidencedClaim] = []
    vocabulary: list[str] = []              # plain strings, no citations
    watering_holes: list[EvidencedClaim] = []
    sub_personas: list[SubPersona] = []
    manual_edits: dict[str, Any] = {}       # reserved, v1 never writes
    status: Literal["draft", "reviewed", "locked"] = "draft"
```

### 5.4 Module-level functions

| Function | Behaviour |
|---|---|
| `_default_icp_id() -> str` | Generates a timestamp-based ID, e.g. `icp_20260425T140312Z`. Module-private. |
| `save_icp(icp, base_dir=Path("data/icp")) -> Path` | Writes `data/icp/{intake_id}/icp_{icp_id}.json`, creating the per-intake folder. |
| `load_icp(icp_path) -> ICPDocument` | Reads JSON and validates via Pydantic; raises `ValidationError` on schema drift. |

## 6. Module: `src/icp_agent/intake.py`

Intake schemas, registry persistence, and the user-facing transcript ingestion step. Registry writes are atomic.

### 6.1 Enums

```python
class ResearchGoal(str, Enum):
    TEST_AD_CREATIVE | TEST_MESSAGING | TEST_PRODUCT_IDEAS
    | GENERAL_RESEARCH | BUILD_SALES_COLLATERAL

class IntakeStatus(str, Enum):
    INTAKE_COMPLETE | RESEARCH_PLAN_PENDING | RESEARCH_PLAN_READY
    | RESEARCH_EXECUTING | RESEARCH_COMPLETE | ICP_SYNTHESIZED | CHAT_READY
```
> `RESEARCH_PLAN_*` and `RESEARCH_EXECUTING` are defined but unused in v1 — `pipeline.py` jumps straight from `INTAKE_COMPLETE` to `ICP_SYNTHESIZED`.

### 6.2 `Intake` (Pydantic `BaseModel`)

```python
class Intake(BaseModel):
    product_name: str = Field(min_length=2, max_length=100)
    product_description: str = Field(min_length=20, max_length=1000)
    company_name: str | None = Field(default=None, max_length=100)
    icp_hypothesis: str = Field(min_length=15, max_length=500)
    competitors: list[str] = Field(min_length=2, max_length=5)
    research_goals: list[ResearchGoal] = Field(min_length=1)
    target_geography: str | None = Field(default=None, max_length=200)
    intake_id: str
    intake_dir: Path
    transcript_files: list[Path] = []
    parsed_transcripts_path: Path | None = None
    created_at: datetime
    updated_at: datetime
    version: int = 1
```
`competitors` are stripped, deduplicated case-insensitively, and capped at 5 by a field validator.

### 6.3 `IntakeRegistryEntry` & `IntakeRegistry`

```python
class IntakeRegistryEntry(BaseModel):
    intake_id: str
    product_name: str
    company_name: str | None
    created_at: datetime
    updated_at: datetime
    version: int
    intake_dir: Path
    transcript_count: int
    turn_count: int
    research_goals: list[str]
    status: IntakeStatus

class IntakeRegistry(BaseModel):
    intakes: list[IntakeRegistryEntry] = []
    latest_intake_id: str | None = None
```

### 6.4 Public functions

| Function | Behaviour |
|---|---|
| `ingest_transcripts(source_files, dest_dir) -> list[Path]` | Copies uploads into `dest_dir`, filtering by `{.txt, .md, .pdf, .docx, .vtt, .srt}`. Skips unsupported files quietly. |
| `load_registry(base_dir) -> IntakeRegistry` | Reads `registry.json`, or returns an empty registry. Tolerant of a missing directory. |
| `save_registry(registry, base_dir)` | **Atomic write**: serialises to `registry.json.tmp`, then `os.replace()` into `registry.json`. A crash mid-write leaves the previous registry intact. |
| `register_intake(intake, status, base_dir)` | Loads the registry, replaces any existing entry with the same ID, updates `latest_intake_id`, saves atomically. |
| `save_intake(intake, base_dir) -> Path` | Writes `data/inputs/{intake_id}/intake.json`; calls `register_intake()` with `INTAKE_COMPLETE`. |
| `load_intake(intake_dir) -> Intake` | Reads and validates `intake.json`. |
| `load_latest_intake(base_dir) -> Intake \| None` | Resolves `registry.latest_intake_id`; `None` if the registry is empty. |
| `load_intake_transcripts(intake_id, base_dir) -> list[Transcript]` | Loads `parsed_transcripts.json`; falls back to re-parsing the `transcripts/` folder if missing. |
| `load_latest_intake_transcripts(base_dir) -> list[Transcript]` | Convenience wrapper: latest intake, then its transcripts. |

## 7. Module: `src/icp_agent/transcripts.py`

Multi-format transcript parser with multi-interview detection, metadata extraction, speaker-turn parsing, and section tagging. The most format-fragile module in the codebase.

### 7.1 Constants & regexes

```python
HEADER_RE   = r"In-Depth Qualitative Interview\s*-\s*(.+)"
SECTION_RE  = r"^\d+\.\s+([A-Z][A-Z &\-/]+)$"
SPEAKER_RE  = r"^([A-Z][A-Za-z .'\-]+):\s+(.+)$"
METADATA_RE = r"^([A-Za-z ]+?):\s*(.+)$"
# VTT_TIMESTAMP_RE / SRT_TIMESTAMP_RE / VTT_VOICE_RE — for subtitle parsing

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".vtt", ".srt", ".docx", ".pdf"}
```

### 7.2 Dataclasses

```python
@dataclass
class Turn:
    speaker: str
    text: str
    timestamp_seconds: float | None = None
    section: str | None = None

@dataclass
class Transcript:
    id: str                       # sha256 prefix
    source_path: Path
    title: str
    interviewee_name: str | None
    interviewee_metadata: dict[str, str]
    speakers: list[str]
    turns: list[Turn]
    raw_text: str
    source_type: str = "transcript"
    trust_weight: float = 1.0
    loaded_at: datetime = datetime.utcnow()
```

### 7.3 Public functions

| Function | Behaviour |
|---|---|
| `load_transcripts(folder_path) -> list[Transcript]` | Walks the folder, extracts raw text per file, detects and splits multi-interview files, parses turns and tags sections. |
| `save_transcripts(transcripts, output_path)` | JSON-serialises with a custom default handler for `Path`, `datetime`, and dataclasses. |
| `load_transcripts_from_json(json_path) -> list[Transcript]` | Inverse of `save_transcripts()`. |

### 7.4 Internal helpers

```python
_extract_raw_text(path, ext) -> str | None
_extract_docx(path) -> str            # python-docx, paragraph join
_extract_pdf(path) -> str             # pdfplumber, page join
_maybe_split(raw, path) -> list[tuple[str | None, dict[str, str], str]]
_extract_metadata(lines) -> dict[str, str]
_parse_text_turns(text) -> list[Turn]
_tag_sections(turns, raw_text) -> list[Turn]
_normalize_section(raw) -> str        # "BRAND PERCEPTIONS" -> "BRAND_PERCEPTIONS"
_parse_vtt(path) -> list[Turn]        # WebVTT cue blocks
_parse_srt(path) -> list[Turn]        # SubRip cue blocks
_build_transcript(...) -> Transcript
_make_id(path, raw_text, interviewee_name) -> str   # sha256(...).hexdigest()[:16]
```

`_maybe_split()` detects 2+ interview headers and splits accordingly; otherwise the file is treated as a single transcript. `_parse_text_turns()` filters out lines that look like metadata before any section header appears, so a line like `Age: 33` never becomes a `Turn` with speaker `"Age"`.

## 8. Module: `src/icp_agent/rag.py`

Chunking, embedding, ChromaDB indexing, and trust-weighted retrieval — the single source of truth for vector storage in the system.

### 8.1 `Chunk` dataclass

```python
@dataclass
class Chunk:
    id: str                      # sha256 prefix
    text: str
    transcript_id: str
    interviewee_name: str
    turn_ids: list[int]
    metadata: dict[str, Any]     # age, generation, location, section, source_type, trust_weight
    char_count: int
    token_count: int
```

### 8.2 Public functions

```python
def chunk_transcripts(
    transcripts: list[Transcript],
    target_tokens: int = 400,
    max_tokens: int = 600,
) -> list[Chunk]:
    """Convert transcripts into retrieval-ready chunks."""
```
Soft target: stop adding turns once `token_count > target_tokens` **and** the next turn looks like an interviewer question. Hard ceiling: never exceed `max_tokens`. A single turn that alone exceeds `max_tokens` becomes its own (oversized) chunk. Chunks never cross a section boundary.

```python
def build_index(
    chunks: list[Chunk],
    persist_dir: Path,
    collection_name: str = "icp_evidence",
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Embed all chunks via OpenAI and store in a persistent ChromaDB collection."""
```
Initialises `chromadb.PersistentClient(path=persist_dir)`, `get_or_create_collection(metadata={"hnsw:space": "cosine"})`. Embeds in batches of 100. Idempotent — re-running on the same chunks is a no-op.

```python
def search(
    query: str,
    persist_dir: Path,
    collection_name: str = "icp_evidence",
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> list[tuple[Chunk, float]]:
    """Return top_k chunks most similar to query, with similarity scores.
    `filters` uses ChromaDB's `where` clause syntax, e.g. {"generation": "Gen Z"}."""
```

```python
def search_with_trust_priority(
    query: str,
    persist_dir: Path,
    collection_name: str = "icp_evidence",
    transcript_top_k: int = 5,
    deep_research_top_k: int = 3,
    reddit_top_k: int = 3,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, list[tuple[Chunk, float]]]:
    """Embed the query once, then run three filtered searches."""
```
Returns `{"primary": [...transcripts], "secondary": [...deep_research], "corroboration": [...reddit]}`. The single-embedding optimisation matters at scale — an 8-section synthesis run issuing 35 queries embeds 35 times, not 105.

### 8.3 Internal helpers

```python
_make_chunk(transcript, indexed) -> Chunk
_to_chroma_metadata(chunk) -> dict[str, Any]   # flattens lists for Chroma's scalar-only metadata
_from_chroma_result(doc, meta) -> Chunk        # reverse of the above
```

## 9. Module: `src/icp_agent/pipeline.py`

Five-step synthesis orchestrator: verify → plan → retrieve → synthesise sections → synthesise personas → assemble. One Anthropic client. Deterministic retries on invalid JSON. Citation guard. Deterministic confidence re-grading. Partial checkpointing.

| Function | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `plan_and_retrieve()` | Builds the query map and runs retrieval per section | Query map, `persist_dir` | `dict[section -> list[evidence]]` |
| `synthesize_section()` | Per-section claim generator — one Claude call per section. Validates every cited `chunk_id` exists in the retrieved evidence; **rejects hallucinated citations**. Retries once on invalid JSON. | Section name, evidence chunks, intake | `list[EvidencedClaim]` |
| `synthesize_sub_personas()` | Cross-section persona synthesiser — one Claude call across the union of all evidence and claims. Emits 3–5 archetypes; truncates by evidence coverage if over the limit. | All evidence + claims + intake | `list[SubPersona]` |
| `assemble_and_save_icp()` | Final assembler. Re-grades every claim's confidence with deterministic rules (2+ transcript chunks → high; 1 → medium; 2+ non-transcript → medium; else low). Writes the `ICPDocument`, clears the checkpoint, flips registry status to `ICP_SYNTHESIZED`. | Intake, completed sections, personas, evidence | `data/icp/{intake_id}/icp_{icp_id}.json` |

Checkpointing: a partial file is written after each section and deleted on successful completion, so a crash mid-run resumes rather than restarts.

## 10. Module: `src/icp_agent/chat.py`

Per-turn input processing, query decomposition, persona prompt assembly, streaming response, and history compression. Stateless with respect to process state — `chat_web.py` owns the session dictionaries.

```python
def process_input(
    text: str | None = None,
    image_path: Path | None = None,
    file_path: Path | None = None,
) -> str:
```
Exactly one of `{text, image_path, file_path}` must be provided. Images are base64-encoded and sent to Claude Vision. Documents are extracted via `_extract_pdf`, `_extract_docx`, or native read.

```python
def build_chat_system_prompt(icp_doc: ICPDocument, active_persona: SubPersona) -> str:
```
Embeds the persona's name, description, key traits, motivations, and objections, plus the top 3 pains, top 3 gains, and top 15 vocabulary terms. Hard-codes the response rules: cite inline as `[transcript: name]` or `[reddit: username]`, 120-word cap, no formatting, end with *"what would change your mind?"*

```python
def decompose_and_retrieve(
    user_text: str,
    active_persona: SubPersona,
    persist_dir: Path,
    transcript_top_k: int = 3,
    deep_research_top_k: int = 2,
    reddit_top_k: int = 2,
) -> str:
```
Breaks the question into 3–5 atomic queries, retrieves via `search_with_trust_priority` for each, deduplicates, ranks by matched-query count then score, and renders the top 10 chunks as an evidence block.

```python
def stream_response(
    user_message: str,
    evidence_block: str,
    history: list[dict[str, str]],
    system_prompt: str,
    model: str = CHAT_MODEL,
) -> str:
```
Compresses history once it reaches the 14-message threshold, then streams the Claude response token by token.

**Internal helpers:** `_process_image()`, `_process_file()`, `_extract_pdf()`, `_extract_docx()`, `_decompose_user_text()` (falls back to a single query if the model returns invalid JSON twice), `_summarize_history()` (summarises everything except the last 6 messages into a short placeholder once history hits 14 messages).

## 11. Module: `src/icp_agent/web.py`

Flask intake form server. Single purpose: serve `templates/intake.html`, accept multipart submissions, build an `Intake`, parse transcripts immediately, and shut down.

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "inputs"
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB
```

| Route | Handler | Behaviour |
|---|---|---|
| `GET /` | `index()` | Renders `templates/intake.html` |
| `POST /api/intake` | `create_intake()` | Validates, ingests transcripts, parses, saves, prints a success banner, then schedules a graceful shutdown |

The server intentionally shuts itself down ~2 seconds after a successful submission — the rest of the pipeline runs from the command line, so there's no reason to keep a background process alive. Error paths (validation or otherwise) leave the server running so the user can fix the issue and retry.

## 12. Module: `src/icp_agent/chat_web.py`

Flask SSE chat server. Owns per-browser session IDs, in-memory caches, attachment handling, persona switching, session save/load, and the ReportLab PDF export.

### 12.1 Module state

```python
ICP_DIR     = PROJECT_ROOT / "data" / "icp"
CHAT_DIR    = PROJECT_ROOT / "data" / "chat"
LOG_DIR     = PROJECT_ROOT / "data" / "logs"
PERSIST_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"

_ICP_CACHE          # dict[intake_id -> (ICPDocument, Path)]
_CHAT_HISTORIES     # dict[session_id -> list[{role, content}]]
_ACTIVE_ATTACHMENTS # dict[session_id -> {filename, extracted_text, type}]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_FILE_EXTS  = {".pdf", ".docx", ".txt", ".md"}
```

### 12.2 Routes

| Route | Handler |
|---|---|
| `GET /` | `index()` — serves `chat.html` |
| `GET /api/chat/init` | `init_session()` — personas + intake metadata |
| `POST /api/chat/message` | `post_message()` — SSE stream |
| `POST /api/chat/persona` | `switch_persona()` |
| `POST /api/chat/save` | `save_session()` |
| `DELETE /api/chat/attachment` | `clear_attachment()` |
| `POST /api/chat/export` | `export_pdf()` |
| `GET /api/chat/history` | `history_stats()` — token estimate |

### 12.3 SSE streaming

`post_message()` yields three event shapes: `{type: "token", text}`, `{type: "done", full_text}`, and `{type: "error", message}`. The frontend appends tokens until it sees `done`.

### 12.4 PDF export

A `_FooterCanvas` subclass draws "Page X of Y" footers — needed because ReportLab doesn't know the total page count until after layout. The cover page renders the Omni-Vera wordmark, persona/product metadata, and export date, followed by the message log with role-coloured headers.

## 13. Scripts

All scripts run as `python scripts/<name>.py`. None are wired up as `console_scripts` entry points yet — promoting them to a proper CLI is on the v2 list.

| Script | Args | Behaviour |
|---|---|---|
| `run_web_intake.py` | `--host --port --debug` | Starts the intake server on `127.0.0.1:5000` by default |
| `build_index.py` | *(none)* | Loads parsed transcripts, chunks, builds the index, runs 4 smoke-test queries |
| `ingest_reddit.py` | *(none)* | Reads a Reddit CSV export, chunks with `trust_weight=0.6`, deduplicates and upserts |
| `ingest_deep_research.py` | *(none)* | Extracts PDFs, chunks with `trust_weight=0.8`, deduplicates and upserts |
| `run_pipeline.py` | `--intake-id {id\|latest}` | End-to-end synthesis; resumes from checkpoint if present |
| `run_chat.py` | `--intake-id --persona-index` | CLI REPL chat loop |
| `run_chat_server.py` | `--host --port --debug` | Starts the chat server on `127.0.0.1:5001` by default |
| `smoke_test.py` | *(none)* | End-to-end validation: latest intake → retrieval → ICP path |
| `test_retrieval.py` | *(none)* | Ad-hoc retrieval testing against the live index |

---

<div align="center">

*Part of the Omni-Vera documentation set — see [PRD.md](./PRD.md) and [ARCHITECTURE.md](./ARCHITECTURE.md).*

</div>
