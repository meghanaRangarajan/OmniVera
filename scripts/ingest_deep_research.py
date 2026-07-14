"""Ingest a Deep Research PDF report into the existing ChromaDB icp_evidence collection.

Extracts text from PDF(s) in the configured directory, cleans it, splits on ##
section headers, chunks to ~400 tokens (never splitting tables), embeds via
OpenAI text-embedding-3-small, and inserts only new chunks — leaving all
existing transcript and reddit chunks untouched. Safe to run multiple times.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import openai
import pdfplumber
import tiktoken
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icp_agent.rag import Chunk, _to_chroma_metadata, search, search_with_trust_priority  # noqa: E402

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PDF_DIR = ROOT / "data" / "raw" / "test_deep_research_report"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma"
COLLECTION_NAME = "icp_evidence"
EMBEDDING_MODEL = "text-embedding-3-small"
SOURCE_TYPE = "deep_research"
TRUST_WEIGHT = 0.8
TARGET_TOKENS = 400
MAX_TOKENS = 600
BATCH_SIZE = 100

# Human-readable titles for known PDF stems; falls back to stem-derived title.
_TITLE_MAP: dict[str, str] = {
    "deep_research_report": "Gen Z Outdoor Smartwatch Market and Garmin Roam Launch Context",
}

_enc = tiktoken.get_encoding("cl100k_base")

VALIDATION_QUERIES: list[str] = [
    "Gen Z price sensitivity and value perception",
    "Garmin brand perception among young buyers",
    "fit comfort unmet needs small ears",
    "Gen Z discovery and purchase journey social media",
    "messaging authenticity what works with Gen Z",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _make_id(pdf_path: Path, section_chunk_key: str, report_title: str) -> str:
    key = f"{pdf_path.stem}:{section_chunk_key}:{report_title}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _report_title(pdf_path: Path) -> str:
    return _TITLE_MAP.get(pdf_path.stem, pdf_path.stem.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Step 1 — Text extraction and cleaning
# ---------------------------------------------------------------------------


def extract_and_clean_text(pdf_path: Path) -> str:
    """Extract full text from all PDF pages and apply cleaning passes.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cleaned concatenated text.
    """
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    raw_text = "\n".join(pages)

    # Capture a before/after sample if (cid: artifacts are present
    cid_match = re.search(r".{0,80}\(cid:\d+\).{0,80}", raw_text)
    raw_sample = cid_match.group(0) if cid_match else None

    text = raw_text

    # Pass 0a — remove (cid:N) tokens produced by pdfplumber for special chars
    text = re.sub(r"\(cid:\d+\)", "", text)

    # Pass 0b — strip ChatGPT citation markers (cid removal may have exposed these)
    text = re.sub(r"citeturn\d+\w*", "", text)

    # Pass 0c — clean up whitespace artifacts left by the removals
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" ([.,;:])", r"\1", text)

    # Pass 1 — normalize entity tags
    text = re.sub(r'entity\["[^"]+","([^"]+)","[^"]+"\]', r"\1", text)

    # Pass 2 — fix ligature corruption (pdfplumber renders fi/ff/fl etc. as _)
    ligature_fixes = {
        "di_icult": "difficult",
        "di_erent": "different",
        "o_icial": "official",
        "o_er": "offer",
        "o_ers": "offers",
        "o_ering": "offering",
        "a_ordability": "affordability",
        "a_ordable": "affordable",
        "su_icient": "sufficient",
        "e_ect": "effect",
        "e_ects": "effects",
        "pro_it": "profit",
        "ful_ill": "fulfill",
        "con_irm": "confirm",
        "speci_ic": "specific",
        "speci_ically": "specifically",
        "signi_icant": "significant",
        "bene_it": "benefit",
        "bene_its": "benefits",
    }
    for broken, fixed in ligature_fixes.items():
        text = text.replace(broken, fixed)

    if raw_sample is not None:
        # Find the corresponding cleaned region for a before/after print
        # Use a nearby unique word from the raw sample (stripped of cid tokens)
        cleaned_sample = re.sub(r"\(cid:\d+\)", "", raw_sample)
        cleaned_sample = re.sub(r"citeturn\d+\w*", "", cleaned_sample).strip()
        print("\n── CID Artifact Cleaning Sample ────────────────────────────")
        print(f"  BEFORE: {raw_sample!r}")
        print(f"  AFTER:  {cleaned_sample!r}")
        print("─────────────────────────────────────────────────────────────\n")
    else:
        print("  No (cid:N) artifacts found in raw PDF text — already clean.")

    return text


# ---------------------------------------------------------------------------
# Step 2 — Section parsing
# ---------------------------------------------------------------------------


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Split text on ## headers into (section_name, section_text) pairs.

    ### subsection headers collapse into their parent ## section — they are not
    treated as new sections.  Everything before the first ## becomes
    "Executive Summary".

    Args:
        text: Cleaned full document text.

    Returns:
        Ordered list of (section_name, section_text) tuples.
    """
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("Executive Summary", text.strip())]

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("Executive Summary", preamble))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append((section_name, section_text))

    return sections


# ---------------------------------------------------------------------------
# Step 3+4 — Chunking logic
# ---------------------------------------------------------------------------


def _is_table_block(block: str) -> bool:
    """Return True if more than half the non-empty lines start with '|'."""
    lines = [ln for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    return sum(1 for ln in lines if ln.lstrip().startswith("|")) > len(lines) / 2


def _split_into_blocks(section_text: str) -> list[str]:
    """Split section text into paragraph blocks.

    Priority:
    1. Double-newline split (standard markdown paragraphs).
    2. Group consecutive non-empty lines separated by blank lines.
    3. When no blank lines exist at all (pdfplumber with zero empty lines),
       treat every individual line as its own block so the downstream
       accumulator can group them up to the token target.
    """
    blocks = [b.strip() for b in section_text.split("\n\n") if b.strip()]
    if len(blocks) > 1:
        return blocks

    lines = section_text.split("\n")
    blocks = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                blocks.append(" ".join(current))
                current = []
    if current:
        blocks.append(" ".join(current))

    if len(blocks) > 1:
        return blocks

    # No blank lines at all — treat each non-empty line as its own block so the
    # token-aware accumulator in chunk_section can group them to ~target_tokens.
    line_blocks = [ln.strip() for ln in lines if ln.strip()]
    return line_blocks if line_blocks else [section_text.strip()]


def _reassemble_tables(blocks: list[str]) -> list[str]:
    """Merge table rows and their wrapped-cell continuations into single atomic blocks.

    pdfplumber wraps long table cells onto the next line without the leading '|',
    producing alternating table-row / continuation-text pairs.  A non-table line
    that sits between two table-starting lines is treated as a continuation of the
    current table rather than a standalone block.
    """
    reassembled: list[str] = []
    current_table: list[str] = []
    for i, block in enumerate(blocks):
        if block.strip().startswith("|"):
            current_table.append(block)
        else:
            next_is_table = i + 1 < len(blocks) and blocks[i + 1].strip().startswith("|")
            if current_table and next_is_table:
                current_table.append(block)
            else:
                if current_table:
                    reassembled.append("\n".join(current_table))
                    current_table = []
                reassembled.append(block)
    if current_table:
        reassembled.append("\n".join(current_table))
    return reassembled


def chunk_section(
    section_name: str,
    section_text: str,
    report_title: str,
    pdf_path: Path,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
) -> list[Chunk]:
    """Chunk one section into Chunk objects.

    Args:
        section_name: Parent ## section name.
        section_text: Full text of the section.
        report_title: Human-readable report title for metadata.
        pdf_path: Source PDF path (used for stable ID generation).
        target_tokens: Soft target per chunk.
        max_tokens: Hard ceiling; oversized single blocks become their own chunk.

    Returns:
        Ordered list of Chunk objects for this section.
    """
    if not section_text.strip():
        return []
    blocks = _reassemble_tables(_split_into_blocks(section_text))
    chunks: list[Chunk] = []
    current_blocks: list[str] = []
    current_tokens = 0
    chunk_index = 1

    def _flush() -> None:
        nonlocal current_blocks, current_tokens, chunk_index
        if not current_blocks:
            return
        text = "\n\n".join(current_blocks).strip()
        cid = _make_id(pdf_path, section_name + str(chunk_index), report_title)
        chunks.append(
            Chunk(
                id=cid,
                text=text,
                transcript_id="deep_research_garmin_roam",
                interviewee_name="",
                turn_ids=[],
                metadata={
                    "source_type": SOURCE_TYPE,
                    "trust_weight": TRUST_WEIGHT,
                    "section": section_name,
                    "interviewee_name": "",
                    "generation": "",
                    "age": "",
                    "location": "",
                    "occupation": "",
                    "device": "",
                    "subreddit": "",
                    "username": "",
                    "report_title": report_title,
                    "chunk_index": str(chunk_index),
                },
                char_count=len(text),
                token_count=_estimate_tokens(text),
            )
        )
        current_blocks = []
        current_tokens = 0
        chunk_index += 1

    for block in blocks:
        block_tokens = _estimate_tokens(block)
        # Only treat as atomic if it's a genuine multi-row table; single-row
        # table fragments (< 50 tokens) accumulate with surrounding text.
        is_table = _is_table_block(block) and block_tokens >= 50

        if is_table:
            # Tables are atomic — flush current accumulation, then table alone
            _flush()
            current_blocks = [block]
            current_tokens = block_tokens
            _flush()
        elif not current_blocks and block_tokens > max_tokens:
            # Single oversized non-table block — its own chunk
            current_blocks = [block]
            current_tokens = block_tokens
            _flush()
        elif current_blocks and current_tokens + block_tokens > target_tokens:
            # Adding this block would exceed the soft target — flush first
            _flush()
            current_blocks = [block]
            current_tokens = block_tokens
        else:
            current_blocks.append(block)
            current_tokens += block_tokens

    _flush()
    return chunks


# ---------------------------------------------------------------------------
# Step 5 — Idempotent duplicate check
# ---------------------------------------------------------------------------


def filter_new_chunks(
    chunks: list[Chunk], collection: chromadb.Collection
) -> tuple[list[Chunk], int]:
    """Return (new_chunks, already_exist_count) using a single batch ID fetch.

    Args:
        chunks: Candidate chunks to check.
        collection: ChromaDB collection to check against.

    Returns:
        Tuple of (new chunks only, count of duplicates skipped).
    """
    if not chunks:
        return [], 0

    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])

    new_chunks = [c for c in chunks if c.id not in existing_ids]
    already_exist = len(chunks) - len(new_chunks)
    return new_chunks, already_exist


# ---------------------------------------------------------------------------
# Step 6 — Embed and add
# ---------------------------------------------------------------------------


def embed_and_insert(
    chunks: list[Chunk], collection: chromadb.Collection
) -> tuple[float, int]:
    """Embed via OpenAI text-embedding-3-small and insert into collection.

    Args:
        chunks: New chunks to embed and store.
        collection: Target ChromaDB collection.

    Returns:
        Tuple of (estimated cost in USD, total tokens embedded).

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    total_tokens = sum(c.token_count for c in chunks)
    cost_est = total_tokens * (0.02 / 1_000_000)
    print(f"  Estimated embedding cost: ${cost_est:.6f} ({total_tokens} tokens)")

    client = openai.OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c.text for c in batch],
        )
        embeddings.extend(e.embedding for e in resp.data)
        log.info("Embedded batch %d–%d of %d", i + 1, i + len(batch), len(chunks))

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_to_chroma_metadata(c) for c in chunks],
    )

    return cost_est, total_tokens


# ---------------------------------------------------------------------------
# Step 7 — Validation queries
# ---------------------------------------------------------------------------


def run_validation_queries(chroma_dir: Path) -> None:
    """Run 5 validation queries and print primary, corroboration, and deep_research hits."""
    print("\n── Validation Queries ──────────────────────────────────────\n")
    for query in VALIDATION_QUERIES:
        print(f'Query: "{query}"')
        results = search_with_trust_priority(query, chroma_dir)

        for label, hits in [
            ("primary (transcript)", results["primary"]),
            ("secondary (deep_research)", results.get("secondary", [])),
            ("corroboration (reddit)", results["corroboration"]),
        ]:
            for chunk, score in hits[:2]:
                section = (chunk.metadata.get("section") or "")[:25]
                src = chunk.metadata.get("source_type", "?")
                preview = chunk.text[:120].replace("\n", " | ")
                print(f"  [{label}] src={src:<12} section={section:<25} score={score:.3f}  {preview}")

        print()


# ---------------------------------------------------------------------------
# Step 8 — Summary banner
# ---------------------------------------------------------------------------


def print_summary(
    section_counts: Counter[str],
    already_exist: int,
    new_count: int,
    cost: float,
    total_tokens_today: int,
    collection: chromadb.Collection,
) -> None:
    """Print ingestion summary with per-source breakdown."""
    all_meta = collection.get(include=["metadatas"])
    source_counts: Counter[str] = Counter(
        m.get("source_type", "?") for m in all_meta["metadatas"]
    )

    w = 62
    print(f"\n── Deep Research Ingestion Complete {'─' * (w - 36)}")
    print(f"  Chunks already existed:    {already_exist}  (skipped)")
    print(f"  New chunks embedded:       {new_count}")
    print(f"  Embedding model:           {EMBEDDING_MODEL}")
    print(f"  Collection:                {COLLECTION_NAME}")
    print(f"  Estimated cost today:      ${cost:.6f}")
    print(f"  Tokens embedded today:     {total_tokens_today}")
    if section_counts:
        print()
        print("  Section breakdown (new chunks):")
        for section, count in section_counts.most_common():
            print(f"    {section[:50]:<50} {count}")
    print()
    print(f"  Total chunks in collection: {collection.count()}")
    print(f"    transcript:    {source_counts.get('transcript', 0)}")
    print(f"    reddit:        {source_counts.get('reddit', 0)}")
    print(f"    deep_research: {source_counts.get('deep_research', 0)}")
    print(f"{'─' * w}\n")


# ---------------------------------------------------------------------------
# Re-ingestion helper
# ---------------------------------------------------------------------------


def _clear_existing_source(collection: chromadb.Collection, source_type: str) -> int:
    """Delete all chunks with the given source_type from the collection.

    Args:
        collection: ChromaDB collection to clean.
        source_type: Value of the source_type metadata field to target.

    Returns:
        Number of chunks deleted.
    """
    existing = collection.get(include=["metadatas"])
    ids_to_delete = [
        id_
        for id_, meta in zip(existing["ids"], existing["metadatas"])
        if meta.get("source_type") == source_type
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} stale {source_type} chunks")
    return len(ids_to_delete)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pdf_paths = sorted(
        p for p in PDF_DIR.glob("*.pdf") if not p.name.startswith("~$")
    )
    if not pdf_paths:
        log.error("No PDF files found in %s", PDF_DIR)
        sys.exit(1)

    log.info("Found %d PDF(s) in %s", len(pdf_paths), PDF_DIR)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        "Collection '%s' has %d chunks before ingestion",
        COLLECTION_NAME,
        collection.count(),
    )

    _clear_existing_source(collection, SOURCE_TYPE)

    all_new_chunks: list[Chunk] = []
    all_section_counts: Counter[str] = Counter()
    total_already_exist = 0

    for pdf_path in pdf_paths:
        log.info("Processing %s", pdf_path.name)
        report_title = _report_title(pdf_path)
        log.info("Report title: %s", report_title)

        text = extract_and_clean_text(pdf_path)
        sections = parse_sections(text)

        print(f"\n  Sections in {pdf_path.name}:")
        all_chunks: list[Chunk] = []
        for section_name, section_text in sections:
            tok = _estimate_tokens(section_text)
            print(f"    {section_name[:50]:<50} {tok:>6} tokens")
            section_chunks = chunk_section(
                section_name, section_text, report_title, pdf_path
            )
            all_chunks.extend(section_chunks)

        log.info(
            "%d chunks from %s before dedup", len(all_chunks), pdf_path.name
        )

        new_chunks, already_exist = filter_new_chunks(all_chunks, collection)
        total_already_exist += already_exist
        print(
            f"\n  {already_exist} chunks already exist, "
            f"{len(new_chunks)} new chunks to embed"
        )

        if new_chunks:
            all_new_chunks.extend(new_chunks)
            for c in new_chunks:
                all_section_counts[c.metadata["section"]] += 1

    cost = 0.0
    total_tokens_today = 0
    if all_new_chunks:
        cost, total_tokens_today = embed_and_insert(all_new_chunks, collection)
        log.info("Inserted %d new chunks into '%s'", len(all_new_chunks), COLLECTION_NAME)
    else:
        print("\nNothing to add — all chunks already present.")

    print_summary(
        section_counts=all_section_counts,
        already_exist=total_already_exist,
        new_count=len(all_new_chunks),
        cost=cost,
        total_tokens_today=total_tokens_today,
        collection=collection,
    )

    all_meta = collection.get(include=["metadatas", "documents"])
    dr_docs = [
        doc
        for doc, meta in zip(all_meta["documents"], all_meta["metadatas"])
        if meta.get("source_type") == SOURCE_TYPE
    ]
    dr_count = len(dr_docs)
    avg_tokens = int(sum(_estimate_tokens(d) for d in dr_docs) / dr_count) if dr_count else 0
    print(f"deep_research chunks: {dr_count}")
    print(f"avg tokens per chunk: {avg_tokens}")

    # Artifact verification
    all_dr = collection.get(where={"source_type": "deep_research"}, include=["documents"])
    cid_found = [d for d in all_dr["documents"] if "(cid:" in d]
    cite_found = [d for d in all_dr["documents"] if "citeturn" in d]
    print(f"Chunks with (cid:) artifacts remaining: {len(cid_found)}")
    print(f"Chunks with citeturn artifacts remaining: {len(cite_found)}")
    if not cid_found and not cite_found:
        print("CLEAN — no artifacts in any chunk")
    else:
        print("WARNING — artifacts still present, check cleaning logic")

    run_validation_queries(CHROMA_DIR)


if __name__ == "__main__":
    main()
