"""Ingest Reddit CSV data into the existing ChromaDB icp_evidence collection.

Reads Raw_Reddit_Data.csv, builds Chunk objects, deduplicates against the
existing index, embeds only new chunks, and inserts them alongside transcript
chunks without touching any existing data.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import chromadb
import openai
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icp_agent.rag import Chunk, _to_chroma_metadata, search  # noqa: E402

load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "raw" / "test_reddit_data" / "Raw_Reddit_Data.csv"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma"
COLLECTION_NAME = "icp_evidence"
EMBEDDING_MODEL = "text-embedding-3-small"

VALIDATION_QUERIES: list[tuple[str, dict | None]] = [
    ("earbuds falling out during exercise or sweating", None),
    ("Garmin brand perception among young people", None),
    ("price sensitivity and value for money", {"source_type": "reddit"}),
]


# ---------------------------------------------------------------------------
# Generation inference
# ---------------------------------------------------------------------------


def _infer_generation(subreddit: str) -> str:
    mapping = {
        "r/GenZ": "Gen Z",
        "r/college": "Gen Z",
        "r/Earbuds": "",
        "r/headphones": "",
        "r/HeadphoneAdvice": "",
        "r/BudgetAudiophile": "",
    }
    return mapping.get(subreddit, "")


# ---------------------------------------------------------------------------
# CSV → Chunk
# ---------------------------------------------------------------------------


def _build_chunk(row: dict[str, str]) -> Chunk:
    subreddit = row["subreddit"].strip()
    username = row["username"].strip()
    thread = row["thread"].strip()

    text = f"{subreddit} | u/{username.lstrip('u/')}\n{thread}"

    chunk_id = hashlib.sha256(
        f"reddit_{subreddit}_{username}_{thread[:50]}".encode()
    ).hexdigest()[:16]

    metadata = {
        "source_type": "reddit",
        "trust_weight": 0.6,
        "subreddit": subreddit,
        "username": username,
        "section": "",
        "interviewee_name": username,
        "generation": _infer_generation(subreddit),
        "age": "",
        "location": "",
        "occupation": "",
        "device": "",
    }

    return Chunk(
        id=chunk_id,
        text=text,
        transcript_id=f"reddit_{subreddit.lstrip('r/')}",
        interviewee_name=username,
        turn_ids=[],
        metadata=metadata,
        char_count=len(text),
        token_count=int(len(text.split()) * 1.3),
    )


def load_chunks_from_csv(path: Path) -> tuple[list[Chunk], int]:
    """Read CSV and return (chunks, skipped_count)."""
    chunks: list[Chunk] = []
    skipped = 0

    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            thread = row.get("thread", "").strip()
            if not thread or len(thread) < 10:
                log.warning("Skipping short/empty row: username=%s", row.get("username", "?"))
                skipped += 1
                continue
            chunks.append(_build_chunk(row))

    return chunks, skipped


# ---------------------------------------------------------------------------
# Duplicate check
# ---------------------------------------------------------------------------


def filter_new_chunks(
    chunks: list[Chunk], collection: chromadb.Collection
) -> tuple[list[Chunk], int]:
    """Return only chunks whose IDs are not already in the collection."""
    if not chunks:
        return [], 0

    existing = collection.get(ids=[c.id for c in chunks])
    existing_ids = set(existing["ids"])

    new_chunks: list[Chunk] = []
    duplicate_count = 0
    for chunk in chunks:
        if chunk.id in existing_ids:
            log.warning("Duplicate skipped: id=%s  text=%.60s", chunk.id, chunk.text)
            duplicate_count += 1
        else:
            new_chunks.append(chunk)

    return new_chunks, duplicate_count


# ---------------------------------------------------------------------------
# Embed and insert
# ---------------------------------------------------------------------------


def embed_and_insert(chunks: list[Chunk], collection: chromadb.Collection) -> float:
    """Embed chunks via OpenAI and insert into collection. Returns estimated cost."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    client = openai.OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[c.text for c in batch],
        )
        embeddings.extend(e.embedding for e in resp.data)

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_to_chroma_metadata(c) for c in chunks],
    )

    total_tokens = sum(c.token_count for c in chunks)
    return total_tokens / 1_000_000 * 0.02


# ---------------------------------------------------------------------------
# Summary and validation
# ---------------------------------------------------------------------------


def print_summary(
    total_read: int,
    skipped: int,
    duplicates: int,
    new_count: int,
    cost: float,
    subreddit_counts: Counter,
    collection_total: int,
) -> None:
    w = 60
    print(f"\n── Reddit Ingestion Complete {'─' * (w - 28)}")
    print(f"  CSV rows read:        {total_read}")
    print(f"  Rows skipped:         {skipped}  (too short or empty)")
    print(f"  Chunks already exist: {duplicates}  (skipped duplicates)")
    print(f"  New chunks embedded:  {new_count}")
    print(f"  Embedding model:      {EMBEDDING_MODEL}")
    print(f"  Collection:           {COLLECTION_NAME}")
    print(f"  Estimated cost:       ${cost:.4f}")
    print()
    print("  Subreddit breakdown:")
    for sub, count in subreddit_counts.most_common():
        print(f"    {sub:<30} {count} chunks")
    print()
    print(f"  Total chunks in collection (after): {collection_total}")
    print(f"{'─' * w}\n")


def run_validation_queries(chroma_dir: Path) -> None:
    print("── Validation Queries ──────────────────────────────────────\n")
    for query, filt in VALIDATION_QUERIES:
        filter_label = f"  filter={filt}" if filt else ""
        print(f'Query: "{query}"{filter_label}')
        results = search(query, chroma_dir, top_k=5, filters=filt)
        if not results:
            print("  (no results)\n")
            continue
        for rank, (chunk, score) in enumerate(results, 1):
            source = chunk.metadata.get("source_type", "?")
            label = chunk.metadata.get("subreddit", "") or chunk.interviewee_name or "?"
            preview = chunk.text[:120].replace("\n", " | ")
            print(f"  [{rank}] source={source:<12} label={label:<22} score={score:.3f}  {preview}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("Loading CSV from %s", CSV_PATH)
    chunks, skipped = load_chunks_from_csv(CSV_PATH)
    total_read = len(chunks) + skipped
    log.info("CSV loaded: %d rows usable, %d skipped", len(chunks), skipped)

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("Collection '%s' has %d chunks before ingestion", COLLECTION_NAME, collection.count())

    new_chunks, duplicates = filter_new_chunks(chunks, collection)
    log.info("%d new chunks to embed, %d duplicates skipped", len(new_chunks), duplicates)

    subreddit_counts: Counter = Counter(
        c.metadata["subreddit"] for c in new_chunks
    )

    cost = 0.0
    if new_chunks:
        cost = embed_and_insert(new_chunks, collection)
        log.info("Inserted %d chunks into '%s'", len(new_chunks), COLLECTION_NAME)
    else:
        log.info("Nothing to insert — all chunks already present")

    print_summary(
        total_read=total_read,
        skipped=skipped,
        duplicates=duplicates,
        new_count=len(new_chunks),
        cost=cost,
        subreddit_counts=subreddit_counts,
        collection_total=collection.count(),
    )

    run_validation_queries(CHROMA_DIR)


if __name__ == "__main__":
    main()
