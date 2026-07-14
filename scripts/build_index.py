"""Build the ChromaDB index from transcripts.json and run proof-of-retrieval queries."""

from __future__ import annotations

import logging
import shutil
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Project root is one level up from scripts/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icp_agent.rag import Chunk, build_index, chunk_transcripts, search
from icp_agent.transcripts import load_transcripts_from_json

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

TRANSCRIPTS_JSON = ROOT / "data" / "processed" / "transcripts.json"
CHROMA_DIR = ROOT / "data" / "processed" / "chroma"

TEST_QUERIES: list[tuple[str, dict | None]] = [
    ("fit problems and earbuds falling out", None),
    ("how people feel about Garmin as a brand", None),
    ("Gen Z frustrations with premium pricing", {"generation": "Gen Z"}),
    ("value means peace of mind and reliability", None),
]


def print_separator(title: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def print_stats(chunks: list[Chunk]) -> None:
    print_separator("Chunk summary stats")

    total = len(chunks)
    avg_tokens = sum(c.token_count for c in chunks) / total if total else 0
    print(f"  Total chunks   : {total}")
    print(f"  Avg tokens     : {avg_tokens:.1f}")

    by_section: Counter = Counter(c.metadata.get("section", "") for c in chunks)
    print(f"\n  By section ({len(by_section)} sections):")
    for sec, count in sorted(by_section.items(), key=lambda x: -x[1]):
        label = sec if sec else "(none)"
        print(f"    {label:<40} {count:>4}")

    by_person: Counter = Counter(c.interviewee_name for c in chunks)
    print(f"\n  By interviewee ({len(by_person)} people):")
    for name, count in sorted(by_person.items(), key=lambda x: -x[1]):
        label = name if name else "(unknown)"
        print(f"    {label:<40} {count:>4}")


def run_queries(chroma_dir: Path) -> None:
    for query, filt in TEST_QUERIES:
        filter_label = f"  filter={filt}" if filt else ""
        print_separator(f'Query: "{query}"{filter_label}')

        results = search(query, chroma_dir, top_k=5, filters=filt)

        if not results:
            print("  (no results)")
            continue

        for rank, (chunk, score) in enumerate(results, 1):
            section = chunk.metadata.get("section", "—")
            name = chunk.interviewee_name or "unknown"
            preview = chunk.text[:200].replace("\n", " ↵ ")
            print(f"\n  [{rank}] {name}  |  {section}  |  score={score:.4f}")
            print(f"      {preview}…")


def build_index_from_disk(
    transcripts_json: Path = TRANSCRIPTS_JSON,
    chroma_dir: Path = CHROMA_DIR,
) -> dict:
    """Load transcripts, chunk, and rebuild the ChromaDB index from scratch.

    Always removes the existing chroma_dir before building so re-runs do not
    accumulate duplicates. Raises on any failure — does not call sys.exit so
    callers (the agent orchestrator) can surface errors cleanly.

    Args:
        transcripts_json: Path to the combined parsed-transcripts JSON.
        chroma_dir: Directory for the ChromaDB persistent store.

    Returns:
        {"chunks": int, "persist_dir": str}

    Raises:
        FileNotFoundError: If transcripts_json does not exist.
        Exception: Anything raised by chunk_transcripts or build_index
            (e.g. missing OPENAI_API_KEY) is propagated unchanged.
    """
    if not transcripts_json.exists():
        raise FileNotFoundError(
            f"transcripts file not found: {transcripts_json}"
        )

    print(f"Loading transcripts from {transcripts_json}")
    transcripts = load_transcripts_from_json(transcripts_json)
    print(f"Loaded {len(transcripts)} transcript(s)")

    chunks = chunk_transcripts(transcripts)
    print_stats(chunks)

    if chroma_dir.exists():
        print(f"\nRemoving existing index at {chroma_dir}")
        shutil.rmtree(chroma_dir)

    print(f"\nBuilding index → {chroma_dir}")
    build_index(chunks, chroma_dir)
    print("Index built.")

    return {"chunks": len(chunks), "persist_dir": str(chroma_dir)}


def resolve_transcripts_source() -> Path:
    """Find the parsed-transcripts file to index.

    Prefers the combined file at data/processed/transcripts.json. If that is
    absent, falls back to the latest intake's parsed_transcripts.json, which is
    what scripts/run_web_intake.py actually writes. This spares first-time users
    a manual copy between the two locations.

    Returns:
        Path to an existing parsed-transcripts JSON file.

    Raises:
        FileNotFoundError: If neither location has a usable file.
    """
    if TRANSCRIPTS_JSON.exists():
        return TRANSCRIPTS_JSON

    from icp_agent.intake import get_latest_intake

    intake = get_latest_intake(ROOT / "data" / "inputs")
    if intake is not None and intake.parsed_transcripts_path is not None:
        candidate = Path(intake.parsed_transcripts_path)
        if candidate.exists():
            log.info("Using parsed transcripts from latest intake: %s", intake.intake_id)
            return candidate

    raise FileNotFoundError(
        f"No parsed transcripts found.\n"
        f"  Looked in: {TRANSCRIPTS_JSON}\n"
        f"  and in the latest intake under: {ROOT / 'data' / 'inputs'}\n\n"
        f"Run `python scripts/run_web_intake.py` first to submit an intake and "
        f"upload transcripts, or copy a parsed-transcripts JSON to "
        f"{TRANSCRIPTS_JSON}.\n"
        f"See docs/GETTING_STARTED.md for the expected transcript format."
    )


def main() -> None:
    try:
        source = resolve_transcripts_source()
        build_index_from_disk(source, CHROMA_DIR)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    run_queries(CHROMA_DIR)

    print("\n✓ Done.\n")


if __name__ == "__main__":
    main()
