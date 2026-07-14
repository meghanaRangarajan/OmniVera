"""Permanent validation script for the RAG retrieval layer.

Cross-source tests use search_with_trust_priority (two-stage, evidence hierarchy).
Single-source filtered tests use search directly (respects the filter as passed).
"""

from pathlib import Path

from dotenv import load_dotenv

from icp_agent.rag import search, search_with_trust_priority, Chunk

load_dotenv()

CHROMA_DIR = Path("data/processed/chroma")
COLLECTION = "icp_evidence"
_SEP = "═" * 62


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _fmt_transcript(chunk: Chunk, score: float, idx: int, prefix: str = "T") -> str:
    name = chunk.interviewee_name or chunk.metadata.get("_interviewee_name", "unknown")
    section = chunk.metadata.get("section", "")
    snippet = chunk.text[:120].replace("\n", " ")
    return f"  [{prefix}{idx}] {name} | {section} | {score:.3f}\n       {snippet}"


def _fmt_reddit(chunk: Chunk, score: float, idx: int, prefix: str = "R") -> str:
    subreddit = chunk.metadata.get("subreddit", "")
    username = chunk.metadata.get("username", chunk.interviewee_name or "")
    snippet = chunk.text[:120].replace("\n", " ")
    return f"  [{prefix}{idx}] {subreddit} | {username} | {score:.3f}\n       {snippet}"


def _fmt_generic(chunk: Chunk, score: float, idx: int) -> str:
    src = chunk.metadata.get("source_type", "")
    if src == "transcript":
        return _fmt_transcript(chunk, score, idx, prefix="")
    if src == "reddit":
        return _fmt_reddit(chunk, score, idx, prefix="")
    snippet = chunk.text[:120].replace("\n", " ")
    return f"  [{idx}] {chunk.interviewee_name} | {score:.3f}\n       {snippet}"


# ---------------------------------------------------------------------------
# Query runners
# ---------------------------------------------------------------------------


def _fmt_deep_research(chunk: Chunk, score: float, idx: int, prefix: str = "D") -> str:
    section = chunk.metadata.get("section", "")
    snippet = chunk.text[:120].replace("\n", " ")
    return f"  [{prefix}{idx}] {section} | {score:.3f}\n       {snippet}"


def run_cross_source_query(
    label: str,
    query: str,
    transcript_top_k: int = 5,
    reddit_top_k: int = 3,
) -> None:
    """Three-lane retrieval via search_with_trust_priority."""
    print(f"\n{_SEP}")
    print(f"TEST {label}")
    print(f'QUERY: "{query}"')
    print(_SEP)

    results = search_with_trust_priority(
        query,
        CHROMA_DIR,
        COLLECTION,
        transcript_top_k=transcript_top_k,
        reddit_top_k=reddit_top_k,
    )

    primary = results["primary"]
    secondary = results.get("secondary", [])
    corroboration = results["corroboration"]

    t_trust = primary[0][0].metadata.get("trust_weight", 1.0) if primary else 1.0
    print(f"\n── PRIMARY EVIDENCE (transcripts, trust: {t_trust}) ───────────────────")
    if primary:
        for i, (chunk, score) in enumerate(primary, 1):
            print(_fmt_transcript(chunk, score, i))
    else:
        print("  (no transcript results)")

    if secondary:
        d_trust = secondary[0][0].metadata.get("trust_weight", 0.8)
        print(f"\n── SECONDARY EVIDENCE (deep research, trust: {d_trust}) ──────────────")
        for i, (chunk, score) in enumerate(secondary, 1):
            print(_fmt_deep_research(chunk, score, i))

    r_trust = corroboration[0][0].metadata.get("trust_weight", 0.6) if corroboration else 0.6
    print(f"\n── CORROBORATION (reddit, trust: {r_trust}) ──────────────────────────────")
    if corroboration:
        for i, (chunk, score) in enumerate(corroboration, 1):
            print(_fmt_reddit(chunk, score, i))
    else:
        print("  (no reddit results)")


def run_filtered_query(
    label: str,
    query: str,
    filters: dict,
    top_k: int = 5,
) -> None:
    """Single-stage retrieval via search with explicit filter."""
    print(f"\n{_SEP}")
    print(f"TEST {label}")
    print(f'QUERY: "{query}"')
    print(f"FILTER: {filters}  TOP_K: {top_k}")
    print(_SEP)

    results = search(query, CHROMA_DIR, COLLECTION, top_k=top_k, filters=filters)

    if not results:
        print("  (no results)")
        return

    for i, (chunk, score) in enumerate(results, 1):
        print(_fmt_generic(chunk, score, i))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Cross-source (two-stage)
run_cross_source_query("1 — Fit problems", "earbuds keep falling out during workouts and exercise")
run_cross_source_query("2 — Bose brand perception", "Bose feels like a brand for older people not for me")
run_cross_source_query("6 — Purchase decision", "how I decided which earbuds to buy research reviews decision")
run_cross_source_query("7 — ANC as emotional need", "noise canceling headphones help me focus block out world anxiety")
run_cross_source_query("8 — Vague query robustness", "good earbuds", transcript_top_k=3, reddit_top_k=2)
run_cross_source_query("9 — Customer vocabulary", "vibes aesthetic does it match my style look")
run_cross_source_query("10 — Premium reliability", "reliability longevity premium product worth paying more for quality")

# Single-source filtered
run_filtered_query("3 — Price sensitivity", "spending too much money on premium headphones not worth it", filters={"source_type": "reddit"})
run_filtered_query("4 — Emotional role of audio", "music helps me regulate my mood and emotions throughout the day", filters={"source_type": "transcript"})
run_filtered_query("5 — Gen Z filtered", "what Gen Z wants from earbuds brands", filters={"generation": "Gen Z"}, top_k=6)
