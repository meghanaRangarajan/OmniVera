"""Pipeline orchestrator: single source of truth for end-to-end synthesis runs.

Step 1 (this module) verifies that every prerequisite — a registered intake,
a persisted intake.json, and a populated ChromaDB index with mandatory
transcript evidence — exists before any paid API calls are made. Future steps
(research planning, synthesis, chat prep) will hang off the same orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
import chromadb

from pydantic import ValidationError

from icp_agent.intake import (
    Intake,
    IntakeStatus,
    load_intake,
    load_registry,
    register_intake,
)
from icp_agent.models import EvidencedClaim, ICPDocument, SubPersona, save_icp
from icp_agent.rag import search_with_trust_priority

logger = logging.getLogger(__name__)

COLLECTION_NAME = "icp_evidence"
_KNOWN_SOURCE_TYPES: tuple[str, ...] = ("transcript", "reddit", "deep_research")

SYNTHESIS_MODEL = "claude-sonnet-4-5"
SYNTHESIS_SECTIONS: tuple[str, ...] = (
    "demographics",
    "jobs_to_be_done",
    "pains",
    "gains",
    "objections",
    "vocabulary",
    "watering_holes",
)
_FALLBACK_QUERIES: dict[str, list[str]] = {
    "demographics": ["who are the typical customers for this product"],
    "jobs_to_be_done": ["what task or goal are customers trying to accomplish"],
    "pains": ["what problems frustrate customers about current solutions"],
    "gains": ["what outcomes do customers want from this product"],
    "objections": ["why would customers hesitate to buy this product"],
    "vocabulary": ["what words and phrases do customers use to describe this"],
    "watering_holes": ["where do customers hang out online to discuss this"],
}

_QUERY_SYSTEM_PROMPT = """\
You are a customer research specialist. Your job is to generate precise
semantic-search queries that will be run against a vector database of
real customer evidence (interview transcripts, Reddit threads, and deep
research reports).

Rules you MUST follow:
1. Every query must be specific to the actual product and market described
   in the user message. Do not emit generic queries. Bad: "what are customer
   pains". Good: "what frustrations do Gen Z consumers have with earbud
   fit during exercise".
2. Queries must use the vocabulary real customers would use — informal,
   concrete, emotional. Avoid corporate or marketing language ("value
   proposition", "pain points", "synergy").
3. Each query targets exactly ONE specific customer concern and is 5-10
   words, phrased conversationally.
4. Return ONLY a valid JSON object. No preamble, no explanation, no
   markdown code fences, no trailing text.
"""


def setup_logging(log_dir: Path = Path("data/logs")) -> Path:
    """Attach console (INFO) and file (DEBUG) handlers to the icp_agent logger.

    Idempotent — calling more than once does not stack duplicate handlers.

    Args:
        log_dir: Directory for the timestamped pipeline log file.

    Returns:
        Path to the log file created for this run.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_{timestamp}.log"

    root = logging.getLogger("icp_agent")
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.propagate = False

    return log_path


def verify_pipeline_inputs(intake_id: str, persist_dir: Path) -> dict[str, Any]:
    """Verify all synthesis prerequisites exist and are healthy.

    Runs before any paid API call. Confirms the intake is registered and
    saved to disk, and that the ChromaDB index has at least transcript
    evidence (mandatory) — deep_research and reddit produce warnings, not
    errors, so ingestion can be partial without blocking synthesis.

    Args:
        intake_id: Registered intake identifier, or the literal string
            "latest" to resolve against ``registry.latest_intake_id``.
        persist_dir: Directory containing the ChromaDB persistent store.

    Returns:
        Dict with keys: intake_id, intake, chunk_counts, warnings, verified.

    Raises:
        FileNotFoundError: Registry missing/empty, or intake.json not on disk.
        ValueError: intake_id not found in the registry.
        RuntimeError: Index empty or missing mandatory transcript chunks.
    """
    registry = load_registry()
    if not registry.intakes:
        raise FileNotFoundError(
            "No intake sessions found. Run the intake form first."
        )

    if intake_id == "latest":
        if registry.latest_intake_id is None:
            raise FileNotFoundError(
                "No intake sessions found. Run the intake form first."
            )
        intake_id = registry.latest_intake_id

    entry = next((e for e in registry.intakes if e.intake_id == intake_id), None)
    if entry is None:
        available = [e.intake_id for e in registry.intakes]
        raise ValueError(
            f"Intake '{intake_id}' not found. Available intakes: {available}"
        )

    intake_json_path = entry.intake_dir / "intake.json"
    if not intake_json_path.exists():
        raise FileNotFoundError(
            f"intake.json not found for intake '{intake_id}': {intake_json_path}"
        )

    intake: Intake = load_intake(entry.intake_dir)
    logger.debug("Loaded intake %s (%s)", intake.intake_id, intake.product_name)

    chunk_counts = _count_chunks_by_source(persist_dir)
    total = chunk_counts["total"]

    if total == 0:
        raise RuntimeError(
            "Index is empty. Run build_index.py and ingest scripts before synthesizing."
        )

    if chunk_counts["transcript"] == 0:
        raise RuntimeError(
            "No transcript chunks found. Transcripts are mandatory — "
            "run build_index.py before synthesizing."
        )

    warnings: list[str] = []
    if chunk_counts["deep_research"] == 0:
        msg = "No deep_research chunks found. Synthesis will proceed with lower confidence."
        logger.warning(msg)
        warnings.append(msg)
    if chunk_counts["reddit"] == 0:
        msg = "No reddit chunks found. Synthesis will proceed without corroboration."
        logger.warning(msg)
        warnings.append(msg)

    _print_summary(intake, chunk_counts, warnings)

    return {
        "intake_id": intake.intake_id,
        "intake": intake,
        "chunk_counts": chunk_counts,
        "warnings": warnings,
        "verified": True,
    }


def _count_chunks_by_source(persist_dir: Path) -> dict[str, int]:
    """Count chunks in the icp_evidence collection, grouped by source_type.

    Args:
        persist_dir: Directory of the ChromaDB persistent store.

    Returns:
        Dict with keys: total, transcript, reddit, deep_research.
    """
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = collection.count()
    counts: dict[str, int] = {"total": total}
    for src in _KNOWN_SOURCE_TYPES:
        counts[src] = 0

    if total == 0:
        return counts

    result = collection.get(include=["metadatas"])
    for meta in result["metadatas"]:
        src = meta.get("source_type")
        if src in counts:
            counts[src] += 1
    return counts


def _print_summary(
    intake: Intake,
    chunk_counts: dict[str, int],
    warnings: list[str],
) -> None:
    """Write the verification summary to stdout for the CLI user."""
    print(f"✓ Intake loaded: {intake.product_name} ({intake.intake_id})")
    print(
        f"✓ Index verified: {chunk_counts['total']} chunks "
        f"({chunk_counts['transcript']} transcript / "
        f"{chunk_counts['deep_research']} deep_research / "
        f"{chunk_counts['reddit']} reddit)"
    )
    for warning in warnings:
        print(f"⚠ {warning}")
    print("Ready for synthesis.")


def build_synthesis_queries(intake: Intake) -> dict[str, list[str]]:
    """Generate 2-3 retrieval queries per ICP section via one Sonnet call.

    Uses the intake context to produce queries tailored to the product and
    hypothesis, not generic ones. Retries once if the model returns
    non-JSON. Fills any missing section with a generic fallback query and
    logs a warning rather than failing.

    Args:
        intake: The loaded Intake object.

    Returns:
        Dict mapping each of the 7 ICP section names to a list of queries.

    Raises:
        ValueError: If the model returns invalid JSON on both attempts.
    """
    user_message = _build_query_user_message(intake)
    est_tokens = (len(_QUERY_SYSTEM_PROMPT) + len(user_message)) // 4
    logger.info("Decomposition prompt estimated tokens: ~%d", est_tokens)

    client = anthropic.Anthropic()
    response_text = _call_sonnet(client, user_message)

    parsed: dict[str, Any] | None = _try_parse_json(response_text)

    if parsed is None:
        logger.warning("First decomposition response was not valid JSON; retrying once")
        retry_user_message = (
            user_message
            + "\n\nYour previous response was not valid JSON. Return ONLY the JSON object, nothing else."
        )
        response_text = _call_sonnet(client, retry_user_message)
        parsed = _try_parse_json(response_text)

    if parsed is None:
        raise ValueError(
            f"Query decomposition failed. Raw response: {response_text[:500]}"
        )

    queries: dict[str, list[str]] = {}
    for section in SYNTHESIS_SECTIONS:
        value = parsed.get(section)
        if isinstance(value, list) and value:
            queries[section] = [str(q) for q in value]
        else:
            logger.warning(
                "Section '%s' missing from decomposition — using fallback query",
                section,
            )
            queries[section] = list(_FALLBACK_QUERIES[section])

    total_queries = sum(len(v) for v in queries.values())
    logger.info(
        "Query decomposition complete: %d queries across %d sections",
        total_queries,
        len(queries),
    )
    _print_query_summary(queries, total_queries)
    return queries


def _build_query_user_message(intake: Intake) -> str:
    """Assemble the user prompt with intake context and required JSON shape."""
    competitors = ", ".join(intake.competitors)
    research_goals = [g.value for g in intake.research_goals]
    required_shape = (
        "{\n"
        '  "demographics": ["query1", "query2"],\n'
        '  "jobs_to_be_done": ["query1", "query2"],\n'
        '  "pains": ["query1", "query2", "query3"],\n'
        '  "gains": ["query1", "query2"],\n'
        '  "objections": ["query1", "query2"],\n'
        '  "vocabulary": ["query1", "query2"],\n'
        '  "watering_holes": ["query1", "query2"]\n'
        "}"
    )
    return (
        f"Product name: {intake.product_name}\n"
        f"Product description: {intake.product_description}\n"
        f"ICP hypothesis: {intake.icp_hypothesis}\n"
        f"Competitors: {competitors}\n"
        f"Research goals: {research_goals}\n\n"
        "Generate 2-3 retrieval queries for each of the 7 ICP sections below.\n"
        "Return a JSON object exactly matching this shape (keys and list-of-string values):\n"
        f"{required_shape}\n\n"
        "Each query must be 5-10 words, conversational, and target ONE specific "
        "customer concern — phrased the way a real customer would say it, not the "
        "way a marketer would write it."
    )


def _call_sonnet(
    client: anthropic.Anthropic,
    user_message: str,
    system: str = _QUERY_SYSTEM_PROMPT,
    max_tokens: int = 1500,
    model: str = SYNTHESIS_MODEL,
) -> str:
    """Call Sonnet with the given system prompt and return raw response text."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding ```json ... ``` (or plain ```) fence if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(_strip_code_fence(text))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_parse_json_array(text: str) -> list[Any] | None:
    try:
        parsed = json.loads(_strip_code_fence(text))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _print_query_summary(queries: dict[str, list[str]], total: int) -> None:
    """Write the Step-2 summary to stdout for the CLI user."""
    print(f"✓ Queries built: {total} queries across 7 sections")
    label_width = max(len(s) for s in SYNTHESIS_SECTIONS) + 1
    for section in SYNTHESIS_SECTIONS:
        label = f"{section}:".ljust(label_width)
        print(f"  {label} {queries[section]}")


# ---------------------------------------------------------------------------
# Step 3 — evidence retrieval
# ---------------------------------------------------------------------------

MAX_CHUNKS_PER_SECTION = 8
LOW_CONFIDENCE_THRESHOLD = 3


def retrieve_evidence_for_synthesis(
    queries: dict[str, list[str]],
    persist_dir: Path,
    transcript_top_k: int = 5,
    deep_research_top_k: int = 3,
    reddit_top_k: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Run every query in every section and return ranked, deduplicated evidence.

    Each query gets its own call to ``search_with_trust_priority`` (which
    embeds the query once and fans out across transcript / deep_research /
    reddit lanes). Within a section, a chunk that surfaces for multiple
    queries gets ``matched_query_count`` incremented — a corroboration
    signal used as the primary ranking key. The same chunk may still
    appear in multiple sections; dedup is intentionally per-section.

    Args:
        queries: Section-name → list of semantic search queries.
        persist_dir: ChromaDB persistent store directory.
        transcript_top_k: Max transcript results per query.
        deep_research_top_k: Max deep_research results per query.
        reddit_top_k: Max reddit results per query.

    Returns:
        Dict mapping each of the 7 section names to a ranked list (up to
        ``MAX_CHUNKS_PER_SECTION`` entries) of evidence dicts.
    """
    total_queries = sum(len(qs) for qs in queries.values())
    logger.info(
        "Starting evidence retrieval for %d queries across %d sections",
        total_queries,
        len(SYNTHESIS_SECTIONS),
    )

    evidence: dict[str, list[dict[str, Any]]] = {}

    for section in SYNTHESIS_SECTIONS:
        section_queries = queries.get(section, [])
        per_chunk: dict[str, dict[str, Any]] = {}
        queries_run = 0

        for query in section_queries:
            try:
                results = search_with_trust_priority(
                    query,
                    persist_dir,
                    transcript_top_k=transcript_top_k,
                    deep_research_top_k=deep_research_top_k,
                    reddit_top_k=reddit_top_k,
                )
            except Exception as exc:
                logger.warning(
                    "Query '%s' in section '%s' failed (%s) — skipping",
                    query,
                    section,
                    exc,
                )
                continue

            queries_run += 1
            for lane_name in ("primary", "secondary", "corroboration"):
                for chunk, score in results.get(lane_name, []):
                    existing = per_chunk.get(chunk.id)
                    if existing is None:
                        per_chunk[chunk.id] = _chunk_to_evidence_dict(
                            chunk, score, lane_name
                        )
                    else:
                        existing["matched_query_count"] += 1
                        if score > existing["score"]:
                            existing["score"] = score

        ranked = sorted(
            per_chunk.values(),
            key=lambda e: (-e["matched_query_count"], -e["score"]),
        )
        top = ranked[:MAX_CHUNKS_PER_SECTION]
        evidence[section] = top

        logger.debug(
            "%s: %d unique chunks retrieved (%d queries run)",
            section,
            len(per_chunk),
            queries_run,
        )

        if not top:
            logger.warning(
                "Section '%s' returned 0 chunks. Claims for this section "
                "will be marked low confidence.",
                section,
            )
        elif len(top) < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Section '%s' has only %d chunks — synthesis may produce "
                "low-confidence claims.",
                section,
                len(top),
            )

    _print_evidence_summary(evidence)
    return evidence


def _chunk_to_evidence_dict(chunk: Any, score: float, lane: str) -> dict[str, Any]:
    """Convert a (Chunk, score, lane) triple into the evidence-dict shape."""
    source_type = chunk.metadata.get("source_type", "")
    if source_type == "reddit":
        interviewee_name = chunk.metadata.get("username", "") or ""
    else:
        interviewee_name = chunk.interviewee_name or ""
    return {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "source_type": source_type,
        "trust_weight": float(chunk.metadata.get("trust_weight", 0.0)),
        "interviewee_name": interviewee_name,
        "section": chunk.metadata.get("section", "") or "",
        "score": float(score),
        "matched_query_count": 1,
        "lane": lane,
    }


def _print_evidence_summary(evidence: dict[str, list[dict[str, Any]]]) -> None:
    """Write the Step-3 summary to stdout for the CLI user."""
    print("✓ Evidence retrieved")
    unique_ids: set[str] = set()
    label_width = max(len(s) for s in SYNTHESIS_SECTIONS) + 1
    for section in SYNTHESIS_SECTIONS:
        chunks = evidence.get(section, [])
        counts = {"transcript": 0, "deep_research": 0, "reddit": 0}
        for c in chunks:
            src = c["source_type"]
            if src in counts:
                counts[src] += 1
            unique_ids.add(c["chunk_id"])
        label = f"{section}:".ljust(label_width)
        print(
            f"  {label} {len(chunks)} chunks "
            f"(transcript: {counts['transcript']}, "
            f"deep_research: {counts['deep_research']}, "
            f"reddit: {counts['reddit']})"
        )
    print(f"  Total unique chunks across all sections: {len(unique_ids)}")
    print("  (Note: same chunk may appear in multiple sections — this is correct)")


def _format_evidence_block(section_chunks: list[dict[str, Any]]) -> str:
    """Format a section's chunks for direct injection into a synthesis prompt.

    Produces one header line per chunk with chunk_id, source_type, trust
    weight, source name, and matched_query_count, followed by the chunk
    text and a ``---`` separator. Consumed by Step 4 (synthesize_section).
    """
    blocks: list[str] = []
    for c in section_chunks:
        header = (
            f"[CHUNK {c['chunk_id']} | {c['source_type']} | "
            f"trust: {c['trust_weight']} | from: {c['interviewee_name']} | "
            f"matched {c['matched_query_count']} queries]"
        )
        blocks.append(f"{header}\n{c['text']}\n---")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Step 4 — per-section synthesis and sub-persona generation
# ---------------------------------------------------------------------------

MAX_SUB_PERSONAS = 5
_CONFIDENCE_EMOJI = {"high": "🟢", "medium": "🟡", "low": "🔴"}

_SECTION_GUIDANCE: dict[str, str] = {
    "demographics": (
        "Synthesize who these customers actually are — age, generation, "
        "occupation, lifestyle, device usage. Ground every claim in what "
        "interviewees stated."
    ),
    "jobs_to_be_done": (
        "Synthesize the core tasks or goals customers are trying to "
        "accomplish with this type of product."
    ),
    "pains": (
        "Synthesize the real frustrations, failures, and unmet needs "
        "customers experience today."
    ),
    "gains": (
        "Synthesize the outcomes and feelings customers want to achieve — "
        "what success looks like to them."
    ),
    "objections": (
        "Synthesize the specific reasons customers hesitate, doubt, or "
        "resist purchasing."
    ),
    "vocabulary": (
        "Extract the exact words, phrases, and expressions customers use — "
        "in their own language, not yours. Each claim.claim field should be "
        "a direct phrase or expression, not a description of language."
    ),
    "watering_holes": (
        "Synthesize where these customers spend time online, what "
        "communities they belong to, what media they consume."
    ),
}

_SECTION_SYSTEM_PROMPT = """\
You are a senior customer research analyst synthesizing qualitative evidence
into structured ICP claims.

You will be given a set of evidence chunks from customer interviews, Reddit
posts, and deep research reports. Each chunk has an ID, a source type, a
trust weight, and the raw text.

Rules you MUST follow:
1. Synthesize ONLY what the evidence actually says. Do not invent, infer
   beyond the evidence, or generalize to customers not represented in the
   chunks.
2. Every claim MUST cite at least one chunk_id from the provided evidence.
   A claim with no chunk_ids will be rejected.
3. Apply these confidence rules exactly:
   - "high"   → 2 or more transcript chunks support the claim
   - "medium" → 1 transcript chunk, OR 2+ reddit/deep_research chunks
   - "low"    → only 1 non-transcript chunk supports it
4. Return ONLY a valid JSON array of claim objects. No preamble, no
   explanation, no markdown code fences, no trailing text.
"""

_PERSONA_SYSTEM_PROMPT = """\
You are a customer segmentation specialist identifying distinct customer
archetypes from qualitative research evidence.

Look across ALL the synthesized claims and raw evidence below and identify
3-5 genuinely distinct customer types. Each persona must be meaningfully
different from the others — different motivations, different objections,
different relationship with the product.

Rules you MUST follow:
1. Do NOT invent personas. Every persona must be grounded in the evidence.
   Each persona must cite the chunk_ids that define it.
2. Each persona needs a vivid, specific name (not "Persona A") that
   captures their essence (e.g. "The Performance Purist", "The
   Budget-Conscious Student").
3. If the evidence only clearly supports 2 distinct archetypes, return 2.
   Do not force 5 personas from thin or homogeneous data.
4. Return ONLY a valid JSON array. No preamble, no explanation, no
   markdown fences, no trailing text.
"""


def synthesize_section(
    section_name: str,
    section_chunks: list[dict[str, Any]],
    intake: Intake,
    model: str = SYNTHESIS_MODEL,
) -> list[EvidencedClaim]:
    """Synthesize one ICP section into a list of evidenced claims.

    Single Sonnet call. Retries once on invalid JSON. Every claim must
    cite at least one real chunk_id from ``section_chunks`` — empty or
    hallucinated citations are logged and skipped, never crash.

    Args:
        section_name: One of the 7 SYNTHESIS_SECTIONS.
        section_chunks: Evidence dicts from retrieve_evidence_for_synthesis.
        intake: Loaded Intake for product context.
        model: Model id override (defaults to SYNTHESIS_MODEL).

    Returns:
        List of validated EvidencedClaim objects (possibly empty).
    """
    if not section_chunks:
        logger.warning(
            "Section '%s' has no evidence — skipping synthesis.", section_name
        )
        return []

    user_message = _build_section_user_message(section_name, section_chunks, intake)
    client = anthropic.Anthropic()

    response_text = _call_sonnet(
        client, user_message, system=_SECTION_SYSTEM_PROMPT, max_tokens=2500, model=model
    )
    parsed = _try_parse_json_array(response_text)

    if parsed is None:
        logger.warning(
            "Section '%s' first response was not a valid JSON array; retrying once",
            section_name,
        )
        retry = (
            user_message
            + "\n\nYour previous response was not valid JSON. Return ONLY the JSON array, nothing else."
        )
        response_text = _call_sonnet(
            client, retry, system=_SECTION_SYSTEM_PROMPT, max_tokens=2500, model=model
        )
        parsed = _try_parse_json_array(response_text)

    if parsed is None:
        logger.error(
            "Section '%s' synthesis failed — invalid JSON on both attempts. Raw: %s",
            section_name,
            response_text[:500],
        )
        return []

    valid_ids = {c["chunk_id"] for c in section_chunks}
    valid_claims: list[EvidencedClaim] = []
    rejected = 0

    for raw in parsed:
        if not isinstance(raw, dict):
            rejected += 1
            logger.warning("Claim rejected — not a JSON object: %r", raw)
            continue
        claim_text = str(raw.get("claim", ""))
        chunk_ids = raw.get("chunk_ids") or []
        if not chunk_ids:
            rejected += 1
            logger.warning(
                "Claim rejected — no chunk_ids cited: %s", claim_text[:80]
            )
            continue
        hallucinated = [cid for cid in chunk_ids if cid not in valid_ids]
        if hallucinated:
            rejected += 1
            logger.warning(
                "Claim rejected — hallucinated chunk_id %s: %s",
                hallucinated[0],
                claim_text[:80],
            )
            continue
        try:
            valid_claims.append(
                EvidencedClaim(
                    claim=claim_text,
                    chunk_ids=list(chunk_ids),
                    source_types=list(raw.get("source_types") or []),
                    confidence=raw.get("confidence", "low"),
                )
            )
        except ValidationError as exc:
            rejected += 1
            logger.warning(
                "Claim rejected — validation error (%s): %s",
                exc.errors()[0].get("msg", "invalid"),
                claim_text[:80],
            )

    if not valid_claims:
        logger.error(
            "Section '%s' produced 0 valid claims after validation.", section_name
        )
        _print_section_claims(section_name, valid_claims)
        return []

    logger.info(
        "Section '%s': %d claims (%d rejected)",
        section_name,
        len(valid_claims),
        rejected,
    )
    _print_section_claims(section_name, valid_claims)
    return valid_claims


def synthesize_sub_personas(
    all_evidence: dict[str, list[dict[str, Any]]],
    all_claims: dict[str, list[EvidencedClaim]],
    intake: Intake,
    model: str = SYNTHESIS_MODEL,
) -> list[SubPersona]:
    """Identify 3-5 distinct customer archetypes across all claims and evidence.

    Args:
        all_evidence: Section → list of evidence dicts (may overlap across sections).
        all_claims: Section → list of synthesized EvidencedClaim objects.
        intake: Loaded Intake for product context.
        model: Model id override.

    Returns:
        List of validated SubPersona objects (possibly empty).
    """
    context_block = _build_persona_context(all_evidence, all_claims)
    user_message = _build_persona_user_message(intake, context_block)
    client = anthropic.Anthropic()

    response_text = _call_sonnet(
        client, user_message, system=_PERSONA_SYSTEM_PROMPT, max_tokens=3000, model=model
    )
    parsed = _try_parse_json_array(response_text)

    if parsed is None:
        logger.warning("Sub-persona first response was not a valid JSON array; retrying once")
        retry = (
            user_message
            + "\n\nYour previous response was not valid JSON. Return ONLY the JSON array, nothing else."
        )
        response_text = _call_sonnet(
            client, retry, system=_PERSONA_SYSTEM_PROMPT, max_tokens=3000, model=model
        )
        parsed = _try_parse_json_array(response_text)

    if parsed is None:
        logger.error(
            "Sub-persona synthesis failed — invalid JSON on both attempts. Raw: %s",
            response_text[:500],
        )
        return []

    valid: list[SubPersona] = []
    for raw in parsed:
        if not isinstance(raw, dict):
            logger.warning("Persona rejected — not a JSON object: %r", raw)
            continue
        evidence_ids = raw.get("evidence_chunk_ids") or []
        key_traits = raw.get("key_traits") or []
        name = str(raw.get("name", ""))
        if not evidence_ids:
            logger.warning("Persona '%s' rejected — empty evidence_chunk_ids", name)
            continue
        if len(key_traits) < 2:
            logger.warning(
                "Persona '%s' rejected — fewer than 2 key_traits (%d)",
                name,
                len(key_traits),
            )
            continue
        try:
            valid.append(
                SubPersona(
                    name=name,
                    description=str(raw.get("description", "")),
                    key_traits=list(key_traits),
                    motivations=list(raw.get("motivations") or []),
                    objections=list(raw.get("objections") or []),
                    evidence_chunk_ids=list(evidence_ids),
                )
            )
        except ValidationError as exc:
            logger.warning(
                "Persona '%s' rejected — validation error: %s",
                name,
                exc.errors()[0].get("msg", "invalid"),
            )

    if not valid:
        logger.error("Sub-persona synthesis produced 0 valid personas after validation.")
        _print_persona_summary(valid)
        return []

    if len(valid) > MAX_SUB_PERSONAS:
        valid.sort(key=lambda p: len(p.evidence_chunk_ids), reverse=True)
        valid = valid[:MAX_SUB_PERSONAS]
        logger.info("Truncated to %d personas by evidence coverage.", MAX_SUB_PERSONAS)

    _print_persona_summary(valid)
    return valid


def _save_partial(
    intake_id: str,
    completed_sections: dict[str, list[EvidencedClaim]],
    base_dir: Path = Path("data/icp"),
) -> None:
    """Checkpoint completed-so-far section claims to disk.

    Overwrites ``base_dir/{intake_id}/icp_partial.json`` each call. Swallows
    any IO/serialization error with a WARNING so a partial-save failure
    never aborts an in-progress synthesis run.
    """
    try:
        out_dir = base_dir / intake_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "icp_partial.json"
        payload = {
            "intake_id": intake_id,
            "updated_at": datetime.now().isoformat(),
            "sections": {
                section: [claim.model_dump() for claim in claims]
                for section, claims in completed_sections.items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save partial checkpoint for %s: %s", intake_id, exc)


def _build_section_user_message(
    section_name: str,
    section_chunks: list[dict[str, Any]],
    intake: Intake,
) -> str:
    guidance = _SECTION_GUIDANCE.get(section_name, "")
    evidence_block = _format_evidence_block(section_chunks)
    required_shape = (
        "[\n"
        "  {\n"
        '    "claim": "string — the synthesized insight",\n'
        '    "chunk_ids": ["id1", "id2"],\n'
        '    "source_types": ["transcript", "reddit"],\n'
        '    "confidence": "high" | "medium" | "low"\n'
        "  }\n"
        "]"
    )
    return (
        f"Product name: {intake.product_name}\n"
        f"Product description: {intake.product_description}\n"
        f"ICP hypothesis: {intake.icp_hypothesis}\n\n"
        f"Section to synthesize: {section_name}\n"
        f"Guidance: {guidance}\n\n"
        "EVIDENCE:\n"
        f"{evidence_block}\n\n"
        "Produce 3-6 claims. If the evidence only supports fewer, produce "
        "fewer — do not pad with weak claims.\n\n"
        "Return a JSON array matching this shape exactly:\n"
        f"{required_shape}"
    )


def _build_persona_context(
    all_evidence: dict[str, list[dict[str, Any]]],
    all_claims: dict[str, list[EvidencedClaim]],
) -> str:
    lines = ["SYNTHESIZED CLAIMS:"]
    for section in SYNTHESIS_SECTIONS:
        lines.append(f"[{section}]")
        for claim in all_claims.get(section, []):
            lines.append(f"- {claim.claim} {claim.chunk_ids}")
        lines.append("")

    seen: dict[str, dict[str, Any]] = {}
    for section_chunks in all_evidence.values():
        for c in section_chunks:
            existing = seen.get(c["chunk_id"])
            if existing is None or c["score"] > existing["score"]:
                seen[c["chunk_id"]] = c
    top_chunks = sorted(
        seen.values(),
        key=lambda c: (c["trust_weight"], c["score"]),
        reverse=True,
    )[:15]

    lines.append("RAW EVIDENCE SAMPLE (top 15 chunks by trust_weight then score):")
    for c in top_chunks:
        lines.append(
            f"[CHUNK {c['chunk_id']} | {c['source_type']} | "
            f"trust: {c['trust_weight']} | from: {c['interviewee_name']}]"
        )
        lines.append(c["text"])
        lines.append("---")
    return "\n".join(lines)


def _build_persona_user_message(intake: Intake, context_block: str) -> str:
    required_shape = (
        "[\n"
        "  {\n"
        '    "name": "The Performance Purist",\n'
        '    "description": "2-3 sentence archetype summary",\n'
        '    "key_traits": ["trait 1", "trait 2", "trait 3"],\n'
        '    "motivations": ["motivation 1", "motivation 2"],\n'
        '    "objections": ["objection 1", "objection 2"],\n'
        '    "evidence_chunk_ids": ["id1", "id2", "id3"]\n'
        "  }\n"
        "]"
    )
    return (
        f"Product name: {intake.product_name}\n"
        f"Product description: {intake.product_description}\n"
        f"ICP hypothesis: {intake.icp_hypothesis}\n\n"
        f"{context_block}\n\n"
        "Identify 3-5 genuinely distinct customer archetypes.\n"
        "Return a JSON array matching this shape exactly:\n"
        f"{required_shape}"
    )


def _print_section_claims(
    section_name: str, claims: list[EvidencedClaim]
) -> None:
    print(f"✓ {section_name}: {len(claims)} claims synthesized")
    for claim in claims:
        emoji = _CONFIDENCE_EMOJI.get(claim.confidence, "⚪")
        preview = claim.claim[:60]
        ellipsis = "..." if len(claim.claim) > 60 else ""
        print(f"  [{emoji} {preview}{ellipsis}]")


def _print_persona_summary(personas: list[SubPersona]) -> None:
    print(f"✓ Sub-personas identified: {len(personas)}")
    for persona in personas:
        desc = persona.description[:80]
        ellipsis = "..." if len(persona.description) > 80 else ""
        print(f"  • {persona.name}: {desc}{ellipsis}")


# ---------------------------------------------------------------------------
# Step 5 — ICP assembly, confidence re-grading, save and registry update
# ---------------------------------------------------------------------------

_CLAIM_SECTIONS: tuple[str, ...] = (
    "demographics",
    "jobs_to_be_done",
    "pains",
    "gains",
    "objections",
    "watering_holes",
)


def assemble_and_save_icp(
    intake: Intake,
    completed_sections: dict[str, list[EvidencedClaim]],
    sub_personas: list[SubPersona],
    all_evidence: dict[str, list[dict[str, Any]]],
    base_dir: Path = Path("data/icp"),
) -> ICPDocument:
    """Finalize a synthesis run: re-grade, assemble, save, checkpoint-cleanup, register.

    Args:
        intake: Loaded Intake for this run.
        completed_sections: Section → list of EvidencedClaim objects.
        sub_personas: Identified SubPersona objects.
        all_evidence: Section → evidence dicts from Step 3 (needed for the
            chunk_id → source_type lookup used by confidence re-grading).
        base_dir: Root directory for ICP artifacts (defaults to data/icp).

    Returns:
        The assembled, persisted ICPDocument.
    """
    chunk_source_map = _build_chunk_source_map(all_evidence)
    regraded_sections = _regrade_sections(completed_sections, chunk_source_map)

    vocabulary_strings = [
        claim.claim for claim in regraded_sections.get("vocabulary", [])
    ]
    logger.info("Extracted %d vocabulary terms", len(vocabulary_strings))

    icp = ICPDocument(
        intake_id=intake.intake_id,
        product_name=intake.product_name,
        demographics=regraded_sections.get("demographics", []),
        jobs_to_be_done=regraded_sections.get("jobs_to_be_done", []),
        pains=regraded_sections.get("pains", []),
        gains=regraded_sections.get("gains", []),
        objections=regraded_sections.get("objections", []),
        vocabulary=vocabulary_strings,
        watering_holes=regraded_sections.get("watering_holes", []),
        sub_personas=sub_personas,
        status="draft",
    )

    saved_path = save_icp(icp, base_dir=base_dir)
    logger.info("ICP saved to %s", saved_path)

    partial_path = base_dir / intake.intake_id / "icp_partial.json"
    if partial_path.exists():
        try:
            os.remove(partial_path)
            logger.info("Partial checkpoint removed")
        except OSError as exc:
            logger.warning(
                "Failed to remove partial checkpoint %s: %s", partial_path, exc
            )

    try:
        register_intake(
            intake,
            IntakeStatus.ICP_SYNTHESIZED,
            base_dir=Path("data/inputs"),
        )
        logger.info("Intake status updated to icp_synthesized")
    except Exception as exc:
        logger.warning("Failed to update intake status: %s", exc)

    _print_final_summary(icp, saved_path)
    return icp


def _build_chunk_source_map(
    all_evidence: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Flatten all_evidence into a chunk_id → source_type lookup."""
    lookup: dict[str, str] = {}
    for chunks in all_evidence.values():
        for c in chunks:
            lookup[c["chunk_id"]] = c.get("source_type", "")
    return lookup


def _regrade_sections(
    completed_sections: dict[str, list[EvidencedClaim]],
    chunk_source_map: dict[str, str],
) -> dict[str, list[EvidencedClaim]]:
    """Override each claim's confidence via client-side rules.

    Rules (override whatever Sonnet returned):
        transcript_count >= 2  → "high"
        transcript_count == 1  → "medium"
        total chunks >= 2      → "medium"
        otherwise              → "low"
    """
    regraded_total = 0
    regraded_sections: dict[str, list[EvidencedClaim]] = {}
    for section, claims in completed_sections.items():
        regraded_claims: list[EvidencedClaim] = []
        for claim in claims:
            transcript_count = sum(
                1
                for cid in claim.chunk_ids
                if chunk_source_map.get(cid, "") == "transcript"
            )
            if transcript_count >= 2:
                new_confidence = "high"
            elif transcript_count == 1:
                new_confidence = "medium"
            elif len(claim.chunk_ids) >= 2:
                new_confidence = "medium"
            else:
                new_confidence = "low"
            regraded_claims.append(
                claim.model_copy(update={"confidence": new_confidence})
            )
            regraded_total += 1
        regraded_sections[section] = regraded_claims
    logger.info(
        "Re-graded %d claims across %d sections",
        regraded_total,
        len(regraded_sections),
    )
    return regraded_sections


def _print_final_summary(icp: ICPDocument, saved_path: Path) -> None:
    """Print the Step-5 completion banner."""
    claim_lists = [
        icp.demographics,
        icp.jobs_to_be_done,
        icp.pains,
        icp.gains,
        icp.objections,
        icp.watering_holes,
    ]
    total_claims = sum(len(cs) for cs in claim_lists) + len(icp.vocabulary)

    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    unique_chunks: set[str] = set()
    for claims in claim_lists:
        for claim in claims:
            confidence_counts[claim.confidence] += 1
            unique_chunks.update(claim.chunk_ids)
    for persona in icp.sub_personas:
        unique_chunks.update(persona.evidence_chunk_ids)

    print("══════════════════════════════════════════")
    print("✓ ICP Build Complete")
    print("══════════════════════════════════════════")
    print(f"Product:          {icp.product_name}")
    print(f"Intake ID:        {icp.intake_id}")
    print("──────────────────────────────────────────")
    print("Sections:         7 / 7")
    print(f"Total claims:     {total_claims}")
    print(f"Vocabulary terms: {len(icp.vocabulary)}")
    print(f"Sub-personas:     {len(icp.sub_personas)}")
    print("──────────────────────────────────────────")
    print(
        f"Confidence:       🟢 high: {confidence_counts['high']}  "
        f"🟡 medium: {confidence_counts['medium']}  "
        f"🔴 low: {confidence_counts['low']}"
    )
    print(f"Unique chunks cited: {len(unique_chunks)}")
    print("──────────────────────────────────────────")
    print(f"Saved to: {saved_path}")
    print("Status:   icp_synthesized")
    print("══════════════════════════════════════════")
