"""Chat pipeline: input normalization, prompt building, RAG retrieval, streaming.

Called by ``scripts/run_chat.py`` once an ICPDocument is on disk. Every
response is grounded in the same evidence corpus the synthesis pipeline
consumed, with per-turn retrieval and a persona-specific system prompt so
the model answers from the active archetype's point of view.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from icp_agent.models import ICPDocument, SubPersona
from icp_agent.pipeline import (
    _chunk_to_evidence_dict,
    _format_evidence_block,
    _strip_code_fence,
)
from icp_agent.rag import search_with_trust_priority

logger = logging.getLogger(__name__)

CHAT_MODEL = "claude-sonnet-4-5"
IMAGE_MODEL = "claude-sonnet-4-5"

_SUPPORTED_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
_SUPPORTED_FILE_EXTS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")
_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_IMAGE_SYSTEM_PROMPT = (
    "You are analyzing an ad creative or visual asset. Describe it concisely "
    "for a customer research system. Cover: headline copy (exact text), body "
    "copy (exact text), visual tone (dark/light, busy/minimal), color palette "
    "(2-3 dominant colors), what is shown (product shot vs lifestyle vs "
    "abstract), perceived price point (premium/mid/budget signals), and any "
    "cultural or demographic cues visible. Be specific and factual. No opinions."
)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You decompose a user question or product description into atomic retrieval "
    "queries for a customer evidence database. Each query must target ONE "
    "specific customer concern. Return ONLY a JSON array of strings. No "
    "explanation, no markdown."
)

_HISTORY_COMPRESSION_THRESHOLD = 14
_HISTORY_KEEP_TAIL = 6


# ---------------------------------------------------------------------------
# Function 1 — process_input
# ---------------------------------------------------------------------------


def process_input(
    text: str | None = None,
    image_path: Path | None = None,
    file_path: Path | None = None,
) -> str:
    """Normalize any supported input modality into plain text.

    Exactly one of ``text``, ``image_path``, or ``file_path`` must be supplied.

    Args:
        text: Raw typed user message.
        image_path: Ad creative or visual asset (jpg/jpeg/png/webp).
        file_path: Document upload (pdf/docx/txt/md).

    Returns:
        Text description suitable as the user turn of a chat exchange.

    Raises:
        ValueError: If zero or more than one input is provided, the text is
            empty after strip, the file format is unsupported, or a PDF's
            extracted text is too short to be useful.
    """
    provided = sum(x is not None for x in (text, image_path, file_path))
    if provided != 1:
        raise ValueError(
            "process_input requires exactly one of text, image_path, or file_path"
        )

    if text is not None:
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty.")
        return stripped

    if image_path is not None:
        return _process_image(image_path)

    assert file_path is not None
    return _process_file(file_path)


def _process_image(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext not in _SUPPORTED_IMAGE_EXTS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Supported: {list(_SUPPORTED_IMAGE_EXTS)}"
        )

    raw_bytes = image_path.read_bytes()
    encoded = base64.standard_b64encode(raw_bytes).decode("ascii")
    media_type = _IMAGE_MEDIA_TYPES[ext]

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=IMAGE_MODEL,
        max_tokens=800,
        system=_IMAGE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Describe this image for customer research analysis.",
                    },
                ],
            }
        ],
    )
    description = message.content[0].text
    logger.info(
        "Image processed: %s (%dkb)",
        image_path.name,
        len(raw_bytes) // 1024,
    )
    return description


def _process_file(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext not in _SUPPORTED_FILE_EXTS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: {list(_SUPPORTED_FILE_EXTS)}"
        )

    if ext == ".pdf":
        text = _extract_pdf(file_path)
        if len(text.strip()) < 50:
            raise ValueError(
                "PDF appears to be scanned or empty. Please paste content as text."
            )
    elif ext == ".docx":
        text = _extract_docx(file_path)
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    stripped = text.strip()
    logger.info(
        "File processed: %s (%d chars extracted)",
        file_path.name,
        len(stripped),
    )
    return stripped


def _extract_pdf(path: Path) -> str:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    import docx

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


# ---------------------------------------------------------------------------
# Function 2 — build_chat_system_prompt
# ---------------------------------------------------------------------------


def build_chat_system_prompt(
    icp_doc: ICPDocument,
    active_persona: SubPersona,
) -> str:
    """Assemble the system prompt for a given (ICP, persona) pair."""
    key_traits = "\n".join(f"- {t}" for t in active_persona.key_traits)
    motivations = "\n".join(f"- {m}" for m in active_persona.motivations)
    objections = "\n".join(f"- {o}" for o in active_persona.objections)

    top_pains = "\n".join(f"- {c.claim}" for c in icp_doc.pains[:3])
    top_gains = "\n".join(f"- {c.claim}" for c in icp_doc.gains[:3])
    vocab = ", ".join(icp_doc.vocabulary[:15])

    return (
        "You are a customer research assistant embodying a specific customer "
        "archetype from real research evidence. You speak from this persona's "
        "perspective when answering questions about products, ads, messaging, "
        "and experiences.\n\n"
        "GROUNDING RULES — follow these exactly:\n"
        "- Every claim you make must be grounded in the evidence provided in "
        "the user message. Do not invent opinions, preferences, or reactions.\n"
        "- If the evidence doesn't cover what's being asked, say so explicitly: "
        "'I don't have strong evidence on that from the research.'\n"
        "- Cite sources inline as [transcript: {interviewee_name}] or "
        "[reddit: {username}] or [research: {section}] after each claim.\n"
        "- Speak in first person as the persona, but ground every statement "
        "in real evidence. You are synthesizing real customer voices, not "
        "roleplaying fiction.\n"
        "- Never break character to discuss the research process itself unless "
        "directly asked.\n\n"
        "RESPONSE STYLE — non-negotiable:\n"
        "- Maximum 120 words. Hard limit. Cut ruthlessly.\n"
        "- No bold text, no headers, no numbered lists, no bullet points.\n"
        "- Write exactly like a real person talking — one flowing paragraph "
        "or two short ones maximum.\n"
        "- If generating options (like taglines), give 3 max, each on its "
        "own line with no formatting — just the tagline in quotes, "
        "then one sentence of context.\n"
        "- Never use the words 'cinematic', 'engineered', or 'premium' "
        "unless quoting the ad.\n"
        "- End every response with one concrete, specific sentence about "
        "what would change your mind.\n\n"
        "YOUR PERSONA:\n"
        f"Name: {active_persona.name}\n"
        f"Description: {active_persona.description}\n"
        f"Key traits:\n{key_traits}\n"
        f"Motivations:\n{motivations}\n"
        f"Objections:\n{objections}\n\n"
        "BROADER ICP CONTEXT:\n"
        f"Product: {icp_doc.product_name}\n"
        f"Core pains:\n{top_pains}\n"
        f"Core gains:\n{top_gains}\n"
        f"Customer vocabulary: {vocab}"
    )


# ---------------------------------------------------------------------------
# Function 3 — decompose_and_retrieve
# ---------------------------------------------------------------------------


def decompose_and_retrieve(
    user_text: str,
    active_persona: SubPersona,
    persist_dir: Path,
    transcript_top_k: int = 3,
    deep_research_top_k: int = 2,
    reddit_top_k: int = 2,
) -> str:
    """Break user_text into atomic queries, retrieve evidence, format for prompt.

    Returns a string ready to inject into the synthesis user message as
    RELEVANT EVIDENCE. Always returns something — falls back to the raw
    user_text if query decomposition fails both attempts.
    """
    queries = _decompose_user_text(user_text, active_persona)

    per_chunk: dict[str, dict[str, Any]] = {}
    for query in queries:
        try:
            results = search_with_trust_priority(
                query,
                persist_dir,
                transcript_top_k=transcript_top_k,
                deep_research_top_k=deep_research_top_k,
                reddit_top_k=reddit_top_k,
            )
        except Exception as exc:
            logger.warning("Chat query '%s' failed (%s) — skipping", query, exc)
            continue

        for lane_name in ("primary", "secondary", "corroboration"):
            for chunk, score in results.get(lane_name, []):
                existing = per_chunk.get(chunk.id)
                if existing is None:
                    per_chunk[chunk.id] = _chunk_to_evidence_dict(chunk, score, lane_name)
                else:
                    existing["matched_query_count"] += 1
                    if score > existing["score"]:
                        existing["score"] = score

    ranked = sorted(
        per_chunk.values(),
        key=lambda e: (-e["matched_query_count"], -e["score"]),
    )
    top = ranked[:10]

    counts = {"transcript": 0, "deep_research": 0, "reddit": 0}
    for c in top:
        src = c["source_type"]
        if src in counts:
            counts[src] += 1

    logger.info(
        "Retrieved %d unique chunks for %d queries "
        "(%d transcript / %d deep_research / %d reddit)",
        len(top),
        len(queries),
        counts["transcript"],
        counts["deep_research"],
        counts["reddit"],
    )

    return _format_evidence_block(top)


def _decompose_user_text(user_text: str, active_persona: SubPersona) -> list[str]:
    """Ask Sonnet to split user_text into 3-5 atomic queries; fall back on failure."""
    user_message = (
        f"Active persona: {active_persona.name} — {active_persona.description}\n\n"
        f"User question/input: {user_text}\n\n"
        "Break this into 3-5 atomic search queries that would retrieve the "
        "most relevant customer evidence for answering this question from "
        "this persona's perspective. Each query should be 5-10 words, "
        "conversational, targeting one concern.\n\n"
        'Return format: ["query 1", "query 2", "query 3"]'
    )

    client = anthropic.Anthropic()
    for attempt in range(2):
        message = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=500,
            system=_DECOMPOSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        response_text = message.content[0].text
        try:
            parsed = json.loads(_strip_code_fence(response_text))
        except (ValueError, TypeError):
            parsed = None

        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed) and parsed:
            return parsed

        if attempt == 0:
            user_message = (
                user_message
                + "\n\nYour previous response was not valid JSON. "
                "Return ONLY the JSON array, nothing else."
            )

    logger.warning("Query decomposition failed — using raw input as single query.")
    return [user_text]


# ---------------------------------------------------------------------------
# Function 4 — stream_response
# ---------------------------------------------------------------------------


def stream_response(
    user_message: str,
    evidence_block: str,
    history: list[dict[str, str]],
    system_prompt: str,
    model: str = CHAT_MODEL,
) -> str:
    """Stream a Sonnet response to stdout and update conversation history.

    Mutates ``history`` in place: compresses it when >= 14 messages (before
    the new turn is appended) and then appends the user + assistant pair.

    Args:
        user_message: The user's raw input (without evidence block).
        evidence_block: Formatted retrieval output to inject per-turn.
        history: Conversation so far (mutated in place).
        system_prompt: Persona-specific system prompt.
        model: Sonnet model id.

    Returns:
        The complete assistant response as a single string.
    """
    enriched_user = (
        f"{user_message}\n\n"
        f"RELEVANT EVIDENCE:\n{evidence_block}\n\n"
        "Answer from your persona's perspective. Cite sources inline after "
        "each claim as [transcript: name], [reddit: username], or "
        "[research: section]. If evidence doesn't cover the question, "
        "say so explicitly."
    )
    messages = list(history) + [{"role": "user", "content": enriched_user}]

    client = anthropic.Anthropic()
    response_parts: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            response_parts.append(text)
    print()  # newline after streaming completes

    response_text = "".join(response_parts)

    if len(history) >= _HISTORY_COMPRESSION_THRESHOLD:
        compressed = _summarize_history(history)
        history.clear()
        history.extend(compressed)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response_text})
    return response_text


# ---------------------------------------------------------------------------
# Private helper — _summarize_history
# ---------------------------------------------------------------------------


def _summarize_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compress all-but-last-six messages into a 2-message summary placeholder."""
    old = history[:-_HISTORY_KEEP_TAIL]
    tail = history[-_HISTORY_KEEP_TAIL:]

    transcript = "\n".join(f"[{m['role']}] {m['content']}" for m in old)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=400,
        system=(
            "Summarize this conversation history in 3-4 sentences, preserving "
            "the key topics discussed, decisions made, and any specific product "
            "details or creative concepts that were mentioned. Be concise."
        ),
        messages=[{"role": "user", "content": transcript}],
    )
    summary = message.content[0].text.strip()

    compressed = [
        {"role": "user", "content": f"[Earlier conversation summary: {summary}]"},
        {
            "role": "assistant",
            "content": "Understood, I have context from our earlier discussion.",
        },
    ] + tail

    logger.info(
        "History compressed: %d messages → 2 summary messages",
        len(old),
    )
    return compressed
