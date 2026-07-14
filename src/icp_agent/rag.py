"""Chunking, embedding, and retrieval for ICP evidence."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import openai

from icp_agent.transcripts import Transcript

logger = logging.getLogger(__name__)

_INTERVIEWER_SPEAKERS: frozenset[str] = frozenset({"INT", "INTERVIEWER"})

# ---------------------------------------------------------------------------
# Token estimation — tiktoken when available, word-count fallback otherwise
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken

    _enc = _tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(text: str) -> int:
        return len(_enc.encode(text))

except ImportError:

    def _estimate_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A retrieval-ready slice of a transcript."""

    id: str
    text: str
    transcript_id: str
    interviewee_name: str
    turn_ids: list[int]
    metadata: dict[str, Any]
    char_count: int
    token_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_transcripts(
    transcripts: list[Transcript],
    target_tokens: int = 400,
    max_tokens: int = 600,
) -> list[Chunk]:
    """Convert transcripts into retrieval-ready chunks.

    Args:
        transcripts: Parsed Transcript objects to chunk.
        target_tokens: Soft target — stop adding turns once this is exceeded
            and the next turn is an interviewer question (natural boundary).
        max_tokens: Hard ceiling — never add a turn that would push the chunk
            past this limit.  A single turn that already exceeds max_tokens
            becomes its own chunk.

    Returns:
        Ordered list of Chunk objects across all transcripts.
    """
    logger.info("Chunking %d transcript(s)", len(transcripts))
    all_chunks: list[Chunk] = []

    for transcript in transcripts:
        turns = transcript.turns
        n = len(turns)
        i = 0

        while i < n:
            current_section = turns[i].section
            indexed: list[tuple[int, Any]] = []
            chunk_tokens = 0
            j = i

            while j < n:
                turn = turns[j]
                turn_text = f"{turn.speaker}: {turn.text}"
                turn_tokens = _estimate_tokens(turn_text)

                # Section boundary — stop before crossing it
                if indexed and turn.section != current_section:
                    break

                # Single oversized turn — include alone and stop
                if not indexed and turn_tokens > max_tokens:
                    indexed.append((j, turn))
                    j += 1
                    break

                # Hard ceiling — stop before adding this turn
                if indexed and chunk_tokens + turn_tokens > max_tokens:
                    break

                indexed.append((j, turn))
                chunk_tokens += turn_tokens
                j += 1

                # Past soft target and the next turn opens a new Q — natural break
                if chunk_tokens >= target_tokens and j < n:
                    if turns[j].speaker in _INTERVIEWER_SPEAKERS:
                        break

            if not indexed:
                i += 1
                continue

            all_chunks.append(_make_chunk(transcript, indexed))
            i = j

    logger.info("Chunking done: %d chunks produced", len(all_chunks))
    return all_chunks


def build_index(
    chunks: list[Chunk],
    persist_dir: Path,
    collection_name: str = "icp_evidence",
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Embed all chunks via OpenAI and store in a persistent ChromaDB collection.

    Args:
        chunks: Chunks to embed and store.
        persist_dir: Directory for the ChromaDB persistent store.
        collection_name: Name of the ChromaDB collection.
        embedding_model: OpenAI embedding model to use.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    total_tokens = sum(c.token_count for c in chunks)
    cost_est = total_tokens / 1_000_000 * 0.02
    logger.info(
        "Starting embedding: %d chunks, ~%d tokens, est. cost $%.4f",
        len(chunks),
        total_tokens,
        cost_est,
    )

    client = openai.OpenAI(api_key=api_key)
    embeddings: list[list[float]] = []

    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        resp = client.embeddings.create(
            model=embedding_model,
            input=[c.text for c in batch],
        )
        embeddings.extend(e.embedding for e in resp.data)

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_to_chroma_metadata(c) for c in chunks],
    )

    logger.info(
        "Index built: %d chunks in collection '%s'",
        len(chunks),
        collection_name,
    )


def search(
    query: str,
    persist_dir: Path,
    collection_name: str = "icp_evidence",
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    embedding_model: str = "text-embedding-3-small",
    *,
    _query_embedding: list[float] | None = None,
) -> list[tuple[Chunk, float]]:
    """Return top_k chunks most similar to the query, with similarity scores.

    Args:
        query: Natural-language search string.
        persist_dir: Directory of the ChromaDB persistent store.
        collection_name: Name of the ChromaDB collection.
        top_k: Maximum number of results to return.
        filters: ChromaDB ``where`` clause, e.g. ``{"generation": "Gen Z"}``.
        embedding_model: OpenAI embedding model to use.
        _query_embedding: Pre-computed query vector. When provided, skips the
            OpenAI embedding call. Used by search_with_trust_priority to avoid
            making two separate API calls for the same query.

    Returns:
        List of (Chunk, similarity_score) pairs, highest score first.

    Raises:
        ValueError: If OPENAI_API_KEY is not set and no pre-computed embedding
            is supplied.
    """
    if _query_embedding is not None:
        query_embedding = _query_embedding
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment")
        client = openai.OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=embedding_model, input=[query])
        query_embedding = resp.data[0].embedding

    chroma = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    kwargs: dict[str, Any] = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    output: list[tuple[Chunk, float]] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunk = _from_chroma_result(doc, meta)
        score = 1.0 - dist  # cosine distance [0,1] → similarity score [1,0]
        output.append((chunk, score))

    return output


def search_with_trust_priority(
    query: str,
    persist_dir: Path,
    collection_name: str = "icp_evidence",
    transcript_top_k: int = 5,
    deep_research_top_k: int = 3,
    reddit_top_k: int = 3,
    embedding_model: str = "text-embedding-3-small",
) -> dict[str, list[tuple[Chunk, float]]]:
    """Three-lane retrieval that enforces evidence hierarchy.

    Embeds the query once and reuses the vector for all three filtered
    searches.  Lanes are ordered by trust weight: transcripts (primary,
    trust 1.0) → deep research reports (secondary, trust 0.8) → Reddit
    (corroboration, trust 0.6).  If a lane has no matching chunks (e.g.
    deep research not yet ingested) it returns an empty list without
    raising.  Used by the synthesizer and chat assistant.

    Args:
        query: Natural-language search string.
        persist_dir: Directory of the ChromaDB persistent store.
        collection_name: Name of the ChromaDB collection.
        transcript_top_k: Max transcript results to return.
        deep_research_top_k: Max deep-research results to return.
        reddit_top_k: Max Reddit results to return.
        embedding_model: OpenAI embedding model to use.

    Returns:
        {
            "primary":       [(Chunk, score), ...],  # transcripts, trust 1.0
            "secondary":     [(Chunk, score), ...],  # deep_research, trust 0.8
            "corroboration": [(Chunk, score), ...],  # reddit, trust 0.6
        }

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment")

    client = openai.OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=embedding_model, input=[query])
    embedding = resp.data[0].embedding

    primary = search(
        query,
        persist_dir,
        collection_name,
        top_k=transcript_top_k,
        filters={"source_type": "transcript"},
        embedding_model=embedding_model,
        _query_embedding=embedding,
    )
    secondary = search(
        query,
        persist_dir,
        collection_name,
        top_k=deep_research_top_k,
        filters={"source_type": "deep_research"},
        embedding_model=embedding_model,
        _query_embedding=embedding,
    )
    corroboration = search(
        query,
        persist_dir,
        collection_name,
        top_k=reddit_top_k,
        filters={"source_type": "reddit"},
        embedding_model=embedding_model,
        _query_embedding=embedding,
    )

    return {"primary": primary, "secondary": secondary, "corroboration": corroboration}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    transcript: Transcript,
    indexed: list[tuple[int, Any]],
) -> Chunk:
    turn_ids = [idx for idx, _ in indexed]
    turns = [t for _, t in indexed]
    text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)

    chunk_id = hashlib.sha256(
        f"{transcript.id}:{turn_ids[0]}:{turn_ids[-1]}".encode()
    ).hexdigest()[:16]

    meta = transcript.interviewee_metadata
    metadata: dict[str, Any] = {
        "age": meta.get("age") or "",
        "generation": meta.get("generation") or "",
        "location": meta.get("location") or "",
        "occupation": meta.get("occupation") or "",
        "device": meta.get("device") or "",
        "section": turns[0].section or "",
        "source_type": transcript.source_type,
        "trust_weight": transcript.trust_weight,
    }

    return Chunk(
        id=chunk_id,
        text=text,
        transcript_id=transcript.id,
        interviewee_name=transcript.interviewee_name or "",
        turn_ids=turn_ids,
        metadata=metadata,
        char_count=len(text),
        token_count=_estimate_tokens(text),
    )


def _to_chroma_metadata(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        k: (v if v is not None else "") for k, v in chunk.metadata.items()
    }
    meta["_id"] = chunk.id
    meta["_transcript_id"] = chunk.transcript_id
    meta["_interviewee_name"] = chunk.interviewee_name
    meta["_turn_ids"] = ",".join(str(t) for t in chunk.turn_ids)
    meta["_char_count"] = chunk.char_count
    meta["_token_count"] = chunk.token_count
    return meta


_ICP_META_KEYS = frozenset(
    {
        "age", "generation", "location", "occupation", "device",
        "section", "source_type", "trust_weight",
        "subreddit", "username",  # reddit-specific fields
    }
)


def _from_chroma_result(doc: str, meta: dict[str, Any]) -> Chunk:
    turn_ids_str = meta.get("_turn_ids", "")
    turn_ids = [int(x) for x in turn_ids_str.split(",") if x]
    icp_meta = {k: meta[k] for k in _ICP_META_KEYS if k in meta}
    return Chunk(
        id=meta.get("_id", ""),
        text=doc,
        transcript_id=meta.get("_transcript_id", ""),
        interviewee_name=meta.get("_interviewee_name", ""),
        turn_ids=turn_ids,
        metadata=icp_meta,
        char_count=int(meta.get("_char_count", len(doc))),
        token_count=int(meta.get("_token_count", 0)),
    )
