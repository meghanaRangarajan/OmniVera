"""Tests for src/icp_agent/rag.py"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from icp_agent.rag import Chunk, build_index, chunk_transcripts, search, search_with_trust_priority
from icp_agent.transcripts import Transcript, Turn


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FAKE_DIM = 8
_FAKE_EMBEDDING = [0.1] * _FAKE_DIM


def _make_transcript(
    tid: str,
    name: str,
    turns: list[tuple[str, str, str | None]],  # (speaker, text, section)
    generation: str = "Gen Z",
    age: str = "22",
) -> Transcript:
    """Build a minimal Transcript from (speaker, text, section) tuples."""
    turn_objs = [
        Turn(speaker=s, text=t, section=sec) for s, t, sec in turns
    ]
    return Transcript(
        id=tid,
        source_path=Path("fake.txt"),
        title=name,
        interviewee_name=name,
        interviewee_metadata={"generation": generation, "age": age},
        speakers=list(dict.fromkeys(s for s, _, _ in turns)),
        turns=turn_objs,
        raw_text="",
        loaded_at=datetime(2024, 1, 1),
    )


def _qa_turns(
    n_pairs: int,
    section: str | None = "INTRODUCTIONS",
    response_words: int = 30,
) -> list[tuple[str, str, str | None]]:
    """Generate n_pairs alternating INT/CUSTOMER turns."""
    turns = []
    for i in range(n_pairs):
        turns.append(("INT", f"Question {i} here?", section))
        turns.append(("CUSTOMER", " ".join(["word"] * response_words), section))
    return turns


# ---------------------------------------------------------------------------
# Test 1: Happy path — 2 transcripts × 20 turns → reasonable chunk count
# ---------------------------------------------------------------------------


def test_happy_path_chunk_count():
    t1 = _make_transcript("t1", "Alice", _qa_turns(10, response_words=40))
    t2 = _make_transcript("t2", "Bob", _qa_turns(10, response_words=40))
    chunks = chunk_transcripts([t1, t2], target_tokens=100, max_tokens=200)

    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"

    for chunk in chunks:
        assert chunk.transcript_id in {"t1", "t2"}
        assert len(chunk.turn_ids) >= 1
        assert chunk.char_count == len(chunk.text)
        assert chunk.token_count > 0
        assert "generation" in chunk.metadata
        assert "section" in chunk.metadata


# ---------------------------------------------------------------------------
# Test 2: Section boundaries — chunks must not span two sections
# ---------------------------------------------------------------------------


def test_chunks_respect_section_boundaries():
    sec_a = [("INT", "Question A?", "SECTION_A"), ("CUSTOMER", "Answer A.", "SECTION_A")] * 5
    sec_b = [("INT", "Question B?", "SECTION_B"), ("CUSTOMER", "Answer B.", "SECTION_B")] * 5
    transcript = _make_transcript("t1", "Alice", sec_a + sec_b)

    chunks = chunk_transcripts([transcript], target_tokens=50, max_tokens=150)

    for chunk in chunks:
        sections_in_chunk = {
            transcript.turns[idx].section for idx in chunk.turn_ids
        }
        assert len(sections_in_chunk) == 1, (
            f"Chunk spans multiple sections: {sections_in_chunk}, turn_ids={chunk.turn_ids}"
        )


# ---------------------------------------------------------------------------
# Test 3: Long single turn — oversized turn becomes its own chunk
# ---------------------------------------------------------------------------


def test_long_single_turn_becomes_own_chunk():
    long_text = " ".join(["word"] * 600)  # ~780 tokens via word-count estimate
    turns = [
        ("INT", "Can you elaborate?", "SECTION_A"),
        ("CUSTOMER", long_text, "SECTION_A"),
        ("INT", "Thanks. Anything else?", "SECTION_A"),
        ("CUSTOMER", "No that is all.", "SECTION_A"),
    ]
    transcript = _make_transcript("t1", "Alice", turns)

    chunks = chunk_transcripts([transcript], target_tokens=400, max_tokens=600)

    long_chunks = [c for c in chunks if "word word word" in c.text]
    assert len(long_chunks) == 1, "Long turn should be isolated in its own chunk"
    long_chunk = long_chunks[0]
    # The oversized customer turn should be the only customer turn in this chunk
    assert sum(1 for idx in long_chunk.turn_ids if transcript.turns[idx].speaker == "CUSTOMER") == 1


# ---------------------------------------------------------------------------
# Test 4: build_index + search — mocked OpenAI, real temp ChromaDB
# ---------------------------------------------------------------------------


def test_build_index_and_search(tmp_path, mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})

    turns = _qa_turns(5, response_words=20)
    transcript = _make_transcript("t1", "Alice", turns)
    chunks = chunk_transcripts([transcript], target_tokens=80, max_tokens=160)
    assert len(chunks) >= 1

    # Each chunk needs its own fake embedding; query needs one too
    n_chunks = len(chunks)
    build_embeddings = [[float(i % 8) / 8.0] * _FAKE_DIM for i in range(n_chunks)]
    query_embedding = [0.5] * _FAKE_DIM

    mock_client = mocker.MagicMock()
    build_resp = mocker.MagicMock()
    build_resp.data = [mocker.MagicMock(embedding=e) for e in build_embeddings]
    search_resp = mocker.MagicMock()
    search_resp.data = [mocker.MagicMock(embedding=query_embedding)]
    mock_client.embeddings.create.side_effect = [build_resp, search_resp]

    mocker.patch("icp_agent.rag.openai.OpenAI", return_value=mock_client)

    chroma_dir = tmp_path / "chroma"
    build_index(chunks, chroma_dir)
    results = search("test query", chroma_dir, top_k=3)

    assert len(results) >= 1
    chunk, score = results[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(score, float)
    assert chunk.transcript_id == "t1"
    assert chunk.interviewee_name == "Alice"
    # Confirm build call included correct number of texts
    build_call_args = mock_client.embeddings.create.call_args_list[0]
    assert len(build_call_args.kwargs["input"]) == n_chunks


# ---------------------------------------------------------------------------
# Test 5: Metadata filtering — filter by generation returns only matches
# ---------------------------------------------------------------------------


def test_metadata_filter(tmp_path, mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})

    gen_z = _make_transcript("tz", "Zara", _qa_turns(3, response_words=20), generation="Gen Z")
    millennial = _make_transcript("tm", "Mike", _qa_turns(3, response_words=20), generation="Millennial")

    all_chunks = chunk_transcripts([gen_z, millennial], target_tokens=60, max_tokens=120)
    gz_chunks = [c for c in all_chunks if c.metadata["generation"] == "Gen Z"]
    mil_chunks = [c for c in all_chunks if c.metadata["generation"] == "Millennial"]
    assert len(gz_chunks) >= 1
    assert len(mil_chunks) >= 1

    n_total = len(all_chunks)
    all_embeddings = [[float(i) / n_total] * _FAKE_DIM for i in range(n_total)]
    query_embedding = [0.5] * _FAKE_DIM

    mock_client = mocker.MagicMock()
    build_resp = mocker.MagicMock()
    build_resp.data = [mocker.MagicMock(embedding=e) for e in all_embeddings]
    search_resp = mocker.MagicMock()
    search_resp.data = [mocker.MagicMock(embedding=query_embedding)]
    mock_client.embeddings.create.side_effect = [build_resp, search_resp]

    mocker.patch("icp_agent.rag.openai.OpenAI", return_value=mock_client)

    chroma_dir = tmp_path / "chroma"
    build_index(all_chunks, chroma_dir)
    results = search(
        "some query",
        chroma_dir,
        top_k=20,
        filters={"generation": "Gen Z"},
    )

    assert len(results) == len(gz_chunks), (
        f"Expected {len(gz_chunks)} Gen Z results, got {len(results)}"
    )
    for chunk, score in results:
        assert chunk.metadata["generation"] == "Gen Z", (
            f"Filter leak: got generation={chunk.metadata['generation']!r}"
        )


# ---------------------------------------------------------------------------
# Test 6: search_with_trust_priority — correct structure, single embed call
# ---------------------------------------------------------------------------


def test_search_with_trust_priority(tmp_path, mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})

    transcript_chunk = _make_transcript("t1", "Alice", _qa_turns(3, response_words=20))
    reddit_turns = [("INT", "Reddit post?", "GENERAL"), ("CUSTOMER", "earbuds keep falling out", "GENERAL")]
    reddit_chunk_transcript = _make_transcript("r1", "u/reddit_user", reddit_turns)

    all_transcripts = [transcript_chunk, reddit_chunk_transcript]
    all_chunks = chunk_transcripts(all_transcripts, target_tokens=60, max_tokens=120)

    # Manually tag one chunk as reddit so we have both source_types in the index
    reddit_chunk = all_chunks[-1]
    reddit_chunk.metadata["source_type"] = "reddit"
    reddit_chunk.metadata["trust_weight"] = 0.6

    n_chunks = len(all_chunks)
    build_embeddings = [[float(i) / max(n_chunks, 1)] * _FAKE_DIM for i in range(n_chunks)]
    query_embedding = [0.5] * _FAKE_DIM

    mock_client = mocker.MagicMock()
    build_resp = mocker.MagicMock()
    build_resp.data = [mocker.MagicMock(embedding=e) for e in build_embeddings]
    query_resp = mocker.MagicMock()
    query_resp.data = [mocker.MagicMock(embedding=query_embedding)]
    mock_client.embeddings.create.side_effect = [build_resp, query_resp]

    mocker.patch("icp_agent.rag.openai.OpenAI", return_value=mock_client)

    chroma_dir = tmp_path / "chroma"
    build_index(all_chunks, chroma_dir)

    # Reset so search_with_trust_priority gets exactly one embed call
    mock_client.embeddings.create.reset_mock(side_effect=True)
    mock_client.embeddings.create.side_effect = [query_resp]

    result = search_with_trust_priority("earbuds falling out", chroma_dir)

    assert set(result.keys()) == {"primary", "secondary", "corroboration"}
    assert isinstance(result["primary"], list)
    assert isinstance(result["secondary"], list)
    assert isinstance(result["corroboration"], list)
    for item in result["primary"] + result["secondary"] + result["corroboration"]:
        assert isinstance(item, tuple) and len(item) == 2
        chunk, score = item
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    # Exactly one embedding API call inside search_with_trust_priority
    assert mock_client.embeddings.create.call_count == 1
