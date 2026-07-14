"""Tests for src/icp_agent/pipeline.py"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from icp_agent import pipeline
from icp_agent.intake import (
    Intake,
    IntakeRegistry,
    IntakeRegistryEntry,
    IntakeStatus,
    ResearchGoal,
)
from icp_agent.models import EvidencedClaim, SubPersona


def _make_intake(intake_id: str, intake_dir: Path) -> Intake:
    now = datetime.now()
    return Intake(
        product_name="TestProduct",
        product_description="A product that does something really meaningful for users.",
        company_name="TestCorp",
        icp_hypothesis="Our ideal customer is a mid-market SaaS buyer aged 30-45.",
        competitors=["Competitor A", "Competitor B"],
        research_goals=[ResearchGoal.GENERAL_RESEARCH],
        target_geography="United States",
        intake_id=intake_id,
        intake_dir=intake_dir,
        transcript_files=[],
        parsed_transcripts_path=None,
        created_at=now,
        updated_at=now,
        version=1,
    )


def _make_registry(
    intake_ids: list[str],
    latest: str | None,
    intake_dir: Path,
) -> IntakeRegistry:
    now = datetime.now()
    entries = [
        IntakeRegistryEntry(
            intake_id=iid,
            product_name="TestProduct",
            company_name="TestCorp",
            created_at=now,
            updated_at=now,
            version=1,
            intake_dir=intake_dir,
            transcript_count=0,
            turn_count=None,
            research_goals=[ResearchGoal.GENERAL_RESEARCH.value],
            status=IntakeStatus.INTAKE_COMPLETE,
        )
        for iid in intake_ids
    ]
    return IntakeRegistry(intakes=entries, latest_intake_id=latest)


def _patch_chroma(mocker, metadatas: list[dict]) -> None:
    """Patch chromadb.PersistentClient to return a collection with given metadatas."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = len(metadatas)
    mock_collection.get.return_value = {"metadatas": metadatas}

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mocker.patch.object(
        pipeline.chromadb, "PersistentClient", return_value=mock_client
    )


def _patch_loaders(mocker, registry: IntakeRegistry, intake: Intake | None) -> None:
    mocker.patch.object(pipeline, "load_registry", return_value=registry)
    if intake is not None:
        mocker.patch.object(pipeline, "load_intake", return_value=intake)


def test_verify_happy_path(mocker, tmp_path):
    intake_id = "intake_happy"
    intake = _make_intake(intake_id, tmp_path / intake_id)
    (tmp_path / intake_id).mkdir()
    (tmp_path / intake_id / "intake.json").write_text("{}", encoding="utf-8")
    registry = _make_registry([intake_id], intake_id, tmp_path / intake_id)

    _patch_loaders(mocker, registry, intake)
    _patch_chroma(
        mocker,
        [
            {"source_type": "transcript"},
            {"source_type": "transcript"},
            {"source_type": "reddit"},
            {"source_type": "deep_research"},
        ],
    )

    result = pipeline.verify_pipeline_inputs(intake_id, tmp_path / "chroma")

    assert result["verified"] is True
    assert result["intake_id"] == intake_id
    assert result["intake"].product_name == "TestProduct"
    assert result["chunk_counts"] == {
        "total": 4,
        "transcript": 2,
        "reddit": 1,
        "deep_research": 1,
    }
    assert result["warnings"] == []


def test_verify_latest_resolves(mocker, tmp_path):
    intake_id = "intake_real"
    intake = _make_intake(intake_id, tmp_path / intake_id)
    (tmp_path / intake_id).mkdir()
    (tmp_path / intake_id / "intake.json").write_text("{}", encoding="utf-8")
    registry = _make_registry([intake_id], intake_id, tmp_path / intake_id)

    _patch_loaders(mocker, registry, intake)
    _patch_chroma(
        mocker,
        [
            {"source_type": "transcript"},
            {"source_type": "reddit"},
            {"source_type": "deep_research"},
        ],
    )

    result = pipeline.verify_pipeline_inputs("latest", tmp_path / "chroma")
    assert result["intake_id"] == intake_id
    assert result["verified"] is True


def test_verify_empty_registry(mocker, tmp_path):
    _patch_loaders(mocker, IntakeRegistry(), None)
    with pytest.raises(FileNotFoundError, match="No intake sessions found"):
        pipeline.verify_pipeline_inputs("latest", tmp_path / "chroma")


def test_verify_unknown_intake_id(mocker, tmp_path):
    registry = _make_registry(["intake_a", "intake_b"], "intake_a", tmp_path)
    _patch_loaders(mocker, registry, None)

    with pytest.raises(ValueError) as exc:
        pipeline.verify_pipeline_inputs("intake_nope", tmp_path / "chroma")

    msg = str(exc.value)
    assert "intake_nope" in msg
    assert "intake_a" in msg
    assert "intake_b" in msg


def test_verify_empty_collection(mocker, tmp_path):
    intake_id = "intake_empty"
    intake = _make_intake(intake_id, tmp_path / intake_id)
    (tmp_path / intake_id).mkdir()
    (tmp_path / intake_id / "intake.json").write_text("{}", encoding="utf-8")
    registry = _make_registry([intake_id], intake_id, tmp_path / intake_id)

    _patch_loaders(mocker, registry, intake)
    _patch_chroma(mocker, [])

    with pytest.raises(RuntimeError, match="Index is empty"):
        pipeline.verify_pipeline_inputs(intake_id, tmp_path / "chroma")


def test_verify_no_transcripts(mocker, tmp_path):
    intake_id = "intake_no_tr"
    intake = _make_intake(intake_id, tmp_path / intake_id)
    (tmp_path / intake_id).mkdir()
    (tmp_path / intake_id / "intake.json").write_text("{}", encoding="utf-8")
    registry = _make_registry([intake_id], intake_id, tmp_path / intake_id)

    _patch_loaders(mocker, registry, intake)
    _patch_chroma(
        mocker,
        [
            {"source_type": "reddit"},
            {"source_type": "deep_research"},
        ],
    )

    with pytest.raises(RuntimeError, match="transcript chunks"):
        pipeline.verify_pipeline_inputs(intake_id, tmp_path / "chroma")


def test_verify_no_deep_research(mocker, tmp_path):
    intake_id = "intake_no_dr"
    intake = _make_intake(intake_id, tmp_path / intake_id)
    (tmp_path / intake_id).mkdir()
    (tmp_path / intake_id / "intake.json").write_text("{}", encoding="utf-8")
    registry = _make_registry([intake_id], intake_id, tmp_path / intake_id)

    _patch_loaders(mocker, registry, intake)
    _patch_chroma(
        mocker,
        [
            {"source_type": "transcript"},
            {"source_type": "reddit"},
        ],
    )

    result = pipeline.verify_pipeline_inputs(intake_id, tmp_path / "chroma")
    assert result["verified"] is True
    assert any("deep_research" in w for w in result["warnings"])
    assert not any("reddit" in w for w in result["warnings"])


def test_verify_no_reddit(mocker, tmp_path):
    intake_id = "intake_no_rd"
    intake = _make_intake(intake_id, tmp_path / intake_id)
    (tmp_path / intake_id).mkdir()
    (tmp_path / intake_id / "intake.json").write_text("{}", encoding="utf-8")
    registry = _make_registry([intake_id], intake_id, tmp_path / intake_id)

    _patch_loaders(mocker, registry, intake)
    _patch_chroma(
        mocker,
        [
            {"source_type": "transcript"},
            {"source_type": "deep_research"},
        ],
    )

    result = pipeline.verify_pipeline_inputs(intake_id, tmp_path / "chroma")
    assert result["verified"] is True
    assert any("reddit" in w for w in result["warnings"])
    assert not any("deep_research" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# build_synthesis_queries
# ---------------------------------------------------------------------------


def _mock_sonnet_response(text: str) -> MagicMock:
    message = MagicMock()
    message.content = [MagicMock(text=text)]
    return message


def _patch_anthropic(mocker, side_effect: list[str]) -> MagicMock:
    """Patch anthropic.Anthropic so .messages.create returns the queued texts."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _mock_sonnet_response(t) for t in side_effect
    ]
    mocker.patch.object(pipeline.anthropic, "Anthropic", return_value=mock_client)
    return mock_client


def _valid_queries_json(sections: tuple[str, ...] = pipeline.SYNTHESIS_SECTIONS) -> str:
    return json.dumps({
        section: [f"{section} query 1", f"{section} query 2"]
        for section in sections
    })


def test_build_synthesis_queries_returns_all_sections(mocker, tmp_path):
    intake = _make_intake("intake_q1", tmp_path / "intake_q1")
    _patch_anthropic(mocker, [_valid_queries_json()])

    result = pipeline.build_synthesis_queries(intake)

    assert set(result.keys()) == set(pipeline.SYNTHESIS_SECTIONS)
    assert len(result) == 7
    for section, queries in result.items():
        assert isinstance(queries, list)
        assert len(queries) > 0


def test_build_synthesis_queries_invalid_json_retries(mocker, tmp_path):
    intake = _make_intake("intake_q2", tmp_path / "intake_q2")
    mock_client = _patch_anthropic(mocker, ["not json", _valid_queries_json()])

    result = pipeline.build_synthesis_queries(intake)

    assert mock_client.messages.create.call_count == 2
    assert set(result.keys()) == set(pipeline.SYNTHESIS_SECTIONS)


def test_build_synthesis_queries_fails_twice_raises(mocker, tmp_path):
    intake = _make_intake("intake_q3", tmp_path / "intake_q3")
    _patch_anthropic(mocker, ["not json", "still not json"])

    with pytest.raises(ValueError, match="Query decomposition failed"):
        pipeline.build_synthesis_queries(intake)


def test_build_synthesis_queries_missing_section_uses_fallback(
    mocker, tmp_path, caplog
):
    intake = _make_intake("intake_q4", tmp_path / "intake_q4")
    partial_sections = tuple(s for s in pipeline.SYNTHESIS_SECTIONS if s != "watering_holes")
    _patch_anthropic(mocker, [_valid_queries_json(partial_sections)])

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        result = pipeline.build_synthesis_queries(intake)

    assert "watering_holes" in result
    assert result["watering_holes"] == pipeline._FALLBACK_QUERIES["watering_holes"]
    assert any(
        "watering_holes" in rec.message and "fallback" in rec.message
        for rec in caplog.records
    )


def test_build_synthesis_queries_uses_intake_context(mocker, tmp_path):
    intake = _make_intake("intake_q5", tmp_path / "intake_q5")
    mock_client = _patch_anthropic(mocker, [_valid_queries_json()])

    pipeline.build_synthesis_queries(intake)

    kwargs = mock_client.messages.create.call_args.kwargs
    user_content = kwargs["messages"][0]["content"]
    assert intake.product_name in user_content
    assert intake.product_description in user_content
    assert intake.icp_hypothesis in user_content
    assert kwargs["model"] == pipeline.SYNTHESIS_MODEL


# ---------------------------------------------------------------------------
# retrieve_evidence_for_synthesis
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, source_type: str = "transcript", text: str = "sample", name: str = "Alice") -> MagicMock:
    """Build a MagicMock that mimics the shape of a rag.Chunk."""
    trust = {"transcript": 1.0, "deep_research": 0.8, "reddit": 0.6}.get(source_type, 0.5)
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.text = text
    chunk.interviewee_name = "" if source_type == "reddit" else name
    chunk.metadata = {
        "source_type": source_type,
        "trust_weight": trust,
        "section": "",
        "username": name if source_type == "reddit" else "",
    }
    return chunk


def _lane_result(primary=None, secondary=None, corroboration=None) -> dict:
    return {
        "primary": primary or [],
        "secondary": secondary or [],
        "corroboration": corroboration or [],
    }


def _empty_queries() -> dict[str, list[str]]:
    return {section: [] for section in pipeline.SYNTHESIS_SECTIONS}


def test_retrieve_deduplicates_within_section(mocker, tmp_path):
    queries = _empty_queries()
    queries["pains"] = ["q1", "q2"]
    shared = _chunk("chunk_shared")

    mocker.patch.object(
        pipeline,
        "search_with_trust_priority",
        side_effect=[
            _lane_result(primary=[(shared, 0.70)]),
            _lane_result(primary=[(shared, 0.85)]),
        ],
    )

    evidence = pipeline.retrieve_evidence_for_synthesis(queries, tmp_path)

    pains = evidence["pains"]
    assert len(pains) == 1
    assert pains[0]["chunk_id"] == "chunk_shared"
    assert pains[0]["matched_query_count"] == 2
    assert pains[0]["score"] == 0.85  # max across matching queries


def test_retrieve_same_chunk_allowed_across_sections(mocker, tmp_path):
    queries = _empty_queries()
    queries["pains"] = ["q_pains"]
    queries["gains"] = ["q_gains"]
    shared = _chunk("chunk_cross")

    mocker.patch.object(
        pipeline,
        "search_with_trust_priority",
        side_effect=[
            _lane_result(primary=[(shared, 0.80)]),
            _lane_result(primary=[(shared, 0.80)]),
        ],
    )

    evidence = pipeline.retrieve_evidence_for_synthesis(queries, tmp_path)

    assert any(c["chunk_id"] == "chunk_cross" for c in evidence["pains"])
    assert any(c["chunk_id"] == "chunk_cross" for c in evidence["gains"])


def test_retrieve_ranks_by_matched_query_count_first(mocker, tmp_path):
    queries = _empty_queries()
    queries["pains"] = ["q1", "q2"]
    chunk_a = _chunk("chunk_A")  # higher score, matched once
    chunk_b = _chunk("chunk_B")  # lower score, matched twice

    mocker.patch.object(
        pipeline,
        "search_with_trust_priority",
        side_effect=[
            _lane_result(primary=[(chunk_a, 0.95), (chunk_b, 0.80)]),
            _lane_result(primary=[(chunk_b, 0.75)]),
        ],
    )

    evidence = pipeline.retrieve_evidence_for_synthesis(queries, tmp_path)

    pains = evidence["pains"]
    assert pains[0]["chunk_id"] == "chunk_B"
    assert pains[0]["matched_query_count"] == 2
    assert pains[1]["chunk_id"] == "chunk_A"
    assert pains[1]["matched_query_count"] == 1


def test_retrieve_empty_section_returns_empty_list(mocker, tmp_path, caplog):
    queries = _empty_queries()
    queries["watering_holes"] = ["q_wh"]

    mocker.patch.object(
        pipeline,
        "search_with_trust_priority",
        return_value=_lane_result(),
    )

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        evidence = pipeline.retrieve_evidence_for_synthesis(queries, tmp_path)

    assert evidence["watering_holes"] == []
    assert any(
        "watering_holes" in rec.message and "0 chunks" in rec.message
        for rec in caplog.records
    )


def test_retrieve_failed_query_is_skipped(mocker, tmp_path, caplog):
    queries = _empty_queries()
    queries["pains"] = ["bad_query", "good_query"]
    good = _chunk("chunk_good")

    def fake_search(query, *args, **kwargs):
        if query == "bad_query":
            raise RuntimeError("simulated chroma failure")
        return _lane_result(primary=[(good, 0.90)])

    mocker.patch.object(pipeline, "search_with_trust_priority", side_effect=fake_search)

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        evidence = pipeline.retrieve_evidence_for_synthesis(queries, tmp_path)

    pains = evidence["pains"]
    assert len(pains) == 1
    assert pains[0]["chunk_id"] == "chunk_good"
    assert any("bad_query" in rec.message for rec in caplog.records)


def test_retrieve_caps_at_8_chunks_per_section(mocker, tmp_path):
    queries = _empty_queries()
    queries["pains"] = ["q1"]
    # Build 15 unique chunks with descending scores 0.99, 0.98, ..., 0.85
    hits = [(_chunk(f"chunk_{i}"), 0.99 - i * 0.01) for i in range(15)]

    mocker.patch.object(
        pipeline,
        "search_with_trust_priority",
        return_value=_lane_result(primary=hits),
    )

    evidence = pipeline.retrieve_evidence_for_synthesis(queries, tmp_path)

    pains = evidence["pains"]
    assert len(pains) == 8
    assert [c["chunk_id"] for c in pains] == [f"chunk_{i}" for i in range(8)]
    assert pains[0]["score"] > pains[-1]["score"]


def test_format_evidence_block_structure():
    chunks = [
        {
            "chunk_id": "abc123",
            "text": "I hate when my earbuds fall out.",
            "source_type": "transcript",
            "trust_weight": 1.0,
            "interviewee_name": "Alice",
            "section": "",
            "score": 0.91,
            "matched_query_count": 2,
            "lane": "primary",
        },
        {
            "chunk_id": "def456",
            "text": "Bose is expensive but worth it.",
            "source_type": "reddit",
            "trust_weight": 0.6,
            "interviewee_name": "u_redditor",
            "section": "",
            "score": 0.77,
            "matched_query_count": 1,
            "lane": "corroboration",
        },
    ]

    block = pipeline._format_evidence_block(chunks)

    assert "abc123" in block
    assert "def456" in block
    assert "transcript" in block
    assert "reddit" in block
    assert "---" in block
    assert "I hate when my earbuds fall out." in block
    assert "Bose is expensive but worth it." in block


# ---------------------------------------------------------------------------
# synthesize_section / synthesize_sub_personas / _save_partial
# ---------------------------------------------------------------------------


def _evidence_chunk_dict(chunk_id: str, source_type: str = "transcript") -> dict:
    trust = {"transcript": 1.0, "deep_research": 0.8, "reddit": 0.6}.get(source_type, 0.5)
    return {
        "chunk_id": chunk_id,
        "text": f"Sample text for {chunk_id}.",
        "source_type": source_type,
        "trust_weight": trust,
        "interviewee_name": "Alice",
        "section": "",
        "score": 0.80,
        "matched_query_count": 1,
        "lane": "primary",
    }


def _claim_json(claim: str, chunk_ids: list, confidence: str = "high") -> dict:
    return {
        "claim": claim,
        "chunk_ids": chunk_ids,
        "source_types": ["transcript"],
        "confidence": confidence,
    }


def _persona_json(
    name: str,
    chunk_ids: list[str] | None = None,
    key_traits: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "description": f"The {name} archetype wants quality without overpaying.",
        "key_traits": key_traits if key_traits is not None else ["practical", "budget-aware"],
        "motivations": ["reliable performance"],
        "objections": ["too expensive"],
        "evidence_chunk_ids": chunk_ids if chunk_ids is not None else ["c1"],
    }


def test_synthesize_section_returns_evidenced_claims(mocker, tmp_path):
    intake = _make_intake("intake_ss1", tmp_path / "intake_ss1")
    chunks = [
        _evidence_chunk_dict("c1"),
        _evidence_chunk_dict("c2"),
        _evidence_chunk_dict("c3"),
    ]
    payload = json.dumps([
        _claim_json("Earbuds fall out during runs.", ["c1", "c2"]),
        _claim_json("Gen Z wants style too.", ["c2"]),
        _claim_json("Price matters a lot.", ["c3"]),
    ])
    _patch_anthropic(mocker, [payload])

    result = pipeline.synthesize_section("pains", chunks, intake)

    assert len(result) == 3
    assert all(isinstance(c, EvidencedClaim) for c in result)
    assert result[0].chunk_ids == ["c1", "c2"]


def test_synthesize_section_empty_chunks_returns_empty(mocker, tmp_path, caplog):
    intake = _make_intake("intake_ss2", tmp_path / "intake_ss2")
    mock_client = _patch_anthropic(mocker, [])  # should not be consumed

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        result = pipeline.synthesize_section("pains", [], intake)

    assert result == []
    assert mock_client.messages.create.call_count == 0
    assert any("no evidence" in rec.message for rec in caplog.records)


def test_synthesize_section_rejects_empty_chunk_ids(mocker, tmp_path, caplog):
    intake = _make_intake("intake_ss3", tmp_path / "intake_ss3")
    chunks = [_evidence_chunk_dict("c1")]
    payload = json.dumps([
        _claim_json("Valid claim.", ["c1"]),
        _claim_json("Empty-cite claim.", []),
    ])
    _patch_anthropic(mocker, [payload])

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        result = pipeline.synthesize_section("pains", chunks, intake)

    assert len(result) == 1
    assert result[0].claim == "Valid claim."
    assert any("no chunk_ids cited" in rec.message for rec in caplog.records)


def test_synthesize_section_rejects_hallucinated_chunk_id(mocker, tmp_path, caplog):
    intake = _make_intake("intake_ss4", tmp_path / "intake_ss4")
    chunks = [_evidence_chunk_dict("c1")]
    payload = json.dumps([
        _claim_json("Valid claim.", ["c1"]),
        _claim_json("Fake-cite claim.", ["fake_id_999"]),
    ])
    _patch_anthropic(mocker, [payload])

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        result = pipeline.synthesize_section("pains", chunks, intake)

    assert len(result) == 1
    assert result[0].claim == "Valid claim."
    assert any(
        "hallucinated" in rec.message and "fake_id_999" in rec.message
        for rec in caplog.records
    )


def test_synthesize_section_invalid_json_retries(mocker, tmp_path):
    intake = _make_intake("intake_ss5", tmp_path / "intake_ss5")
    chunks = [_evidence_chunk_dict("c1")]
    valid = json.dumps([_claim_json("Valid claim.", ["c1"])])
    mock_client = _patch_anthropic(mocker, ["not json", valid])

    result = pipeline.synthesize_section("pains", chunks, intake)

    assert mock_client.messages.create.call_count == 2
    assert len(result) == 1
    assert result[0].claim == "Valid claim."


def test_synthesize_section_fails_twice_returns_empty(mocker, tmp_path, caplog):
    intake = _make_intake("intake_ss6", tmp_path / "intake_ss6")
    chunks = [_evidence_chunk_dict("c1")]
    _patch_anthropic(mocker, ["not json", "still not json"])

    with caplog.at_level(logging.ERROR, logger="icp_agent.pipeline"):
        result = pipeline.synthesize_section("pains", chunks, intake)

    assert result == []
    assert any(
        "invalid JSON on both attempts" in rec.message for rec in caplog.records
    )


def test_synthesize_sub_personas_returns_valid_personas(mocker, tmp_path):
    intake = _make_intake("intake_sp1", tmp_path / "intake_sp1")
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_evidence["pains"] = [_evidence_chunk_dict("c1")]
    all_claims = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    payload = json.dumps([
        _persona_json("The Performance Purist"),
        _persona_json("The Budget Student", chunk_ids=["c1", "c2"]),
        _persona_json("The Style-First Buyer", chunk_ids=["c3"]),
    ])
    _patch_anthropic(mocker, [payload])

    result = pipeline.synthesize_sub_personas(all_evidence, all_claims, intake)

    assert len(result) == 3
    assert all(isinstance(p, SubPersona) for p in result)
    names = {p.name for p in result}
    assert "The Performance Purist" in names


def test_synthesize_sub_personas_truncates_above_5(mocker, tmp_path, caplog):
    intake = _make_intake("intake_sp2", tmp_path / "intake_sp2")
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_claims = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    # 7 personas with varying evidence_chunk_ids lengths
    personas = [
        _persona_json("P1", chunk_ids=["a"]),
        _persona_json("P2", chunk_ids=["a", "b", "c", "d"]),  # top
        _persona_json("P3", chunk_ids=["a", "b"]),
        _persona_json("P4", chunk_ids=["a", "b", "c"]),
        _persona_json("P5", chunk_ids=["a", "b", "c", "d", "e"]),  # top
        _persona_json("P6", chunk_ids=["a"]),
        _persona_json("P7", chunk_ids=["a", "b", "c", "d", "e", "f"]),  # top
    ]
    _patch_anthropic(mocker, [json.dumps(personas)])

    with caplog.at_level(logging.INFO, logger="icp_agent.pipeline"):
        result = pipeline.synthesize_sub_personas(all_evidence, all_claims, intake)

    assert len(result) == 5
    kept_names = [p.name for p in result]
    assert "P7" in kept_names  # 6 chunks
    assert "P5" in kept_names  # 5 chunks
    assert "P2" in kept_names  # 4 chunks
    assert "P1" not in kept_names  # 1 chunk
    assert "P6" not in kept_names  # 1 chunk
    assert any("Truncated to 5" in rec.message for rec in caplog.records)


def test_synthesize_sub_personas_skips_empty_chunk_ids(mocker, tmp_path, caplog):
    intake = _make_intake("intake_sp3", tmp_path / "intake_sp3")
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_claims = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    payload = json.dumps([
        _persona_json("Good One", chunk_ids=["a"]),
        _persona_json("Empty One", chunk_ids=[]),
        _persona_json("Another Good", chunk_ids=["b"]),
    ])
    _patch_anthropic(mocker, [payload])

    with caplog.at_level(logging.WARNING, logger="icp_agent.pipeline"):
        result = pipeline.synthesize_sub_personas(all_evidence, all_claims, intake)

    assert len(result) == 2
    assert {p.name for p in result} == {"Good One", "Another Good"}
    assert any(
        "Empty One" in rec.message and "empty evidence_chunk_ids" in rec.message
        for rec in caplog.records
    )


def test_save_partial_creates_file(tmp_path):
    claim = EvidencedClaim(
        claim="Pain claim.",
        chunk_ids=["c1"],
        source_types=["transcript"],
        confidence="high",
    )
    completed = {
        "pains": [claim],
        "gains": [claim],
    }

    pipeline._save_partial("intake_xyz", completed, base_dir=tmp_path)

    out_path = tmp_path / "intake_xyz" / "icp_partial.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["intake_id"] == "intake_xyz"
    assert "pains" in payload["sections"]
    assert "gains" in payload["sections"]
    assert payload["sections"]["pains"][0]["claim"] == "Pain claim."


# ---------------------------------------------------------------------------
# assemble_and_save_icp
# ---------------------------------------------------------------------------


def _assemble_claim(claim: str, chunk_ids: list[str], confidence: str = "low") -> EvidencedClaim:
    return EvidencedClaim(
        claim=claim,
        chunk_ids=chunk_ids,
        source_types=["transcript"] * len(chunk_ids),
        confidence=confidence,
    )


def _assemble_evidence(chunks: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """Build an all_evidence dict from (chunk_id, source_type) tuples."""
    ev = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    ev["pains"] = [
        _evidence_chunk_dict(cid, source_type=st) for cid, st in chunks
    ]
    return ev


def _patch_assemble_side_effects(mocker):
    """Stop assemble_and_save_icp's disk/registry side effects during tests."""
    mocker.patch.object(pipeline, "register_intake")
    return mocker


def test_assemble_regraded_high_confidence(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    intake = _make_intake("intake_as1", tmp_path / "intake_as1")
    # Claim with 2 transcript chunks — should be re-graded to "high"
    claim = _assemble_claim("c", ["t1", "t2"], confidence="low")
    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    completed["pains"] = [claim]
    all_evidence = _assemble_evidence([("t1", "transcript"), ("t2", "transcript")])

    icp = pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert icp.pains[0].confidence == "high"


def test_assemble_regraded_medium_one_transcript(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    intake = _make_intake("intake_as2", tmp_path / "intake_as2")
    claim = _assemble_claim("c", ["t1", "r1"], confidence="low")
    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    completed["pains"] = [claim]
    all_evidence = _assemble_evidence([("t1", "transcript"), ("r1", "reddit")])

    icp = pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert icp.pains[0].confidence == "medium"


def test_assemble_regraded_low(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    intake = _make_intake("intake_as3", tmp_path / "intake_as3")
    claim = _assemble_claim("c", ["r1"], confidence="high")
    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    completed["pains"] = [claim]
    all_evidence = _assemble_evidence([("r1", "reddit")])

    icp = pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert icp.pains[0].confidence == "low"


def test_assemble_vocabulary_extracts_strings(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    intake = _make_intake("intake_as4", tmp_path / "intake_as4")
    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    completed["vocabulary"] = [
        _assemble_claim("vibe", ["r1"]),
        _assemble_claim("lowkey", ["r1"]),
    ]
    all_evidence = _assemble_evidence([("r1", "reddit")])

    icp = pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert icp.vocabulary == ["vibe", "lowkey"]


def test_assemble_saves_icp_to_disk(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    save_mock = mocker.patch.object(
        pipeline,
        "save_icp",
        return_value=tmp_path / "mocked_save.json",
    )
    intake = _make_intake("intake_as5", tmp_path / "intake_as5")
    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}

    pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert save_mock.call_count == 1
    saved_doc = save_mock.call_args.args[0]
    assert saved_doc.intake_id == intake.intake_id


def test_assemble_deletes_partial_file(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    intake = _make_intake("intake_as6", tmp_path / "intake_as6")
    # Pre-create the partial file at the expected path
    partial_dir = tmp_path / intake.intake_id
    partial_dir.mkdir(parents=True)
    partial_path = partial_dir / "icp_partial.json"
    partial_path.write_text("{}", encoding="utf-8")

    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}

    pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert not partial_path.exists()


def test_assemble_updates_registry_status(mocker, tmp_path):
    register_mock = mocker.patch.object(pipeline, "register_intake")
    intake = _make_intake("intake_as7", tmp_path / "intake_as7")
    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}

    pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert register_mock.call_count == 1
    args, kwargs = register_mock.call_args
    # status passed positionally as 2nd arg
    assert IntakeStatus.ICP_SYNTHESIZED in args or kwargs.get("status") == IntakeStatus.ICP_SYNTHESIZED


def test_assemble_partial_deletion_failure_does_not_raise(mocker, tmp_path):
    _patch_assemble_side_effects(mocker)
    intake = _make_intake("intake_as8", tmp_path / "intake_as8")
    partial_dir = tmp_path / intake.intake_id
    partial_dir.mkdir(parents=True)
    (partial_dir / "icp_partial.json").write_text("{}", encoding="utf-8")

    mocker.patch.object(pipeline.os, "remove", side_effect=OSError("simulated"))

    completed = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}
    all_evidence = {s: [] for s in pipeline.SYNTHESIS_SECTIONS}

    icp = pipeline.assemble_and_save_icp(
        intake=intake,
        completed_sections=completed,
        sub_personas=[],
        all_evidence=all_evidence,
        base_dir=tmp_path,
    )

    assert icp is not None
    assert icp.intake_id == intake.intake_id
