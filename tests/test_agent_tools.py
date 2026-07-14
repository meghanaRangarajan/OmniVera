"""Tests for src/icp_agent/agent/tools.py.

Every Anthropic / OpenAI / ChromaDB call is mocked. The blocking
``chat_web.run`` is mocked to return immediately so launch_chat_server
behaves like an early-return path in tests.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from icp_agent import chat_web
from icp_agent.agent import tools
from icp_agent.intake import (
    Intake,
    IntakeRegistry,
    IntakeRegistryEntry,
    IntakeStatus,
    ResearchGoal,
)
from icp_agent.models import EvidencedClaim, ICPDocument, SubPersona


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_intake(intake_id: str, intake_dir: Path) -> Intake:
    now = datetime.now()
    return Intake(
        product_name="TestProduct",
        product_description="A product that does something useful for users.",
        company_name="TestCo",
        icp_hypothesis="Mid-market SaaS buyers aged 30-45 are our ICP.",
        competitors=["Competitor A", "Competitor B"],
        research_goals=[ResearchGoal.GENERAL_RESEARCH],
        target_geography="US",
        intake_id=intake_id,
        intake_dir=intake_dir,
        transcript_files=[],
        parsed_transcripts_path=None,
        created_at=now,
        updated_at=now,
        version=1,
    )


def _make_registry_entry(intake_id: str, intake_dir: Path) -> IntakeRegistryEntry:
    now = datetime.now()
    return IntakeRegistryEntry(
        intake_id=intake_id,
        product_name="TestProduct",
        company_name="TestCo",
        created_at=now,
        updated_at=now,
        version=1,
        intake_dir=intake_dir,
        transcript_count=0,
        turn_count=None,
        research_goals=[ResearchGoal.GENERAL_RESEARCH.value],
        status=IntakeStatus.INTAKE_COMPLETE,
    )


def _patch_chroma_count(mocker, metadatas: list[dict]) -> None:
    """Patch chromadb.PersistentClient inside tools.py for chunk counting."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = len(metadatas)
    mock_collection.get.return_value = {"metadatas": metadatas}

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mocker.patch.object(
        tools.chromadb, "PersistentClient", return_value=mock_client
    )


def _stub_intake_dir(tmp_path: Path, intake_id: str) -> Path:
    intake_dir = tmp_path / intake_id
    intake_dir.mkdir(parents=True, exist_ok=True)
    (intake_dir / "intake.json").write_text("{}", encoding="utf-8")
    return intake_dir


# ---------------------------------------------------------------------------
# inspect_state
# ---------------------------------------------------------------------------


def test_inspect_state_happy(mocker, tmp_path):
    intake_id = "intake_happy"
    intake_dir = _stub_intake_dir(tmp_path, intake_id)
    intake = _make_intake(intake_id, intake_dir)
    registry = IntakeRegistry(
        intakes=[_make_registry_entry(intake_id, intake_dir)],
        latest_intake_id=intake_id,
    )

    mocker.patch.object(chat_web, "_resolve_intake_id", return_value=intake_id)
    mocker.patch("icp_agent.intake.load_registry", return_value=registry)
    mocker.patch("icp_agent.intake.load_intake", return_value=intake)

    # Point the ICP_DIR to tmp_path so the existing-files glob is bounded.
    icp_dir = tmp_path / "icp"
    (icp_dir / intake_id).mkdir(parents=True)
    (icp_dir / intake_id / "icp_abc123.json").write_text("{}", encoding="utf-8")
    (icp_dir / intake_id / "icp_partial.json").write_text("{}", encoding="utf-8")
    mocker.patch.object(chat_web, "ICP_DIR", icp_dir)
    mocker.patch.object(chat_web, "PERSIST_DIR", tmp_path / "chroma")

    _patch_chroma_count(
        mocker,
        [
            {"source_type": "transcript"},
            {"source_type": "transcript"},
            {"source_type": "reddit"},
            {"source_type": "deep_research"},
        ],
    )
    # Ensure the persist dir 'exists' so _count_chunks proceeds.
    (tmp_path / "chroma").mkdir()

    result = tools.inspect_state("latest")

    assert result["status"] == "ok"
    assert result["intake_id"] == intake_id
    assert result["product_name"] == "TestProduct"
    assert result["intake_json_exists"] is True
    assert result["chunk_counts"] == {
        "total": 4,
        "transcript": 2,
        "deep_research": 1,
        "reddit": 1,
    }
    # icp_partial.json must be filtered out.
    assert result["existing_icp_files"] == ["icp_abc123.json"]


def test_inspect_state_empty_registry(mocker):
    mocker.patch.object(chat_web, "_resolve_intake_id", return_value=None)

    result = tools.inspect_state("latest")

    assert result["status"] == "error"
    assert "No intakes registered" in result["message"]


def test_inspect_state_intake_not_in_registry(mocker, tmp_path):
    # Resolver pretends to know the id, but the registry has nothing.
    mocker.patch.object(chat_web, "_resolve_intake_id", return_value="intake_ghost")
    mocker.patch(
        "icp_agent.intake.load_registry",
        return_value=IntakeRegistry(intakes=[], latest_intake_id=None),
    )

    result = tools.inspect_state("intake_ghost")

    assert result["status"] == "error"
    assert "intake_ghost" in result["message"]
    assert "Available" in result["message"]


# ---------------------------------------------------------------------------
# build_processing_index
# ---------------------------------------------------------------------------


def test_build_processing_index_happy(mocker):
    fake_module = SimpleNamespace(
        TRANSCRIPTS_JSON=Path("/fake/transcripts.json"),
        CHROMA_DIR=Path("/fake/chroma"),
        build_index_from_disk=MagicMock(
            return_value={"chunks": 42, "persist_dir": "/fake/chroma"}
        ),
    )
    mocker.patch.dict("sys.modules", {"build_index": fake_module})

    result = tools.build_processing_index()

    assert result == {
        "status": "ok",
        "chunks": 42,
        "persist_dir": "/fake/chroma",
    }
    fake_module.build_index_from_disk.assert_called_once()


def test_build_processing_index_missing_transcripts(mocker):
    fake_module = SimpleNamespace(
        TRANSCRIPTS_JSON=Path("/fake/missing.json"),
        CHROMA_DIR=Path("/fake/chroma"),
        build_index_from_disk=MagicMock(
            side_effect=FileNotFoundError("transcripts file not found: /fake/missing.json")
        ),
    )
    mocker.patch.dict("sys.modules", {"build_index": fake_module})

    result = tools.build_processing_index()

    assert result["status"] == "error"
    assert "FileNotFoundError" in result["message"]
    assert "missing.json" in result["message"]


def test_build_processing_index_embedding_failure(mocker):
    fake_module = SimpleNamespace(
        TRANSCRIPTS_JSON=Path("/fake/transcripts.json"),
        CHROMA_DIR=Path("/fake/chroma"),
        build_index_from_disk=MagicMock(
            side_effect=RuntimeError("OPENAI_API_KEY is not set in environment")
        ),
    )
    mocker.patch.dict("sys.modules", {"build_index": fake_module})

    result = tools.build_processing_index()

    assert result["status"] == "error"
    assert "RuntimeError" in result["message"]
    assert "OPENAI_API_KEY" in result["message"]


# ---------------------------------------------------------------------------
# run_synthesis
# ---------------------------------------------------------------------------


def _saved_icp(intake_id: str = "intake_x", icp_id: str = "icpid42") -> ICPDocument:
    return ICPDocument(
        intake_id=intake_id,
        icp_id=icp_id,
        product_name="TestProduct",
        demographics=[],
        jobs_to_be_done=[],
        pains=[],
        gains=[],
        objections=[],
        vocabulary=[],
        watering_holes=[],
        sub_personas=[],
        status="draft",
    )


def _patch_pipeline_happy(mocker, intake, partial_failures: list[str] | None = None):
    """Mock every pipeline.* function used by run_synthesis."""
    from icp_agent import pipeline

    mocker.patch.object(
        pipeline,
        "verify_pipeline_inputs",
        return_value={"intake_id": intake.intake_id, "intake": intake,
                      "chunk_counts": {"total": 1, "transcript": 1,
                                       "deep_research": 0, "reddit": 0},
                      "warnings": [], "verified": True},
    )
    mocker.patch.object(
        pipeline,
        "build_synthesis_queries",
        return_value={s: ["q"] for s in pipeline.SYNTHESIS_SECTIONS},
    )
    mocker.patch.object(
        pipeline,
        "retrieve_evidence_for_synthesis",
        return_value={s: [] for s in pipeline.SYNTHESIS_SECTIONS},
    )

    failing = set(partial_failures or [])

    def fake_synth(section_name, section_chunks, intake, model=None):
        if section_name in failing:
            raise RuntimeError(f"boom in {section_name}")
        return [
            EvidencedClaim(
                claim=f"claim for {section_name}",
                chunk_ids=["c1"],
                source_types=["transcript"],
                confidence="high",
            )
        ]

    mocker.patch.object(pipeline, "synthesize_section", side_effect=fake_synth)
    mocker.patch.object(
        pipeline,
        "synthesize_sub_personas",
        return_value=[
            SubPersona(
                name="P1", description="d", key_traits=["a", "b"],
                motivations=["m"], objections=["o"],
                evidence_chunk_ids=["c1"],
            )
        ],
    )
    mocker.patch.object(pipeline, "_save_partial", return_value=None)
    mocker.patch.object(
        pipeline,
        "assemble_and_save_icp",
        return_value=_saved_icp(intake.intake_id),
    )


def test_run_synthesis_happy(mocker, tmp_path):
    intake = _make_intake("intake_x", tmp_path / "intake_x")
    _patch_pipeline_happy(mocker, intake)

    result = tools.run_synthesis("intake_x")

    assert result["status"] == "ok"
    assert result["sections_completed"] == 7
    assert result["personas"] == 1
    assert result["partial_failures"] == []
    assert "intake_x" in result["icp_path"]


def test_run_synthesis_partial_failure_returns_ok_with_failed_section(mocker, tmp_path):
    intake = _make_intake("intake_x", tmp_path / "intake_x")
    _patch_pipeline_happy(mocker, intake, partial_failures=["pains"])

    result = tools.run_synthesis("intake_x")

    assert result["status"] == "ok"
    assert result["partial_failures"] == ["pains"]
    # 6 sections succeeded; pains was the only failure
    assert result["sections_completed"] == 6


def test_run_synthesis_verify_failure_returns_error(mocker):
    from icp_agent import pipeline

    mocker.patch.object(
        pipeline,
        "verify_pipeline_inputs",
        side_effect=RuntimeError(
            "No transcript chunks found. Transcripts are mandatory."
        ),
    )

    result = tools.run_synthesis("intake_x")

    assert result["status"] == "error"
    assert "RuntimeError" in result["message"]
    assert "transcript" in result["message"].lower()


def test_run_synthesis_assembly_failure_returns_error(mocker, tmp_path):
    intake = _make_intake("intake_x", tmp_path / "intake_x")
    _patch_pipeline_happy(mocker, intake)
    from icp_agent import pipeline

    mocker.patch.object(
        pipeline, "assemble_and_save_icp",
        side_effect=ValueError("bad shape"),
    )

    result = tools.run_synthesis("intake_x")

    assert result["status"] == "error"
    assert "ValueError" in result["message"]


# ---------------------------------------------------------------------------
# launch_chat_server
# ---------------------------------------------------------------------------


def test_launch_chat_server_happy(mocker, tmp_path):
    icp_path = tmp_path / "icp_42.json"
    icp_path.write_text("{}", encoding="utf-8")
    icp_doc = _saved_icp("intake_x")

    mocker.patch.object(chat_web, "_resolve_intake_id", return_value="intake_x")
    mocker.patch.object(chat_web, "_find_latest_icp", return_value=icp_path)
    mocker.patch("icp_agent.models.load_icp", return_value=icp_doc)
    mocker.patch.object(
        chat_web, "_list_personas",
        return_value=[
            SubPersona(
                name="composite", description="d", key_traits=["a", "b"],
                motivations=["m"], objections=["o"],
                evidence_chunk_ids=["c1"],
            )
        ],
    )
    banner = mocker.patch.object(chat_web, "_print_banner")
    browser = mocker.patch.object(chat_web, "launch_browser_async")
    run_mock = mocker.patch.object(chat_web, "run", return_value=None)

    result = tools.launch_chat_server("intake_x", "127.0.0.1", 5001)

    assert result["status"] == "ok"
    banner.assert_called_once()
    browser.assert_called_once()
    run_mock.assert_called_once_with(host="127.0.0.1", port=5001, debug=False)


def test_launch_chat_server_no_intakes(mocker):
    mocker.patch.object(chat_web, "_resolve_intake_id", return_value=None)

    result = tools.launch_chat_server("latest", "127.0.0.1", 5001)

    assert result["status"] == "error"
    assert "No intakes" in result["message"]


def test_launch_chat_server_no_icp(mocker):
    mocker.patch.object(chat_web, "_resolve_intake_id", return_value="intake_x")
    mocker.patch.object(chat_web, "_find_latest_icp", return_value=None)

    result = tools.launch_chat_server("intake_x", "127.0.0.1", 5001)

    assert result["status"] == "error"
    assert "No ICP found" in result["message"]


# ---------------------------------------------------------------------------
# Schema and dispatch sanity
# ---------------------------------------------------------------------------


def test_dispatch_table_matches_tools_schema():
    schema_names = {t["name"] for t in tools.TOOLS}
    assert schema_names == set(tools.DISPATCH.keys())
    # Every schema entry must have an input_schema with type=object.
    for t in tools.TOOLS:
        assert t["input_schema"]["type"] == "object"
