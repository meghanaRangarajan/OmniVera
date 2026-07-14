"""Tests for src/icp_agent/intake.py"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from icp_agent.intake import (
    Intake,
    IntakeRegistry,
    IntakeStatus,
    ResearchGoal,
    ingest_transcripts,
    load_intake,
    load_intake_transcripts,
    load_latest_intake,
    load_latest_intake_transcripts,
    load_registry,
    register_intake,
    save_intake,
)
from icp_agent.transcripts import Transcript, Turn, save_transcripts


def _make_intake(tmp_path: Path, **overrides) -> Intake:
    """Build a minimal valid Intake for testing."""
    now = datetime.now()
    defaults: dict = dict(
        product_name="TestProduct",
        product_description="A product that does something really meaningful for users.",
        company_name="TestCorp",
        icp_hypothesis="Our ideal customer is a mid-market SaaS buyer aged 30-45.",
        competitors=["Competitor A", "Competitor B"],
        research_goals=[ResearchGoal.GENERAL_RESEARCH],
        target_geography="United States",
        intake_id="intake_20240101_120000",
        intake_dir=tmp_path / "intake_20240101_120000",
        transcript_files=[],
        parsed_transcripts_path=None,
        created_at=now,
        updated_at=now,
        version=1,
    )
    defaults.update(overrides)
    return Intake(**defaults)


def test_intake_happy_path(tmp_path):
    intake = _make_intake(tmp_path)
    assert intake.product_name == "TestProduct"
    assert intake.competitors == ["Competitor A", "Competitor B"]
    assert intake.research_goals == [ResearchGoal.GENERAL_RESEARCH]
    assert intake.version == 1
    assert intake.company_name == "TestCorp"


def test_intake_short_description_raises(tmp_path):
    with pytest.raises(ValidationError):
        _make_intake(tmp_path, product_description="Too short")


def test_intake_too_few_competitors_raises(tmp_path):
    with pytest.raises(ValidationError):
        _make_intake(tmp_path, competitors=["OnlyOne"])


def test_intake_too_many_competitors_raises(tmp_path):
    with pytest.raises(ValidationError):
        _make_intake(tmp_path, competitors=["A", "B", "C", "D", "E", "F"])


def test_competitor_dedup(tmp_path):
    intake = _make_intake(tmp_path, competitors=["Notion", "notion", "NOTION"])
    assert intake.competitors == ["Notion"]


def test_ingest_transcripts_filters_unsupported(tmp_path):
    src_dir = tmp_path / "sources"
    src_dir.mkdir()
    dest_dir = tmp_path / "dest"

    (src_dir / "interview.txt").write_text("ALEX: Hello", encoding="utf-8")
    (src_dir / "notes.md").write_text("# Notes", encoding="utf-8")
    (src_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (src_dir / "data.csv").write_text("a,b,c", encoding="utf-8")

    source_files = list(src_dir.iterdir())
    copied = ingest_transcripts(source_files, dest_dir)

    assert len(copied) == 2
    assert all(p.parent == dest_dir for p in copied)
    copied_names = {p.name for p in copied}
    assert "interview.txt" in copied_names
    assert "notes.md" in copied_names
    assert "image.png" not in copied_names
    assert "data.csv" not in copied_names
    assert dest_dir.exists()


def test_save_load_roundtrip(tmp_path):
    intake_dir = tmp_path / "intake_test"
    intake_dir.mkdir()
    intake = _make_intake(tmp_path, intake_dir=intake_dir)

    saved_path = save_intake(intake, base_dir=tmp_path)
    assert saved_path == intake_dir / "intake.json"
    assert saved_path.exists()

    loaded = load_intake(intake_dir)
    assert loaded.intake_id == intake.intake_id
    assert loaded.product_name == intake.product_name
    assert loaded.product_description == intake.product_description
    assert loaded.competitors == intake.competitors
    assert loaded.research_goals == intake.research_goals
    assert loaded.version == intake.version
    assert loaded.intake_dir == intake.intake_dir


def test_load_latest_intake_empty(tmp_path):
    result = load_latest_intake(base_dir=tmp_path / "nonexistent")
    assert result is None


def test_register_intake_add_and_update(tmp_path):
    intake_dir = tmp_path / "intake_reg"
    intake_dir.mkdir()
    intake = _make_intake(tmp_path, intake_dir=intake_dir)

    register_intake(intake, base_dir=tmp_path)
    registry = load_registry(base_dir=tmp_path)
    assert len(registry.intakes) == 1
    assert registry.latest_intake_id == intake.intake_id
    assert registry.intakes[0].product_name == intake.product_name

    intake2 = _make_intake(
        tmp_path,
        intake_id=intake.intake_id,
        intake_dir=intake_dir,
        product_name="UpdatedProduct",
        product_description="A product that does something really meaningful for users.",
    )
    register_intake(intake2, base_dir=tmp_path)
    registry2 = load_registry(base_dir=tmp_path)
    assert len(registry2.intakes) == 1
    assert registry2.intakes[0].product_name == "UpdatedProduct"


def test_load_registry_missing_file(tmp_path):
    registry = load_registry(base_dir=tmp_path / "no_such_dir")
    assert isinstance(registry, IntakeRegistry)
    assert registry.intakes == []
    assert registry.latest_intake_id is None


# ---------------------------------------------------------------------------
# Transcript loader helpers
# ---------------------------------------------------------------------------


def _make_transcript(source_path: Path) -> Transcript:
    return Transcript(
        id="t1",
        source_path=source_path,
        title="Test Interview",
        interviewee_name="Alice",
        interviewee_metadata={},
        speakers=["INT", "Alice"],
        turns=[Turn(speaker="Alice", text="I use this product daily.")],
        raw_text="Alice: I use this product daily.",
    )


def test_load_intake_transcripts_happy_path(tmp_path):
    intake_id = "intake_t1"
    intake_dir = tmp_path / intake_id
    intake_dir.mkdir()

    parsed_path = intake_dir / "parsed_transcripts.json"
    transcript = _make_transcript(intake_dir / "interview.txt")
    save_transcripts([transcript], parsed_path)

    intake = _make_intake(
        tmp_path,
        intake_id=intake_id,
        intake_dir=intake_dir,
        parsed_transcripts_path=parsed_path,
    )
    save_intake(intake, base_dir=tmp_path)

    result = load_intake_transcripts(intake_id, base_dir=tmp_path)
    assert len(result) == 1
    assert result[0].id == "t1"
    assert result[0].turns[0].speaker == "Alice"


def test_load_intake_transcripts_no_transcripts(tmp_path):
    intake_id = "intake_t2"
    intake_dir = tmp_path / intake_id
    intake_dir.mkdir()

    intake = _make_intake(
        tmp_path,
        intake_id=intake_id,
        intake_dir=intake_dir,
        parsed_transcripts_path=None,
    )
    save_intake(intake, base_dir=tmp_path)

    result = load_intake_transcripts(intake_id, base_dir=tmp_path)
    assert result == []


def test_load_intake_transcripts_missing_file_warns(tmp_path, caplog):
    import logging

    intake_id = "intake_t3"
    intake_dir = tmp_path / intake_id
    intake_dir.mkdir()

    ghost_path = intake_dir / "parsed_transcripts.json"  # intentionally not created

    intake = _make_intake(
        tmp_path,
        intake_id=intake_id,
        intake_dir=intake_dir,
        parsed_transcripts_path=ghost_path,
    )
    save_intake(intake, base_dir=tmp_path)

    with caplog.at_level(logging.WARNING, logger="icp_agent.intake"):
        result = load_intake_transcripts(intake_id, base_dir=tmp_path)

    assert result == []
    assert intake_id in caplog.text


def test_load_intake_transcripts_unknown_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no_such_intake"):
        load_intake_transcripts("no_such_intake", base_dir=tmp_path)


def test_load_latest_intake_transcripts_empty_registry(tmp_path):
    result = load_latest_intake_transcripts(base_dir=tmp_path / "empty")
    assert result == []


def test_load_latest_intake_transcripts_returns_latest(tmp_path):
    intake_id = "intake_latest"
    intake_dir = tmp_path / intake_id
    intake_dir.mkdir()

    parsed_path = intake_dir / "parsed_transcripts.json"
    transcript = _make_transcript(intake_dir / "interview.txt")
    save_transcripts([transcript], parsed_path)

    intake = _make_intake(
        tmp_path,
        intake_id=intake_id,
        intake_dir=intake_dir,
        parsed_transcripts_path=parsed_path,
    )
    # Write intake.json and register to tmp_path so the registry lookup works
    (intake_dir / "intake.json").write_text(intake.model_dump_json(indent=2), encoding="utf-8")
    register_intake(intake, base_dir=tmp_path)

    result = load_latest_intake_transcripts(base_dir=tmp_path)
    assert len(result) == 1
    assert result[0].id == "t1"
