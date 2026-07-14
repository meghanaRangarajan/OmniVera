"""Tests for scripts/load_samples.py and build_index's transcript resolver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from load_samples import load_sample_transcripts  # noqa: E402

SAMPLE_TRANSCRIPT = """In-Depth Interview – Test Person

Age: 25
Generation: Gen Z
Occupation: Tester

1. BACKGROUND

INTERVIEWER: Tell me about your week.
TEST: It was fine, mostly writing tests.

2. PAINS

INTERVIEWER: What frustrates you?
TEST: Flaky fixtures and undocumented formats.
"""


# --- happy path ------------------------------------------------------------


def test_load_sample_transcripts_writes_json(tmp_path: Path) -> None:
    """A well-formed transcript is parsed and written to the output JSON."""
    samples = tmp_path / "transcripts"
    samples.mkdir()
    (samples / "interview_01.md").write_text(SAMPLE_TRANSCRIPT)
    out = tmp_path / "processed" / "transcripts.json"

    result = load_sample_transcripts(samples_dir=samples, output_json=out)

    assert result["transcripts"] == 1
    assert result["turns"] == 4
    assert out.exists()

    payload = json.loads(out.read_text())
    assert len(payload) == 1
    assert payload[0]["interviewee_name"] == "Test Person"
    assert payload[0]["interviewee_metadata"]["generation"] == "Gen Z"


def test_bundled_samples_parse() -> None:
    """The transcripts committed to samples/ must actually parse.

    Guards against shipping sample data that a new user cannot run.
    """
    from icp_agent.transcripts import load_transcripts

    transcripts = load_transcripts(ROOT / "samples" / "transcripts")

    assert len(transcripts) == 3
    for t in transcripts:
        assert t.interviewee_name, "every sample must yield an interviewee name"
        assert t.turns, "every sample must yield speaker turns"
        sections = {turn.section for turn in t.turns if turn.section}
        assert "PAINS" in sections
        assert t.interviewee_metadata.get("generation") == "Gen Z"


# --- edge case -------------------------------------------------------------


def test_output_parent_directory_is_created(tmp_path: Path) -> None:
    """A missing output directory is created rather than raising."""
    samples = tmp_path / "transcripts"
    samples.mkdir()
    (samples / "interview_01.md").write_text(SAMPLE_TRANSCRIPT)
    out = tmp_path / "does" / "not" / "exist" / "transcripts.json"

    result = load_sample_transcripts(samples_dir=samples, output_json=out)

    assert out.exists()
    assert result["transcripts"] == 1


# --- failure cases ---------------------------------------------------------


def test_missing_samples_dir_raises(tmp_path: Path) -> None:
    """A non-existent samples directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="sample transcripts not found"):
        load_sample_transcripts(
            samples_dir=tmp_path / "nope",
            output_json=tmp_path / "out.json",
        )


def test_empty_samples_dir_raises(tmp_path: Path) -> None:
    """A directory with no parseable transcripts raises rather than writing an empty file."""
    samples = tmp_path / "transcripts"
    samples.mkdir()
    out = tmp_path / "out.json"

    with pytest.raises(FileNotFoundError, match="no parseable transcripts"):
        load_sample_transcripts(samples_dir=samples, output_json=out)

    assert not out.exists()
