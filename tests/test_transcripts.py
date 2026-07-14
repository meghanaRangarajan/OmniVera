"""Tests for src/icp_agent/transcripts.py"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from icp_agent.transcripts import (
    Transcript,
    Turn,
    load_transcripts,
    load_transcripts_from_json,
    save_transcripts,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_docx(tmp_path: Path, paragraphs: list[str]) -> Path:
    """Create a .docx file with given paragraphs."""
    import docx

    doc = docx.Document()
    for para in paragraphs:
        doc.add_paragraph(para)
    out = tmp_path / "test.docx"
    doc.save(str(out))
    return out


def make_pdf(tmp_path: Path, text: str) -> Path:
    """Create a single-page PDF containing text."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    out = tmp_path / "test.pdf"
    c = canvas.Canvas(str(out), pagesize=LETTER)
    y = 750
    for line in text.splitlines():
        c.drawString(40, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    return out


def write_txt(tmp_path: Path, content: str, name: str = "test.txt") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1: Happy path .txt — 3 speaker turns, 2 unique speakers
# ---------------------------------------------------------------------------


def test_txt_happy_path(tmp_path):
    content = "ALEX: Hi there\nJORDAN: Hello friend\nALEX: How are you"
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    t = results[0]
    assert len(t.turns) == 3
    assert t.turns[0].speaker == "ALEX"
    assert t.turns[1].speaker == "JORDAN"
    assert t.turns[2].speaker == "ALEX"
    assert set(t.speakers) == {"ALEX", "JORDAN"}


# ---------------------------------------------------------------------------
# Test 2: No speaker prefix → 1 Unknown turn
# ---------------------------------------------------------------------------


def test_txt_no_speaker(tmp_path):
    content = "This is just some unattributed text without any speaker labels."
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    assert len(results[0].turns) == 1
    assert results[0].turns[0].speaker == "Unknown"


# ---------------------------------------------------------------------------
# Test 3: .vtt — 2 cues with timestamps
# ---------------------------------------------------------------------------


def test_vtt_timestamps(tmp_path):
    content = (
        "WEBVTT\n\n"
        "1\n"
        "00:00:05.000 --> 00:00:08.000\n"
        "<v Alice>Hello there.\n\n"
        "2\n"
        "00:00:09.500 --> 00:00:12.000\n"
        "<v Bob>How are you?\n"
    )
    p = tmp_path / "test.vtt"
    p.write_text(content, encoding="utf-8")
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    turns = results[0].turns
    assert len(turns) == 2
    assert turns[0].speaker == "Alice"
    assert turns[0].timestamp_seconds == pytest.approx(5.0)
    assert turns[1].speaker == "Bob"
    assert turns[1].timestamp_seconds == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# Test 4: Multi-interview split — 3 headers → 3 Transcripts
# ---------------------------------------------------------------------------


def test_multi_interview_split(tmp_path):
    content = (
        "In-Depth Qualitative Interview – Alice Smith\n"
        "Age: 25\n"
        "INTERVIEWER: Tell me about yourself.\n"
        "ALICE: I love music.\n\n"
        "In-Depth Qualitative Interview – Bob Jones\n"
        "Age: 32\n"
        "INTERVIEWER: What do you listen to?\n"
        "BOB: Podcasts mostly.\n\n"
        "In-Depth Qualitative Interview – Carol Lee\n"
        "Age: 19\n"
        "INTERVIEWER: Favorite device?\n"
        "CAROL: Earbuds.\n"
    )
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    assert len(results) == 3
    names = [t.interviewee_name for t in results]
    assert "Alice Smith" in names
    assert "Bob Jones" in names
    assert "Carol Lee" in names


# ---------------------------------------------------------------------------
# Test 5: Metadata extraction
# ---------------------------------------------------------------------------


def test_metadata_extraction(tmp_path):
    content = (
        "In-Depth Qualitative Interview – Jordan Park\n"
        "Age: 24\n"
        "Generation: Gen Z\n"
        "Location: Portland, OR\n"
        "Occupation: Student\n"
        "INTERVIEWER: Tell me about yourself.\n"
        "JORDAN: Sure!\n"
    )
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    meta = results[0].interviewee_metadata
    assert meta["age"] == "24"
    assert meta["generation"] == "Gen Z"
    assert meta["location"] == "Portland, OR"
    assert meta["occupation"] == "Student"


# ---------------------------------------------------------------------------
# Test 6: Section detection
# ---------------------------------------------------------------------------


def test_section_detection(tmp_path):
    content = (
        "INTERVIEWER: Let's begin.\n"
        "ALEX: Ready.\n"
        "1. INTRODUCTIONS\n"
        "INTERVIEWER: Tell me about yourself.\n"
        "ALEX: I'm a student.\n"
        "5. BRAND PERCEPTIONS\n"
        "INTERVIEWER: What brands do you like?\n"
        "ALEX: I like Sony.\n"
    )
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    turns = results[0].turns
    # Turns before first heading have no section
    intro_turns = [t for t in turns if t.section == "INTRODUCTIONS"]
    brand_turns = [t for t in turns if t.section == "BRAND_PERCEPTIONS"]
    assert len(intro_turns) >= 1
    assert len(brand_turns) >= 1


# ---------------------------------------------------------------------------
# Test 7: Section normalisation with parenthetical
# ---------------------------------------------------------------------------


def test_section_normalization(tmp_path):
    content = (
        "5. BRAND PERCEPTION (for Current Bose User)\n"
        "INTERVIEWER: What do you think of Bose?\n"
        "ALEX: They're premium.\n"
    )
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    turns = results[0].turns
    section_values = {t.section for t in turns if t.section}
    assert "BRAND_PERCEPTIONS" in section_values


# ---------------------------------------------------------------------------
# Test 8: .docx parsing — 3 speaker paragraphs → 3 turns
# ---------------------------------------------------------------------------


def test_docx_parsing(tmp_path):
    paragraphs = [
        "ALEX: Hello from docx",
        "JORDAN: Hi back",
        "ALEX: Goodbye",
    ]
    make_docx(tmp_path, paragraphs)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    assert len(results[0].turns) == 3
    assert results[0].turns[0].speaker == "ALEX"


# ---------------------------------------------------------------------------
# Test 9: .pdf parsing — short interview parses correctly
# ---------------------------------------------------------------------------


def test_pdf_parsing(tmp_path):
    # Mirrors real PDF format: metadata header, then section heading, then INT/NAME turns.
    text = (
        "In-Depth Qualitative Interview – PDF Person\n"
        "Age: 30\n"
        "1. INTRODUCTIONS\n"
        "INT: Hello.\n"
        "PDF PERSON: Hi there.\n"
    )
    make_pdf(tmp_path, text)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    assert results[0].interviewee_name == "PDF Person"
    assert any(t.speaker == "INT" for t in results[0].turns)


# ---------------------------------------------------------------------------
# Test 10: Near-empty PDF → warning logged, skipped
# ---------------------------------------------------------------------------


def test_pdf_near_empty_skipped(tmp_path, caplog):
    # Create a PDF with virtually no extractable text
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    out = tmp_path / "empty.pdf"
    c = canvas.Canvas(str(out), pagesize=LETTER)
    c.save()  # blank page — pdfplumber returns empty string

    with caplog.at_level(logging.WARNING):
        results = load_transcripts(tmp_path)

    assert results == []
    assert any("skipping" in r.message.lower() or "short" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 11: Unsupported extension → warning logged, skipped
# ---------------------------------------------------------------------------


def test_unsupported_extension_skipped(tmp_path, caplog):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes

    with caplog.at_level(logging.WARNING):
        results = load_transcripts(tmp_path)

    assert results == []
    assert any("unsupported" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 12: Missing folder → FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_folder_raises(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_transcripts(nonexistent)


# ---------------------------------------------------------------------------
# Test 14: WRAP-UP section heading (hyphen in name)
# ---------------------------------------------------------------------------


def test_wrap_up_section(tmp_path):
    content = (
        "5. BRAND PERCEPTIONS\n"
        "INT: What brands do you know?\n"
        "ALEX: Sony and Bose.\n"
        "6. WRAP-UP\n"
        "INT: Any final thoughts?\n"
        "ALEX: Nope, all good.\n"
    )
    write_txt(tmp_path, content)
    results = load_transcripts(tmp_path)
    assert len(results) == 1
    turns = results[0].turns
    wrap_turns = [t for t in turns if t.section == "WRAP_UP"]
    brand_turns = [t for t in turns if t.section == "BRAND_PERCEPTIONS"]
    assert len(wrap_turns) == 2, f"Expected 2 WRAP_UP turns, got {len(wrap_turns)}: {[(t.speaker, t.section) for t in turns]}"
    assert len(brand_turns) == 2
    # Confirm no WRAP-UP turns are mis-tagged as BRAND_PERCEPTIONS
    assert all(t.section == "WRAP_UP" for t in wrap_turns)


# ---------------------------------------------------------------------------
# Test 13: Roundtrip save + load
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path):
    content = "ALEX: Hello\nJORDAN: World"
    write_txt(tmp_path, content, "interview.txt")
    transcripts = load_transcripts(tmp_path)
    assert len(transcripts) == 1

    out_json = tmp_path / "out.json"
    save_transcripts(transcripts, out_json)
    loaded = load_transcripts_from_json(out_json)

    assert len(loaded) == 1
    t_orig = transcripts[0]
    t_loaded = loaded[0]

    assert t_loaded.id == t_orig.id
    assert t_loaded.title == t_orig.title
    assert t_loaded.interviewee_name == t_orig.interviewee_name
    assert t_loaded.source_type == t_orig.source_type
    assert t_loaded.trust_weight == t_orig.trust_weight
    assert len(t_loaded.turns) == len(t_orig.turns)
    for a, b in zip(t_orig.turns, t_loaded.turns):
        assert a.speaker == b.speaker
        assert a.text == b.text
        assert a.section == b.section
