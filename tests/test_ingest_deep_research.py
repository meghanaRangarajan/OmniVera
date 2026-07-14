"""Tests for scripts/ingest_deep_research.py.

Covers text cleaning, section parsing, chunking logic, and dedup filtering.
All ChromaDB and OpenAI calls are mocked — no real API calls.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make scripts/ importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import ingest_deep_research as script
from icp_agent.rag import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_chunk(tmp_path: Path) -> Chunk:
    pdf = tmp_path / "test.pdf"
    cid = hashlib.sha256(f"test:{('Introduction1')}:MyReport".encode()).hexdigest()[:16]
    return Chunk(
        id=cid,
        text="Some text",
        transcript_id="deep_research_bose_beat",
        interviewee_name="",
        turn_ids=[],
        metadata={
            "source_type": "deep_research",
            "trust_weight": 0.8,
            "section": "Introduction",
            "interviewee_name": "",
            "generation": "",
            "age": "",
            "location": "",
            "occupation": "",
            "device": "",
            "subreddit": "",
            "username": "",
            "report_title": "MyReport",
            "chunk_index": "1",
        },
        char_count=9,
        token_count=3,
    )


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


class TestExtractAndCleanText:
    def test_removes_citeturn_markers(self, tmp_path: Path) -> None:
        with patch("pdfplumber.open") as mock_open:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Great soundciteturn0abc quality."
            mock_open.return_value.__enter__.return_value.pages = [mock_page]
            result = script.extract_and_clean_text(tmp_path / "dummy.pdf")
        assert "citeturn" not in result
        assert "Great sound quality." in result

    def test_normalizes_entity_tags(self, tmp_path: Path) -> None:
        raw = 'entity["org","Apple","Company"] leads the market.'
        with patch("pdfplumber.open") as mock_open:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = raw
            mock_open.return_value.__enter__.return_value.pages = [mock_page]
            result = script.extract_and_clean_text(tmp_path / "dummy.pdf")
        assert "Apple leads the market." in result
        assert "entity[" not in result

    def test_fixes_ligature_corruptions(self, tmp_path: Path) -> None:
        raw = "It is di_icult to a_ord a_ordable devices."
        with patch("pdfplumber.open") as mock_open:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = raw
            mock_open.return_value.__enter__.return_value.pages = [mock_page]
            result = script.extract_and_clean_text(tmp_path / "dummy.pdf")
        assert "difficult" in result
        assert "afford" in result
        assert "affordable" in result

    def test_concatenates_multiple_pages(self, tmp_path: Path) -> None:
        with patch("pdfplumber.open") as mock_open:
            p1 = MagicMock()
            p1.extract_text.return_value = "Page one."
            p2 = MagicMock()
            p2.extract_text.return_value = "Page two."
            mock_open.return_value.__enter__.return_value.pages = [p1, p2]
            result = script.extract_and_clean_text(tmp_path / "dummy.pdf")
        assert "Page one." in result
        assert "Page two." in result

    def test_handles_empty_page(self, tmp_path: Path) -> None:
        with patch("pdfplumber.open") as mock_open:
            p1 = MagicMock()
            p1.extract_text.return_value = None  # pdfplumber returns None for blank pages
            p2 = MagicMock()
            p2.extract_text.return_value = "Real content."
            mock_open.return_value.__enter__.return_value.pages = [p1, p2]
            result = script.extract_and_clean_text(tmp_path / "dummy.pdf")
        assert "Real content." in result


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_splits_on_h2_headers(self) -> None:
        text = "## Intro\nHello world.\n\n## Findings\nData here."
        sections = script.parse_sections(text)
        names = [s[0] for s in sections]
        assert "Intro" in names
        assert "Findings" in names

    def test_preamble_becomes_executive_summary(self) -> None:
        text = "This is preamble.\n\n## Section One\nContent."
        sections = script.parse_sections(text)
        assert sections[0][0] == "Executive Summary"
        assert "preamble" in sections[0][1]

    def test_no_preamble_skips_executive_summary(self) -> None:
        text = "## Only Section\nContent here."
        sections = script.parse_sections(text)
        assert sections[0][0] == "Only Section"
        assert len(sections) == 1

    def test_h3_subsections_collapse_into_parent_h2(self) -> None:
        text = "## Parent\n### Sub A\nText A.\n### Sub B\nText B."
        sections = script.parse_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Parent"
        assert "Sub A" in sections[0][1]
        assert "Sub B" in sections[0][1]

    def test_no_headers_returns_single_executive_summary(self) -> None:
        text = "Just some plain text with no headers at all."
        sections = script.parse_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Executive Summary"

    def test_empty_text_returns_single_section(self) -> None:
        sections = script.parse_sections("")
        assert len(sections) == 1
        assert sections[0][0] == "Executive Summary"


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------


class TestIsTableBlock:
    def test_detects_table_block(self) -> None:
        block = "| Col1 | Col2 |\n| --- | --- |\n| A | B |"
        assert script._is_table_block(block) is True

    def test_rejects_non_table_block(self) -> None:
        block = "This is a paragraph.\nIt has no pipes.\nJust text."
        assert script._is_table_block(block) is False

    def test_mixed_content_under_threshold_is_not_table(self) -> None:
        # Only 1 of 4 lines starts with |
        block = "Some text\n| Col |\nMore text\nAnd more."
        assert script._is_table_block(block) is False

    def test_empty_block_is_not_table(self) -> None:
        assert script._is_table_block("") is False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunkSection:
    def test_happy_path_produces_chunks(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        section_text = "Block one.\n\nBlock two.\n\nBlock three."
        chunks = script.chunk_section("Intro", section_text, "MyReport", pdf_path)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata_fields(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        chunks = script.chunk_section("SomeSection", "Block A.\n\nBlock B.", "MyTitle", pdf_path)
        for c in chunks:
            assert c.metadata["source_type"] == "deep_research"
            assert c.metadata["trust_weight"] == 0.8
            assert c.metadata["section"] == "SomeSection"
            assert c.metadata["report_title"] == "MyTitle"
            assert c.metadata["subreddit"] == ""
            assert c.metadata["username"] == ""
            assert c.transcript_id == "deep_research_bose_beat"
            assert c.interviewee_name == ""
            assert c.turn_ids == []

    def test_chunk_index_is_1_based_string(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        # Force multiple chunks by making section text large enough
        long_block = "word " * 500
        section_text = f"{long_block}\n\n{long_block}"
        chunks = script.chunk_section("Big", section_text, "T", pdf_path)
        indices = [int(c.metadata["chunk_index"]) for c in chunks]
        assert indices[0] == 1
        if len(indices) > 1:
            assert indices == list(range(1, len(indices) + 1))

    def test_table_block_stays_atomic(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        prose = "word " * 200
        table = "\n".join(f"| {i} | val |" for i in range(20))
        section_text = f"{prose}\n\n{table}\n\n{prose}"
        chunks = script.chunk_section("Mix", section_text, "T", pdf_path)
        # Table lines should all be in the same chunk
        table_chunks = [c for c in chunks if "|" in c.text]
        assert len(table_chunks) == 1, "Table was split across multiple chunks"

    def test_oversized_single_block_becomes_own_chunk(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        big_block = "word " * 700  # well above MAX_TOKENS=600
        small_block = "short text"
        section_text = f"{big_block}\n\n{small_block}"
        chunks = script.chunk_section("Oversized", section_text, "T", pdf_path)
        assert any(c.token_count > script.MAX_TOKENS for c in chunks)

    def test_ids_are_stable_across_calls(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        text = "Block A.\n\nBlock B."
        chunks1 = script.chunk_section("Sec", text, "Title", pdf_path)
        chunks2 = script.chunk_section("Sec", text, "Title", pdf_path)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_empty_section_returns_no_chunks(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "report.pdf"
        chunks = script.chunk_section("Empty", "", "T", pdf_path)
        assert chunks == []


# ---------------------------------------------------------------------------
# Duplicate filtering
# ---------------------------------------------------------------------------


class TestFilterNewChunks:
    def test_all_new_when_collection_empty(self, sample_chunk: Chunk) -> None:
        collection = MagicMock()
        collection.get.return_value = {"ids": []}
        new_chunks, already_exist = script.filter_new_chunks([sample_chunk], collection)
        assert len(new_chunks) == 1
        assert already_exist == 0

    def test_duplicate_is_filtered_out(self, sample_chunk: Chunk) -> None:
        collection = MagicMock()
        collection.get.return_value = {"ids": [sample_chunk.id]}
        new_chunks, already_exist = script.filter_new_chunks([sample_chunk], collection)
        assert len(new_chunks) == 0
        assert already_exist == 1

    def test_partial_dedup(self, tmp_path: Path) -> None:
        def make_chunk(stem: str) -> Chunk:
            cid = hashlib.sha256(stem.encode()).hexdigest()[:16]
            return Chunk(
                id=cid, text=stem, transcript_id="t", interviewee_name="",
                turn_ids=[], metadata={}, char_count=len(stem), token_count=1,
            )

        existing_chunk = make_chunk("existing")
        new_chunk = make_chunk("new_one")
        collection = MagicMock()
        collection.get.return_value = {"ids": [existing_chunk.id]}

        result, count = script.filter_new_chunks([existing_chunk, new_chunk], collection)
        assert len(result) == 1
        assert result[0].id == new_chunk.id
        assert count == 1

    def test_empty_input_returns_empty(self) -> None:
        collection = MagicMock()
        new_chunks, already_exist = script.filter_new_chunks([], collection)
        assert new_chunks == []
        assert already_exist == 0
        collection.get.assert_not_called()


# ---------------------------------------------------------------------------
# Title mapping
# ---------------------------------------------------------------------------


class TestReportTitle:
    def test_known_stem_returns_human_title(self, tmp_path: Path) -> None:
        pdf = tmp_path / "deep_research_report.pdf"
        title = script._report_title(pdf)
        assert title == "Gen Z Premium Wireless Earbuds Market and Bose Beat Launch Context"

    def test_unknown_stem_derives_from_filename(self, tmp_path: Path) -> None:
        pdf = tmp_path / "some_other_report.pdf"
        title = script._report_title(pdf)
        assert title == "Some Other Report"
