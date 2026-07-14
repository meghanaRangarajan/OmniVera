"""Tests for scripts/build_sample_pdf.py.

The contract these lock down: the sample deep-research PDF must survive text
extraction with its `## ` section headers intact, because that is what
scripts/ingest_deep_research.py splits on. A PDF renderer that quietly drops or
restyles those markers would break the deep-research ingestion path with no
error — the ingester would simply find one section instead of nine.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_sample_pdf import build_pdf  # noqa: E402

HEADER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

SAMPLE_MD = """## About This Report

A synthetic report used for testing.

## Findings

Battery life ranks first, and it is not close.
"""


def _extract(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


# --- happy path ------------------------------------------------------------


def test_headers_survive_pdf_extraction(tmp_path: Path) -> None:
    """`## ` headers are still literal `## ` lines after a render/extract round trip."""
    src = tmp_path / "report.md"
    src.write_text(SAMPLE_MD)
    out = tmp_path / "report.pdf"

    build_pdf(source_md=src, output_pdf=out)

    sections = HEADER_RE.findall(_extract(out))
    assert sections == ["About This Report", "Findings"]


def test_body_text_survives(tmp_path: Path) -> None:
    """Body content is preserved, not just the headers."""
    src = tmp_path / "report.md"
    src.write_text(SAMPLE_MD)
    out = tmp_path / "report.pdf"

    build_pdf(source_md=src, output_pdf=out)

    assert "Battery life ranks first" in _extract(out)


# --- the committed sample --------------------------------------------------


def test_bundled_sample_pdf_parses() -> None:
    """The PDF committed to samples/ must yield the sections a new user expects.

    Guards against shipping a sample the deep-research ingester cannot read.
    """
    pdf = ROOT / "samples" / "deep_research" / "deep_research_report.pdf"
    assert pdf.exists(), "sample deep-research PDF is missing from the repo"

    sections = HEADER_RE.findall(_extract(pdf))

    assert len(sections) >= 8
    assert "Executive Summary" in sections
    assert "Brand Perception Among Young Buyers" in sections
    assert len(sections) == len(set(sections)), "section names must be unique"


def test_bundled_sample_pdf_matches_markdown_source() -> None:
    """The committed PDF is in sync with its markdown source.

    Fails if someone edits the .md and forgets to re-run build_sample_pdf.py.
    """
    md = ROOT / "samples" / "deep_research" / "deep_research_report.md"
    pdf = ROOT / "samples" / "deep_research" / "deep_research_report.pdf"

    md_sections = HEADER_RE.findall(md.read_text())
    pdf_sections = HEADER_RE.findall(_extract(pdf))

    assert md_sections == pdf_sections, (
        "PDF is stale — re-run `python scripts/build_sample_pdf.py`"
    )


# --- failure case ----------------------------------------------------------


def test_missing_source_raises(tmp_path: Path) -> None:
    """A missing markdown source raises FileNotFoundError rather than writing an empty PDF."""
    out = tmp_path / "report.pdf"

    with pytest.raises(FileNotFoundError, match="markdown source not found"):
        build_pdf(source_md=tmp_path / "nope.md", output_pdf=out)

    assert not out.exists()
