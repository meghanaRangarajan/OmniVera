"""Render samples/deep_research/deep_research_report.md into a PDF.

The deep-research ingestion path takes PDFs, not markdown, so the sample report
has to ship as a PDF. This script regenerates it from the markdown source, which
stays in the repo as the readable, diffable original.

The one thing that matters: `##` section headers must survive PDF text
extraction as literal `## Name` lines, because that is what
scripts/ingest_deep_research.py splits on.

Usage:
    python scripts/build_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate

ROOT = Path(__file__).parent.parent
SOURCE_MD = ROOT / "samples" / "deep_research" / "deep_research_report.md"
OUTPUT_PDF = ROOT / "samples" / "deep_research" / "deep_research_report.pdf"

TITLE = "Gen Z Outdoor Smartwatch Market and Garmin Roam Launch Context"


def build_pdf(source_md: Path = SOURCE_MD, output_pdf: Path = OUTPUT_PDF) -> Path:
    """Render the markdown source to a PDF, preserving ## headers as literal text.

    Args:
        source_md: Markdown source file.
        output_pdf: Destination PDF path.

    Returns:
        The path to the written PDF.

    Raises:
        FileNotFoundError: If source_md does not exist.
    """
    if not source_md.exists():
        raise FileNotFoundError(f"markdown source not found: {source_md}")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=8
    )
    heading = ParagraphStyle(
        "heading",
        parent=styles["Heading2"],
        fontSize=12.5,
        leading=16,
        spaceBefore=16,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=LETTER,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title=TITLE,
    )

    flow: list[Paragraph] = []
    for raw in source_md.read_text().split("\n"):
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("## "):
            # Keep the literal "## " prefix — the ingester splits on it.
            flow.append(Paragraph(f"## {line[3:]}", heading))
        else:
            flow.append(Paragraph(line.replace("*", ""), body))

    doc.build(flow)
    return output_pdf


def main() -> None:
    try:
        out = build_pdf()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
