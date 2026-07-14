"""Parse qualitative interview transcripts from multiple file formats into structured objects."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section name normalisation
# ---------------------------------------------------------------------------

SECTION_NORMALIZATION: dict[str, str] = {
    "INTRODUCTION": "INTRODUCTIONS",
    "INTRODUCTIONS": "INTRODUCTIONS",
    "AUDIO IN DAILY LIFE": "AUDIO_IN_DAILY_LIFE",
    "SOUND IN DAILY LIFE": "AUDIO_IN_DAILY_LIFE",
    "HEADPHONES & EARBUDS USE": "HEADPHONES_AND_EARBUDS_USE",
    "PURCHASE DECISIONS & VALUE": "PURCHASE_DECISIONS_AND_VALUE",
    "BRAND PERCEPTION": "BRAND_PERCEPTIONS",
    "BRAND PERCEPTIONS": "BRAND_PERCEPTIONS",
    "WRAP UP": "WRAP_UP",
    "WRAP-UP": "WRAP_UP",
}

# Metadata keys that get normalised to a canonical name
_METADATA_KEY_MAP: dict[str, str] = {
    "age": "age",
    "generation": "generation",
    "location": "location",
    "occupation": "occupation",
    "device": "device",
    "primary device": "device",
    "family": "family",
    "interviewer": "interviewer",
}

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^In-Depth (?:Qualitative )?Interview(?: Transcript)?\s*[–—-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_RE = re.compile(
    r"^\s*\d+\.\s+([A-Z][A-Z\s&-]+?)(?:\s*\([^)]*\))?\s*$",
    re.MULTILINE,
)

SPEAKER_RE = re.compile(
    r"^([A-Za-z][A-Za-z']*(?:\s+[A-Za-z][A-Za-z']*){0,3}):\s+(.+)",
)

METADATA_RE = re.compile(r"^([A-Za-z ]+?):\s*(.+)$")

VTT_TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->")
SRT_TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->")
VTT_VOICE_RE = re.compile(r"<v\s+([^>]+)>(.+)")

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".vtt", ".srt", ".docx", ".pdf"}

# Keys that should never be treated as speaker names in the pre-section header block.
# "interviewer" is included because the header has "Interviewer: [INT]"; within interview
# content the interviewer speaks as "INT", not the full word.
_METADATA_KEY_NAMES: frozenset[str] = frozenset({
    "age", "generation", "location", "occupation",
    "device", "primary device", "family", "interviewer",
})

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single speaker utterance within a transcript."""

    speaker: str
    text: str
    timestamp_seconds: float | None = None
    section: str | None = None


@dataclass
class Transcript:
    """One interview session parsed from a source file."""

    id: str
    source_path: Path
    title: str
    interviewee_name: str | None
    interviewee_metadata: dict[str, str]
    speakers: list[str]
    turns: list[Turn]
    raw_text: str
    source_type: str = "transcript"
    trust_weight: float = 1.0
    loaded_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_transcripts(folder_path: Path) -> list[Transcript]:
    """Load all supported transcript files from a folder into Transcript objects.

    Multi-interview files (PDF/DOCX/TXT/MD) are automatically split into one
    Transcript per interview when 2+ In-Depth Interview headers are detected.

    Args:
        folder_path: Directory containing transcript files.

    Returns:
        List of Transcript objects, one per detected interview.

    Raises:
        FileNotFoundError: If folder_path does not exist.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Transcript folder not found: {folder_path}")

    results: list[Transcript] = []

    for path in sorted(folder_path.iterdir()):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        size = path.stat().st_size
        logger.info("Loading %s (%d bytes)", path.name, size)

        if ext not in _SUPPORTED_EXTENSIONS:
            logger.warning("Unsupported file extension, skipping: %s", path.name)
            continue

        if ext == ".vtt":
            turns = _parse_vtt(path)
            results.append(_build_transcript(path, path.stem, None, {}, turns, path.read_text(encoding="utf-8", errors="replace")))
        elif ext == ".srt":
            turns = _parse_srt(path)
            results.append(_build_transcript(path, path.stem, None, {}, turns, path.read_text(encoding="utf-8", errors="replace")))
        else:
            raw = _extract_raw_text(path, ext)
            if raw is None:
                continue
            segments = _maybe_split(raw, path)
            for name, metadata, segment_text in segments:
                turns = _parse_text_turns(segment_text)
                turns = _tag_sections(turns, segment_text)
                title = name or path.stem
                results.append(_build_transcript(path, title, name, metadata, turns, segment_text))

    return results


def save_transcripts(transcripts: list[Transcript], output_path: Path) -> None:
    """Save transcripts to JSON. Handles Path and datetime serialisation.

    Args:
        transcripts: List of Transcript objects to serialise.
        output_path: Destination .json file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [dataclasses.asdict(t) for t in transcripts]
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def load_transcripts_from_json(json_path: Path) -> list[Transcript]:
    """Deserialise transcripts previously saved by save_transcripts.

    Args:
        json_path: Path to the JSON file written by save_transcripts.

    Returns:
        List of Transcript objects. Roundtrip-safe with save_transcripts.
    """
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [_transcript_from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Internal helpers — text extraction
# ---------------------------------------------------------------------------


def _extract_raw_text(path: Path, ext: str) -> str | None:
    """Extract raw text from .txt, .md, .docx, or .pdf.  Returns None to skip."""
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".docx":
        return _extract_docx(path)

    if ext == ".pdf":
        raw = _extract_pdf(path)
        if len(raw.strip()) < 50:
            logger.warning("PDF text too short (possible OCR issue), skipping: %s", path.name)
            return None
        return raw

    return None


def _extract_docx(path: Path) -> str:
    import docx  # python-docx imports as 'docx'

    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_pdf(path: Path) -> str:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Internal helpers — splitting
# ---------------------------------------------------------------------------


def _maybe_split(raw: str, path: Path) -> list[tuple[str | None, dict[str, str], str]]:
    """Detect and split multi-interview documents.

    Returns a list of (interviewee_name, metadata, segment_text) tuples.
    Single-interview files return a one-element list with name=None.
    """
    matches = list(HEADER_RE.finditer(raw))

    if len(matches) < 2:
        # Single interview — extract metadata from full text if there's 1 header
        if matches:
            name = matches[0].group(1).strip()
            lines = raw.splitlines()
            metadata = _extract_metadata(lines)
        else:
            name = None
            metadata = {}
        return [(name, metadata, raw)]

    logger.info("Split %s into %d transcripts", path.name, len(matches))

    segments: list[tuple[str | None, dict[str, str], str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        segment_text = raw[start:end]
        name = match.group(1).strip()
        lines = segment_text.splitlines()
        metadata = _extract_metadata(lines)
        segments.append((name, metadata, segment_text))

    return segments


def _extract_metadata(lines: list[str]) -> dict[str, str]:
    """Extract key-value metadata from the first ~15 lines of an interview header block."""
    metadata: dict[str, str] = {}
    for line in lines[:15]:
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_RE.match(line):
            break
        # Check METADATA_RE first — "Age: 24" and "INTERVIEWER: ..." are both key-value pairs.
        # Only fall through to SPEAKER_RE break when the line is NOT a key-value pair.
        m = METADATA_RE.match(stripped)
        if m:
            # PDFs often pack multiple fields on one line: "Age: 19 | Generation: Gen Z"
            # Split on " | " and treat each segment as a separate key-value pair.
            segments = stripped.split(" | ")
            for seg in segments:
                sm = METADATA_RE.match(seg.strip())
                if sm:
                    raw_key = sm.group(1).strip().lower()
                    value = sm.group(2).strip()
                    canonical_key = _METADATA_KEY_MAP.get(raw_key, raw_key)
                    metadata[canonical_key] = value
        elif SPEAKER_RE.match(stripped):
            break
    return metadata


# ---------------------------------------------------------------------------
# Internal helpers — turn parsing
# ---------------------------------------------------------------------------


def _parse_text_turns(text: str) -> list[Turn]:
    """Parse speaker-prefixed turns from plain text. Returns [Turn(Unknown)] if none found."""
    turns: list[Turn] = []
    current_speaker: str | None = None
    current_lines: list[str] = []
    # True until the first section heading is seen. Metadata-key names are only filtered
    # in the header block — after a section heading they can appear as real speakers.
    in_header = True

    def flush() -> None:
        if current_speaker is not None:
            turns.append(Turn(speaker=current_speaker, text=" ".join(current_lines).strip()))

    for line in text.splitlines():
        if not line.strip():
            continue
        if SECTION_RE.match(line):
            in_header = False
            continue

        m = SPEAKER_RE.match(line)
        if m:
            potential = m.group(1)
            # Always skip lowercase-first names — PDF line-wrap artifacts.
            if potential[0].islower():
                continue
            # Before the first section heading, skip known metadata key names
            # (Age, Location, Interviewer, etc.) which share "Name: value" syntax.
            if in_header and potential.lower() in _METADATA_KEY_NAMES:
                continue
            flush()
            current_speaker = potential
            current_lines = [m.group(2).strip()]
        else:
            if current_speaker is not None:
                current_lines.append(line.strip())

    flush()

    if not turns:
        turns = [Turn(speaker="Unknown", text=text.strip())]

    return turns


def _tag_sections(turns: list[Turn], raw_text: str) -> list[Turn]:
    """Assign section names to turns based on section headings in raw_text."""
    # Build list of (line_index, section_name) from raw_text
    section_events: list[tuple[int, str]] = []
    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        m = SECTION_RE.match(line)
        if m:
            raw_name = m.group(1).strip()
            section_events.append((i, _normalize_section(raw_name)))

    if not section_events:
        return turns

    # Map each turn to a section by matching speaker text position in lines
    # Strategy: scan lines in order; track current section; when a speaker line
    # matches a turn's speaker+text, assign current section.
    current_section: str | None = None
    event_idx = 0
    assigned: list[str | None] = []

    turn_idx = 0
    for line_no, line in enumerate(lines):
        # Advance section pointer
        while event_idx < len(section_events) and section_events[event_idx][0] <= line_no:
            current_section = section_events[event_idx][1]
            event_idx += 1

        if turn_idx >= len(turns):
            break

        m = SPEAKER_RE.match(line)
        if m and m.group(1) == turns[turn_idx].speaker:
            assigned.append(current_section)
            turn_idx += 1

    # Fill remaining turns with the last known section
    while len(assigned) < len(turns):
        assigned.append(current_section)

    for turn, sec in zip(turns, assigned):
        turn.section = sec

    return turns


def _normalize_section(raw: str) -> str:
    """Normalise a section heading to its canonical form."""
    stripped = raw.strip().upper()
    if stripped in SECTION_NORMALIZATION:
        return SECTION_NORMALIZATION[stripped]
    # Fallback
    return stripped.replace(" & ", "_AND_").replace("&", "_AND_").replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Internal helpers — VTT / SRT parsing
# ---------------------------------------------------------------------------


def _parse_vtt(path: Path) -> list[Turn]:
    """Parse WebVTT file into turns with timestamps."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\n+", text)
    turns: list[Turn] = []

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        ts_seconds: float | None = None
        payload_lines: list[str] = []
        speaker = "Unknown"

        for line in lines:
            ts_m = VTT_TIMESTAMP_RE.search(line)
            if ts_m:
                h, m, s, ms = int(ts_m.group(1)), int(ts_m.group(2)), int(ts_m.group(3)), int(ts_m.group(4))
                ts_seconds = h * 3600 + m * 60 + s + ms / 1000.0
                continue
            if line.startswith("WEBVTT") or line.isdigit():
                continue
            payload_lines.append(line)

        if not payload_lines or ts_seconds is None:
            continue

        payload = " ".join(payload_lines)
        voice_m = VTT_VOICE_RE.match(payload)
        if voice_m:
            speaker = voice_m.group(1).strip()
            payload = voice_m.group(2).strip()
        else:
            sp_m = SPEAKER_RE.match(payload)
            if sp_m:
                speaker = sp_m.group(1)
                payload = sp_m.group(2)

        turns.append(Turn(speaker=speaker, text=payload, timestamp_seconds=ts_seconds))

    return turns or [Turn(speaker="Unknown", text=text.strip())]


def _parse_srt(path: Path) -> list[Turn]:
    """Parse SubRip (.srt) file into turns with timestamps."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\n+", text)
    turns: list[Turn] = []

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        ts_seconds: float | None = None
        payload_lines: list[str] = []
        speaker = "Unknown"

        for line in lines:
            if line.isdigit():
                continue
            ts_m = SRT_TIMESTAMP_RE.search(line)
            if ts_m:
                h, m, s, ms = int(ts_m.group(1)), int(ts_m.group(2)), int(ts_m.group(3)), int(ts_m.group(4))
                ts_seconds = h * 3600 + m * 60 + s + ms / 1000.0
                continue
            payload_lines.append(line)

        if not payload_lines:
            continue

        payload = " ".join(payload_lines)
        sp_m = SPEAKER_RE.match(payload)
        if sp_m:
            speaker = sp_m.group(1)
            payload = sp_m.group(2)

        turns.append(Turn(speaker=speaker, text=payload, timestamp_seconds=ts_seconds))

    return turns or [Turn(speaker="Unknown", text=text.strip())]


# ---------------------------------------------------------------------------
# Internal helpers — object construction & serialisation
# ---------------------------------------------------------------------------


def _build_transcript(
    path: Path,
    title: str,
    interviewee_name: str | None,
    metadata: dict[str, str],
    turns: list[Turn],
    raw_text: str,
) -> Transcript:
    tid = _make_id(path, raw_text, interviewee_name)
    speakers = list(dict.fromkeys(t.speaker for t in turns))
    logger.info(
        "Transcript '%s': %d turns, sections=%s",
        title,
        len(turns),
        sorted({t.section for t in turns if t.section}),
    )
    return Transcript(
        id=tid,
        source_path=path,
        title=title,
        interviewee_name=interviewee_name,
        interviewee_metadata=metadata,
        speakers=speakers,
        turns=turns,
        raw_text=raw_text,
    )


def _make_id(path: Path, raw_text: str, interviewee_name: str | None) -> str:
    payload = str(path) + raw_text + (interviewee_name or "")
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def _transcript_from_dict(d: dict) -> Transcript:
    turns = [Turn(**t) for t in d["turns"]]
    return Transcript(
        id=d["id"],
        source_path=Path(d["source_path"]),
        title=d["title"],
        interviewee_name=d.get("interviewee_name"),
        interviewee_metadata=d.get("interviewee_metadata", {}),
        speakers=d["speakers"],
        turns=turns,
        raw_text=d["raw_text"],
        source_type=d.get("source_type", "transcript"),
        trust_weight=d.get("trust_weight", 1.0),
        loaded_at=datetime.fromisoformat(d["loaded_at"]),
    )
