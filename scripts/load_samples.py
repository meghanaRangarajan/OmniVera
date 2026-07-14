"""Parse the bundled sample transcripts into data/processed/transcripts.json.

A convenience entry point for first-time users. It skips the intake form and
turns the synthetic interviews in samples/transcripts/ into the combined
parsed-transcripts file that scripts/build_index.py reads.

Usage:
    python scripts/load_samples.py

No API keys required — this step is pure parsing, no embeddings, no LLM calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icp_agent.transcripts import load_transcripts, save_transcripts

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SAMPLES_DIR = ROOT / "samples" / "transcripts"
OUTPUT_JSON = ROOT / "data" / "processed" / "transcripts.json"


def load_sample_transcripts(
    samples_dir: Path = SAMPLES_DIR,
    output_json: Path = OUTPUT_JSON,
) -> dict:
    """Parse the sample transcripts and write the combined JSON file.

    Args:
        samples_dir: Directory holding the sample transcript files.
        output_json: Destination for the combined parsed-transcripts JSON.

    Returns:
        {"transcripts": int, "turns": int, "output": str}

    Raises:
        FileNotFoundError: If samples_dir does not exist or holds no transcripts.
    """
    if not samples_dir.exists():
        raise FileNotFoundError(f"sample transcripts not found: {samples_dir}")

    transcripts = load_transcripts(samples_dir)
    if not transcripts:
        raise FileNotFoundError(f"no parseable transcripts in {samples_dir}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    save_transcripts(transcripts, output_json)

    total_turns = sum(len(t.turns) for t in transcripts)
    return {
        "transcripts": len(transcripts),
        "turns": total_turns,
        "output": str(output_json),
    }


def main() -> None:
    try:
        result = load_sample_transcripts()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nParsed {result['transcripts']} sample transcript(s), "
          f"{result['turns']} turns total.")
    print(f"Wrote {result['output']}")
    print("\nNext: python scripts/build_index.py\n")


if __name__ == "__main__":
    main()
