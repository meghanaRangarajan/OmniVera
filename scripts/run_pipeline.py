"""Single entry point for the ICP synthesis pipeline.

Usage:
    python scripts/run_pipeline.py --intake-id <intake_id>
    python scripts/run_pipeline.py --intake-id latest

Currently runs only Step 1 (verification). Future steps are appended in main().
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icp_agent.models import EvidencedClaim  # noqa: E402
from icp_agent.pipeline import (  # noqa: E402
    _save_partial,
    assemble_and_save_icp,
    build_synthesis_queries,
    retrieve_evidence_for_synthesis,
    setup_logging,
    synthesize_section,
    synthesize_sub_personas,
    verify_pipeline_inputs,
)

load_dotenv(ROOT / ".env")

PERSIST_DIR = ROOT / "data" / "processed" / "chroma"
LOG_DIR = ROOT / "data" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ICP synthesis pipeline for a given intake.",
    )
    parser.add_argument(
        "--intake-id",
        required=True,
        help="Registered intake_id, or 'latest' to use the most recent intake.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = setup_logging(LOG_DIR)
    log = logging.getLogger("icp_agent.run_pipeline")
    log.debug("Pipeline invoked with intake_id=%s", args.intake_id)

    try:
        verification = verify_pipeline_inputs(args.intake_id, PERSIST_DIR)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.debug("Verification failed:\n%s", traceback.format_exc())
        print(f"\n✗ Pipeline aborted: {exc}")
        print(f"  Full traceback logged to: {log_path}")
        return 1
    except Exception as exc:
        log.debug("Unexpected failure:\n%s", traceback.format_exc())
        print(f"\n✗ Unexpected error: {exc}")
        print(f"  Full traceback logged to: {log_path}")
        return 1

    try:
        queries = build_synthesis_queries(verification["intake"])
    except ValueError as exc:
        log.debug("Query decomposition failed:\n%s", traceback.format_exc())
        print(f"\n✗ Pipeline aborted: {exc}")
        print(f"  Full traceback logged to: {log_path}")
        return 1
    except Exception as exc:
        log.debug("Unexpected failure during decomposition:\n%s", traceback.format_exc())
        print(f"\n✗ Unexpected error: {exc}")
        print(f"  Full traceback logged to: {log_path}")
        return 1

    log.debug("Pipeline Step 2 complete; %d queries ready for retrieval", sum(len(v) for v in queries.values()))

    try:
        evidence = retrieve_evidence_for_synthesis(queries, PERSIST_DIR)
    except Exception as exc:
        log.debug("Evidence retrieval failed:\n%s", traceback.format_exc())
        print(f"\n✗ Pipeline aborted: {exc}")
        print(f"  Full traceback logged to: {log_path}")
        return 1

    log.debug(
        "Pipeline Step 3 complete; %d chunks collected across %d sections",
        sum(len(v) for v in evidence.values()),
        len(evidence),
    )

    # Step 4a — per-section synthesis, error-isolated per section
    intake = verification["intake"]
    SECTIONS = [
        "demographics",
        "jobs_to_be_done",
        "pains",
        "gains",
        "objections",
        "vocabulary",
        "watering_holes",
    ]
    completed_sections: dict[str, list[EvidencedClaim]] = {}
    for section in SECTIONS:
        try:
            claims = synthesize_section(section, evidence.get(section, []), intake)
        except Exception as exc:
            log.error("Section %s synthesis crashed: %s", section, exc)
            log.debug("Traceback:\n%s", traceback.format_exc())
            claims = []
        completed_sections[section] = claims
        _save_partial(intake.intake_id, completed_sections)

    # Step 4b — cross-section sub-persona identification
    try:
        sub_personas = synthesize_sub_personas(evidence, completed_sections, intake)
    except Exception as exc:
        log.error("Sub-persona synthesis crashed: %s", exc)
        log.debug("Traceback:\n%s", traceback.format_exc())
        sub_personas = []

    log.debug(
        "Pipeline Step 4 complete; %d total claims across %d sections, %d personas",
        sum(len(c) for c in completed_sections.values()),
        len(completed_sections),
        len(sub_personas),
    )

    # Step 5 — assemble, re-grade, save, cleanup, register
    try:
        assemble_and_save_icp(
            intake=intake,
            completed_sections=completed_sections,
            sub_personas=sub_personas,
            all_evidence=evidence,
        )
    except Exception as exc:
        log.error("ICP assembly failed: %s", exc)
        log.debug("Traceback:\n%s", traceback.format_exc())
        print(f"\n✗ Pipeline aborted during assembly: {exc}")
        print(f"  Full traceback logged to: {log_path}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
