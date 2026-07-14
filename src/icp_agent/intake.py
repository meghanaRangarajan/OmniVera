"""Intake module: collect user inputs, ingest transcript files, and persist to disk.

Entry point for the ICP Agent pipeline. Builds and persists an Intake record
that downstream phases (research plan, retrieval, synthesis) consume.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from icp_agent.transcripts import Transcript, load_transcripts_from_json

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".pdf", ".docx", ".vtt", ".srt"})


class ResearchGoal(str, Enum):
    TEST_AD_CREATIVE = "test_ad_creative"
    TEST_MESSAGING = "test_messaging"
    TEST_PRODUCT_IDEAS = "test_product_ideas"
    GENERAL_RESEARCH = "general_research"
    BUILD_SALES_COLLATERAL = "build_sales_collateral"


class IntakeStatus(str, Enum):
    INTAKE_COMPLETE = "intake_complete"
    RESEARCH_PLAN_PENDING = "research_plan_pending"
    RESEARCH_PLAN_READY = "research_plan_ready"
    RESEARCH_EXECUTING = "research_executing"
    RESEARCH_COMPLETE = "research_complete"
    ICP_SYNTHESIZED = "icp_synthesized"
    CHAT_READY = "chat_ready"


class Intake(BaseModel):
    product_name: str = Field(min_length=2, max_length=100)
    product_description: str = Field(min_length=20, max_length=1000)
    company_name: str | None = Field(default=None, max_length=100)
    icp_hypothesis: str = Field(min_length=15, max_length=500)
    competitors: list[str] = Field(min_length=2, max_length=5)
    research_goals: list[ResearchGoal] = Field(min_length=1)
    target_geography: str | None = Field(default=None, max_length=200)
    intake_id: str
    intake_dir: Path
    transcript_files: list[Path] = Field(default_factory=list)
    parsed_transcripts_path: Path | None = None
    created_at: datetime
    updated_at: datetime
    version: int = 1

    @field_validator("competitors")
    @classmethod
    def strip_and_dedupe(cls, v: list[str]) -> list[str]:
        """Strip whitespace, dedupe case-insensitively, preserve first-occurrence casing."""
        seen_lower: set[str] = set()
        result: list[str] = []
        for item in v:
            stripped = item.strip()
            if not stripped:
                continue
            if stripped.lower() in seen_lower:
                continue
            seen_lower.add(stripped.lower())
            result.append(stripped)
        return result


class IntakeRegistryEntry(BaseModel):
    intake_id: str
    product_name: str
    company_name: str | None = None
    created_at: datetime
    updated_at: datetime
    version: int = 1
    intake_dir: Path
    transcript_count: int = 0
    turn_count: int | None = None
    research_goals: list[str]
    status: IntakeStatus = IntakeStatus.INTAKE_COMPLETE


class IntakeRegistry(BaseModel):
    intakes: list[IntakeRegistryEntry] = Field(default_factory=list)
    latest_intake_id: str | None = None


def ingest_transcripts(source_files: list[Path], dest_dir: Path) -> list[Path]:
    """Copy source files to dest_dir, skipping unsupported extensions with warnings.

    Args:
        source_files: Paths to files to copy.
        dest_dir: Destination directory (created if needed).

    Returns:
        List of destination Paths that were successfully copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in source_files:
        if src.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type: %s", src.name)
            continue
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        logger.info("Copied transcript: %s → %s", src.name, dest)
        copied.append(dest)
    return copied


def load_registry(base_dir: Path = Path("data/inputs")) -> IntakeRegistry:
    """Load registry.json if it exists, else return an empty IntakeRegistry.

    Args:
        base_dir: Directory containing registry.json.

    Returns:
        IntakeRegistry object (empty if file not found).
    """
    registry_path = base_dir / "registry.json"
    if not registry_path.exists():
        return IntakeRegistry()
    return IntakeRegistry.model_validate_json(registry_path.read_text(encoding="utf-8"))


def save_registry(registry: IntakeRegistry, base_dir: Path = Path("data/inputs")) -> None:
    """Write registry to disk atomically using a tmp-then-replace pattern.

    Args:
        registry: IntakeRegistry object to persist.
        base_dir: Directory where registry.json is stored.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = base_dir / "registry.json.tmp"
    final_path = base_dir / "registry.json"
    tmp_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)
    logger.info("Registry saved: %d entries", len(registry.intakes))


def register_intake(
    intake: Intake,
    status: IntakeStatus = IntakeStatus.INTAKE_COMPLETE,
    base_dir: Path = Path("data/inputs"),
) -> None:
    """Add or update an intake entry in the registry, then persist.

    Computes transcript_count from intake.transcript_files and turn_count
    by lazily loading parsed_transcripts_path if available.

    Args:
        intake: The Intake to register.
        status: Status to record for this entry.
        base_dir: Directory containing registry.json.
    """
    transcript_count = len(intake.transcript_files)

    turn_count: int | None = None
    if intake.parsed_transcripts_path is not None and intake.parsed_transcripts_path.exists():
        transcripts = load_transcripts_from_json(intake.parsed_transcripts_path)
        turn_count = sum(len(t.turns) for t in transcripts)

    new_entry = IntakeRegistryEntry(
        intake_id=intake.intake_id,
        product_name=intake.product_name,
        company_name=intake.company_name,
        created_at=intake.created_at,
        updated_at=intake.updated_at,
        version=intake.version,
        intake_dir=intake.intake_dir,
        transcript_count=transcript_count,
        turn_count=turn_count,
        research_goals=[goal.value for goal in intake.research_goals],
        status=status,
    )

    registry = load_registry(base_dir)

    updated = False
    for i, entry in enumerate(registry.intakes):
        if entry.intake_id == intake.intake_id:
            registry.intakes[i] = new_entry
            updated = True
            break
    if not updated:
        registry.intakes.append(new_entry)

    registry.latest_intake_id = intake.intake_id
    save_registry(registry, base_dir)
    logger.info(
        "Registered intake %s (status=%s, transcripts=%d, turns=%s)",
        intake.intake_id,
        status.value,
        transcript_count,
        turn_count,
    )


def save_intake(intake: Intake, base_dir: Path = Path("data/inputs")) -> Path:
    """Save intake.json to intake.intake_dir and update the registry.

    Args:
        intake: Validated Intake object.
        base_dir: Directory containing registry.json. Forwarded to
            register_intake so tests can redirect writes to a tmp path.

    Returns:
        Path to the written intake.json file.
    """
    intake.intake_dir.mkdir(parents=True, exist_ok=True)
    out_path = intake.intake_dir / "intake.json"
    out_path.write_text(intake.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved intake to %s", out_path)
    register_intake(intake, base_dir=base_dir)
    return out_path


def load_intake(intake_dir: Path) -> Intake:
    """Load intake.json from a given intake_dir.

    Args:
        intake_dir: Directory containing intake.json.

    Returns:
        Validated Intake object.

    Raises:
        FileNotFoundError: If intake.json does not exist.
    """
    json_path = intake_dir / "intake.json"
    return Intake.model_validate_json(json_path.read_text(encoding="utf-8"))


def load_latest_intake(base_dir: Path = Path("data/inputs")) -> Intake | None:
    """Return the most recent Intake via registry's latest_intake_id.

    Args:
        base_dir: Directory containing registry.json.

    Returns:
        The latest Intake, or None if registry is empty or latest_intake_id is unset.
    """
    registry = load_registry(base_dir)
    if registry.latest_intake_id is None:
        return None
    for entry in registry.intakes:
        if entry.intake_id == registry.latest_intake_id:
            return load_intake(entry.intake_dir)
    return None


def load_intake_transcripts(
    intake_id: str,
    base_dir: Path = Path("data/inputs"),
) -> list[Transcript]:
    """Load the parsed transcripts for a given intake_id.

    Looks up the intake's directory, reads parsed_transcripts_path
    from intake.json, and returns the parsed Transcript objects.

    Returns an empty list if the intake has no transcripts
    (e.g., the user skipped transcript upload).

    Args:
        intake_id: The intake identifier string.
        base_dir: Directory containing intake subdirectories.

    Returns:
        List of Transcript objects, empty if no transcripts were uploaded.

    Raises:
        FileNotFoundError: If the intake_id directory does not exist.
    """
    intake_dir = base_dir / intake_id
    if not intake_dir.exists():
        raise FileNotFoundError(f"Intake '{intake_id}' not found in {base_dir}")
    intake = load_intake(intake_dir)
    if intake.parsed_transcripts_path is None:
        return []
    if not intake.parsed_transcripts_path.exists():
        logger.warning(
            "parsed_transcripts_path set but file missing for intake %s: %s",
            intake_id,
            intake.parsed_transcripts_path,
        )
        return []
    return load_transcripts_from_json(intake.parsed_transcripts_path)


def load_latest_intake_transcripts(
    base_dir: Path = Path("data/inputs"),
) -> list[Transcript]:
    """Load parsed transcripts for the most recent intake.

    Args:
        base_dir: Directory containing registry.json and intake subdirectories.

    Returns:
        List of Transcript objects, empty if no intakes exist or latest has none.
    """
    registry = load_registry(base_dir)
    if registry.latest_intake_id is None:
        return []
    return load_intake_transcripts(registry.latest_intake_id, base_dir=base_dir)
