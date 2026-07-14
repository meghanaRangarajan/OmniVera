"""Pydantic models for synthesized ICP output and on-disk persistence.

Produced by the Step 4 synthesis stage of the pipeline and consumed by the
Step 6 chat assistant. Claims carry explicit chunk-id citations so any
downstream surface (review UI, chat persona) can trace them back to raw
evidence.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EvidencedClaim(BaseModel):
    """A single synthesized claim plus the chunks that support it."""

    claim: str
    chunk_ids: list[str]
    source_types: list[str]
    confidence: Literal["high", "medium", "low"]

    @field_validator("chunk_ids")
    @classmethod
    def _non_empty_chunk_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("chunk_ids must contain at least one id")
        return v


class SubPersona(BaseModel):
    """A distinct customer archetype identified across the evidence."""

    name: str
    description: str
    key_traits: list[str]
    motivations: list[str]
    objections: list[str]
    evidence_chunk_ids: list[str]

    @field_validator("key_traits")
    @classmethod
    def _non_empty_key_traits(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("key_traits must contain at least one trait")
        return v

    @field_validator("evidence_chunk_ids")
    @classmethod
    def _non_empty_evidence(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("evidence_chunk_ids must contain at least one id")
        return v


def _default_icp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ICPDocument(BaseModel):
    """Top-level synthesized ICP record, one per synthesis run."""

    intake_id: str
    product_name: str
    icp_id: str = Field(default_factory=_default_icp_id)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    demographics: list[EvidencedClaim] = Field(default_factory=list)
    jobs_to_be_done: list[EvidencedClaim] = Field(default_factory=list)
    pains: list[EvidencedClaim] = Field(default_factory=list)
    gains: list[EvidencedClaim] = Field(default_factory=list)
    objections: list[EvidencedClaim] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    watering_holes: list[EvidencedClaim] = Field(default_factory=list)
    sub_personas: list[SubPersona] = Field(default_factory=list)
    manual_edits: dict[str, Any] = Field(default_factory=dict)
    status: Literal["draft", "reviewed", "locked"] = "draft"


def save_icp(icp: ICPDocument, base_dir: Path = Path("data/icp")) -> Path:
    """Persist an ICPDocument to data/icp/{intake_id}/icp_{icp_id}.json.

    Args:
        icp: Validated ICPDocument to write.
        base_dir: Directory containing per-intake subdirectories.

    Returns:
        Absolute path to the written file.
    """
    out_dir = base_dir / icp.intake_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"icp_{icp.icp_id}.json"
    out_path.write_text(icp.model_dump_json(indent=2), encoding="utf-8")
    return out_path


def load_icp(icp_path: Path) -> ICPDocument:
    """Load and validate an ICPDocument from JSON on disk.

    Args:
        icp_path: Path to the icp_*.json file produced by save_icp.

    Returns:
        Validated ICPDocument.

    Raises:
        FileNotFoundError: If icp_path does not exist.
    """
    if not icp_path.exists():
        raise FileNotFoundError(f"ICP file not found: {icp_path}")
    return ICPDocument.model_validate_json(icp_path.read_text(encoding="utf-8"))
