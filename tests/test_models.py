"""Tests for src/icp_agent/models.py"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from icp_agent.models import (
    EvidencedClaim,
    ICPDocument,
    SubPersona,
    load_icp,
    save_icp,
)


def _make_claim(**overrides) -> EvidencedClaim:
    defaults = dict(
        claim="Customers dislike earbuds that fall out during exercise.",
        chunk_ids=["c1", "c2"],
        source_types=["transcript", "reddit"],
        confidence="high",
    )
    defaults.update(overrides)
    return EvidencedClaim(**defaults)


def _make_persona(**overrides) -> SubPersona:
    defaults = dict(
        name="The Budget-Conscious Student",
        description="College student balancing good earbuds with a tight budget.",
        key_traits=["budget-conscious", "gym-oriented"],
        motivations=["value for money", "workout-friendly fit"],
        objections=["skeptical of premium brands"],
        evidence_chunk_ids=["c1", "c2"],
    )
    defaults.update(overrides)
    return SubPersona(**defaults)


def test_icp_roundtrip(tmp_path):
    icp = ICPDocument(
        intake_id="intake_test",
        product_name="Garmin Roam",
        pains=[_make_claim()],
        sub_personas=[_make_persona()],
        vocabulary=["these slap", "fit is tight"],
    )

    saved_path = save_icp(icp, base_dir=tmp_path)
    loaded = load_icp(saved_path)

    assert loaded.icp_id == icp.icp_id
    assert loaded.intake_id == icp.intake_id
    assert loaded.product_name == icp.product_name
    assert loaded.pains == icp.pains
    assert loaded.sub_personas == icp.sub_personas
    assert loaded.vocabulary == icp.vocabulary
    assert loaded.status == icp.status
    assert loaded.version == icp.version


def test_evidenced_claim_empty_chunk_ids_raises():
    with pytest.raises(ValidationError):
        EvidencedClaim(
            claim="Some claim",
            chunk_ids=[],
            source_types=["transcript"],
            confidence="high",
        )


def test_sub_persona_empty_key_traits_raises():
    with pytest.raises(ValidationError):
        SubPersona(
            name="X",
            description="Y",
            key_traits=[],
            motivations=["m1"],
            objections=["o1"],
            evidence_chunk_ids=["c1"],
        )


def test_sub_persona_empty_evidence_chunk_ids_raises():
    with pytest.raises(ValidationError):
        SubPersona(
            name="X",
            description="Y",
            key_traits=["t1"],
            motivations=["m1"],
            objections=["o1"],
            evidence_chunk_ids=[],
        )


def test_icp_document_defaults():
    icp = ICPDocument(intake_id="intake_test", product_name="Garmin Roam")

    assert icp.demographics == []
    assert icp.jobs_to_be_done == []
    assert icp.pains == []
    assert icp.gains == []
    assert icp.objections == []
    assert icp.vocabulary == []
    assert icp.watering_holes == []
    assert icp.sub_personas == []
    assert icp.manual_edits == {}
    assert icp.status == "draft"
    assert icp.version == 1
    # Format: YYYYMMDD_HHMMSS (8 digits, underscore, 6 digits)
    assert len(icp.icp_id) == 15
    assert icp.icp_id[8] == "_"
    assert icp.icp_id.replace("_", "").isdigit()


def test_save_icp_creates_correct_path(tmp_path):
    icp = ICPDocument(intake_id="intake_test", product_name="Garmin Roam")

    saved_path = save_icp(icp, base_dir=tmp_path)

    assert saved_path == tmp_path / "intake_test" / f"icp_{icp.icp_id}.json"
    assert saved_path.exists()
