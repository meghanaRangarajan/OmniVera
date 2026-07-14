"""Tests for src/icp_agent/chat.py"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from icp_agent import chat
from icp_agent.models import EvidencedClaim, ICPDocument, SubPersona


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_persona() -> SubPersona:
    return SubPersona(
        name="The Budget Pragmatist",
        description="Price-sensitive students who value durability.",
        key_traits=["budget-conscious"],
        motivations=["low price"],
        objections=["too expensive"],
        evidence_chunk_ids=["c1"],
    )


def _make_icp() -> ICPDocument:
    claim = EvidencedClaim(
        claim="Earbuds fall out during runs.",
        chunk_ids=["c1"],
        source_types=["transcript"],
        confidence="high",
    )
    return ICPDocument(
        intake_id="intake_x",
        product_name="Bose Beat",
        pains=[claim, claim, claim],
        gains=[claim, claim, claim],
        vocabulary=["vibe", "lowkey", "fire"],
        sub_personas=[_make_persona()],
    )


def _mock_message_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _patch_anthropic(mocker, create_returns: list[str] | None = None) -> MagicMock:
    """Patch chat.anthropic.Anthropic() with a MagicMock client."""
    mock_client = MagicMock()
    if create_returns is not None:
        mock_client.messages.create.side_effect = [
            _mock_message_response(t) for t in create_returns
        ]
    mocker.patch.object(chat.anthropic, "Anthropic", return_value=mock_client)
    return mock_client


def _make_chunk_mock(chunk_id: str, source_type: str = "transcript") -> MagicMock:
    trust = {"transcript": 1.0, "deep_research": 0.8, "reddit": 0.6}.get(source_type, 0.5)
    chunk = MagicMock()
    chunk.id = chunk_id
    chunk.text = f"Text for {chunk_id}"
    chunk.interviewee_name = "" if source_type == "reddit" else "Alice"
    chunk.metadata = {
        "source_type": source_type,
        "trust_weight": trust,
        "section": "",
        "username": "u_reddit" if source_type == "reddit" else "",
    }
    return chunk


# ---------------------------------------------------------------------------
# process_input
# ---------------------------------------------------------------------------


def test_process_input_text_passthrough():
    assert chat.process_input(text="hello") == "hello"
    assert chat.process_input(text="  hello  ") == "hello"


def test_process_input_empty_text_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        chat.process_input(text="   ")


def test_process_input_unsupported_image_raises():
    with pytest.raises(ValueError, match="Unsupported image format"):
        chat.process_input(image_path=Path("file.gif"))


def test_process_input_image_calls_sonnet(mocker, tmp_path):
    # Minimal valid PNG (8-byte signature + IHDR + IEND)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path = tmp_path / "ad.png"
    img_path.write_bytes(png_bytes)

    mock_client = _patch_anthropic(mocker, ["Dark tone, minimalist layout."])

    result = chat.process_input(image_path=img_path)

    assert result == "Dark tone, minimalist layout."
    assert mock_client.messages.create.call_count == 1
    kwargs = mock_client.messages.create.call_args.kwargs
    image_block = kwargs["messages"][0]["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"


def test_process_input_pdf_extracts_text(mocker, tmp_path):
    pdf_path = tmp_path / "note.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = (
        "Customer said it falls out during runs. "
        "We need better fit for athletic users. More notes here."
    )
    fake_pdf_ctx = MagicMock()
    fake_pdf_ctx.pages = [fake_page]
    fake_pdf_ctx.__enter__.return_value = fake_pdf_ctx
    fake_pdf_ctx.__exit__.return_value = False

    mocker.patch("pdfplumber.open", return_value=fake_pdf_ctx)

    result = chat.process_input(file_path=pdf_path)

    assert "Customer said it falls out during runs." in result


def test_process_input_scanned_pdf_raises(mocker, tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 placeholder")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = ""
    fake_pdf_ctx = MagicMock()
    fake_pdf_ctx.pages = [fake_page]
    fake_pdf_ctx.__enter__.return_value = fake_pdf_ctx
    fake_pdf_ctx.__exit__.return_value = False

    mocker.patch("pdfplumber.open", return_value=fake_pdf_ctx)

    with pytest.raises(ValueError, match="scanned or empty"):
        chat.process_input(file_path=pdf_path)


def test_process_input_multiple_inputs_raises():
    with pytest.raises(ValueError, match="exactly one"):
        chat.process_input(text="hi", image_path=Path("x.jpg"))


# ---------------------------------------------------------------------------
# build_chat_system_prompt
# ---------------------------------------------------------------------------


def test_build_system_prompt_contains_persona_name():
    icp = _make_icp()
    prompt = chat.build_chat_system_prompt(icp, _make_persona())
    assert "The Budget Pragmatist" in prompt


def test_build_system_prompt_contains_grounding_rules():
    icp = _make_icp()
    prompt = chat.build_chat_system_prompt(icp, _make_persona())
    assert "I don't have strong evidence" in prompt


def test_build_system_prompt_contains_vocabulary():
    icp = _make_icp()
    icp.vocabulary = ["vibe", "lowkey"]
    prompt = chat.build_chat_system_prompt(icp, _make_persona())
    assert "vibe" in prompt
    assert "lowkey" in prompt


# ---------------------------------------------------------------------------
# decompose_and_retrieve
# ---------------------------------------------------------------------------


def test_decompose_and_retrieve_deduplicates(mocker, tmp_path):
    _patch_anthropic(mocker, [json.dumps(["q1", "q2"])])

    shared = _make_chunk_mock("chunk_shared")

    def fake_search(query, *args, **kwargs):
        return {
            "primary": [(shared, 0.80)],
            "secondary": [],
            "corroboration": [],
        }

    mocker.patch.object(chat, "search_with_trust_priority", side_effect=fake_search)

    block = chat.decompose_and_retrieve("some question", _make_persona(), tmp_path)

    # Chunk header appears exactly once (chunk_id also appears in text body)
    assert block.count("[CHUNK chunk_shared") == 1
    assert "matched 2 queries" in block


def test_decompose_and_retrieve_fallback_on_bad_json(mocker, tmp_path, caplog):
    _patch_anthropic(mocker, ["not json", "still not json"])

    search_mock = mocker.patch.object(
        chat,
        "search_with_trust_priority",
        return_value={"primary": [], "secondary": [], "corroboration": []},
    )

    with caplog.at_level(logging.WARNING, logger="icp_agent.chat"):
        chat.decompose_and_retrieve("fit during exercise?", _make_persona(), tmp_path)

    # search called once with the raw input as single query
    assert search_mock.call_count == 1
    assert search_mock.call_args.args[0] == "fit during exercise?"
    assert any(
        "Query decomposition failed" in rec.message for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# stream_response / _summarize_history
# ---------------------------------------------------------------------------


def _make_stream_client(mocker, texts: list[str]) -> MagicMock:
    mock_client = MagicMock()
    stream_obj = MagicMock()
    stream_obj.text_stream = iter(texts)
    stream_obj.__enter__.return_value = stream_obj
    stream_obj.__exit__.return_value = False
    mock_client.messages.stream.return_value = stream_obj
    mocker.patch.object(chat.anthropic, "Anthropic", return_value=mock_client)
    return mock_client


def test_stream_response_appends_newline(mocker, capsys):
    _make_stream_client(mocker, ["Hello", " world"])

    history: list[dict[str, str]] = []
    result = chat.stream_response("hi", "evidence block", history, "system")

    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert "Hello" in out
    assert " world" in out
    assert result == "Hello world"


def test_summarize_history_compresses_old_messages(mocker):
    _patch_anthropic(mocker, ["Earlier the user discussed earbud fit and sound."])

    history = []
    for i in range(8):
        history.append({"role": "user", "content": f"u{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    assert len(history) == 16

    compressed = chat._summarize_history(history)

    assert len(compressed) == 2 + 6  # summary pair + tail
    assert compressed[0]["role"] == "user"
    assert "Earlier conversation summary" in compressed[0]["content"]
    assert compressed[1]["role"] == "assistant"
    assert compressed[-6:] == history[-6:]


def test_stream_response_triggers_summarization_at_14(mocker):
    _make_stream_client(mocker, ["ok"])
    summarize_mock = mocker.patch.object(
        chat,
        "_summarize_history",
        return_value=[
            {"role": "user", "content": "[summary]"},
            {"role": "assistant", "content": "Understood."},
        ],
    )

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(14)
    ]

    chat.stream_response("next", "ev", history, "sys")

    assert summarize_mock.call_count == 1


def test_stream_response_no_summarization_below_14(mocker):
    _make_stream_client(mocker, ["ok"])
    summarize_mock = mocker.patch.object(chat, "_summarize_history")

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(13)
    ]

    chat.stream_response("next", "ev", history, "sys")

    assert summarize_mock.call_count == 0
