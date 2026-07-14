"""Tests for src/icp_agent/agent/loop.py.

The Anthropic streaming context manager is fully mocked. No network.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from icp_agent.agent import loop


# ---------------------------------------------------------------------------
# Stream-mocking helpers
# ---------------------------------------------------------------------------


def _text_delta(text: str) -> SimpleNamespace:
    """Approximate an Anthropic content_block_delta event for text output."""
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(
    name: str,
    tool_input: dict,
    block_id: str = "tu_1",
) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use",
        name=name,
        input=tool_input,
        id=block_id,
    )


def _final_message(content: list, stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class _FakeStream:
    """Mimics the Anthropic streaming context manager."""

    def __init__(self, events: list, final):
        self._events = events
        self._final = final

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_anthropic_with_turns(mocker, turns: list[tuple[list, SimpleNamespace]]) -> MagicMock:
    """Patch anthropic.Anthropic so each call to messages.stream returns the next turn."""
    streams = [_FakeStream(events, final) for events, final in turns]
    fake_messages = MagicMock()
    fake_messages.stream.side_effect = streams
    fake_client = MagicMock()
    fake_client.messages = fake_messages
    mocker.patch.object(loop.anthropic, "Anthropic", return_value=fake_client)
    return fake_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loop_terminates_when_model_stops_calling_tools(mocker, capsys):
    """If the very first turn ends with stop_reason=end_turn, the loop exits 0."""
    turns = [
        (
            [_text_delta("All done.")],
            _final_message([_text_block("All done.")], stop_reason="end_turn"),
        ),
    ]
    _patch_anthropic_with_turns(mocker, turns)

    rc = loop.run_agent("latest", "127.0.0.1", 5001)

    assert rc == 0
    out = capsys.readouterr().out
    assert "All done." in out


def test_loop_executes_tool_then_continues(mocker, capsys):
    """Two turns: turn 1 calls inspect_state; turn 2 stops."""
    inspect_call = _tool_use_block(
        "inspect_state", {"intake_id": "latest"}, block_id="tu_inspect"
    )
    turns = [
        (
            [_text_delta("Inspecting…")],
            _final_message(
                [_text_block("Inspecting…"), inspect_call],
                stop_reason="tool_use",
            ),
        ),
        (
            [_text_delta("Got the state, finishing.")],
            _final_message(
                [_text_block("Got the state, finishing.")],
                stop_reason="end_turn",
            ),
        ),
    ]
    _patch_anthropic_with_turns(mocker, turns)

    inspect_mock = mocker.patch.object(
        loop, "DISPATCH",
        {"inspect_state": MagicMock(return_value={"status": "ok", "intake_id": "iX"})},
    )

    rc = loop.run_agent("latest", "127.0.0.1", 5001)

    assert rc == 0
    inspect_mock["inspect_state"].assert_called_once_with(intake_id="latest")
    out = capsys.readouterr().out
    assert "Inspecting…" in out
    assert "Got the state" in out


def test_loop_handoff_via_launch_chat_server(mocker, capsys):
    """When the model calls launch_chat_server, the dispatch fires and the loop
    keeps going only as far as the model wants to go after the tool returns.

    In production the tool blocks forever and the loop never re-enters; here we
    mock the tool to return a non-error and have the model immediately stop on
    the next turn — verifying the dispatch was called with the right args.
    """
    launch_call = _tool_use_block(
        "launch_chat_server",
        {"intake_id": "intake_x", "host": "127.0.0.1", "port": 5001},
        block_id="tu_launch",
    )
    turns = [
        (
            [_text_delta("Launching server…")],
            _final_message(
                [_text_block("Launching server…"), launch_call],
                stop_reason="tool_use",
            ),
        ),
        (
            [],
            _final_message(
                [_text_block("(handoff complete)")],
                stop_reason="end_turn",
            ),
        ),
    ]
    _patch_anthropic_with_turns(mocker, turns)

    handoff_called = {}

    def fake_launch(intake_id, host, port):
        handoff_called["args"] = (intake_id, host, port)
        return {"status": "ok", "message": "(server stub)"}

    mocker.patch.object(
        loop, "DISPATCH",
        {"launch_chat_server": fake_launch},
    )

    rc = loop.run_agent("intake_x", "127.0.0.1", 5001)

    assert rc == 0
    assert handoff_called["args"] == ("intake_x", "127.0.0.1", 5001)
    out = capsys.readouterr().out
    assert "Launching server" in out


def test_loop_surfaces_tool_exception_as_tool_result(mocker):
    """A raising tool must NOT crash the loop — it becomes a structured error
    tool_result so the model can narrate it."""
    bad_call = _tool_use_block("inspect_state", {"intake_id": "latest"})
    turns = [
        (
            [],
            _final_message([bad_call], stop_reason="tool_use"),
        ),
        (
            [_text_delta("Tool failed; stopping.")],
            _final_message(
                [_text_block("Tool failed; stopping.")],
                stop_reason="end_turn",
            ),
        ),
    ]
    _patch_anthropic_with_turns(mocker, turns)

    raising_tool = MagicMock(side_effect=RuntimeError("kaboom"))
    mocker.patch.object(loop, "DISPATCH", {"inspect_state": raising_tool})

    rc = loop.run_agent("latest", "127.0.0.1", 5001)

    assert rc == 0
    raising_tool.assert_called_once()


def test_loop_safety_cap_halts_runaway(mocker, capsys):
    """If the model keeps calling tools forever, the loop must break at MAX_TURNS."""
    inspect_call = _tool_use_block("inspect_state", {"intake_id": "latest"})

    def make_turn():
        return (
            [],
            _final_message(
                [_text_block("looping"), inspect_call],
                stop_reason="tool_use",
            ),
        )

    # Always call a tool, never end_turn — past MAX_TURNS the loop must abort.
    turns = [make_turn() for _ in range(loop.MAX_TURNS + 2)]
    _patch_anthropic_with_turns(mocker, turns)

    mocker.patch.object(
        loop, "DISPATCH",
        {"inspect_state": MagicMock(return_value={"status": "ok"})},
    )

    rc = loop.run_agent("latest", "127.0.0.1", 5001)

    assert rc == 1
    out = capsys.readouterr().out
    assert "safety cap" in out.lower()
