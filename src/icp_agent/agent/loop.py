"""Streaming tool-use loop for the orchestrator agent.

Drives ``claude-sonnet-4-5`` (pinned to the same constant the synthesis
pipeline uses) through the four orchestrator tools defined in
``icp_agent.agent.tools``.

Design notes:

- Model prose is streamed to **stdout** in real time so the user sees
  status updates as they arrive.
- Tool inputs and outputs go to the agent log at DEBUG only — they are
  not echoed to the user.
- Pipeline modules' own ``print`` output is suppressed inside
  ``run_synthesis`` (see ``tools._flush_captured``); only the model's
  narration reaches the user-visible stdout stream.
- The loop terminates when the model returns ``stop_reason != "tool_use"``
  (it stopped calling tools), or implicitly when ``launch_chat_server``
  succeeds and ``chat_web.run`` blocks the process forever.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from icp_agent.agent.prompts import SYSTEM_PROMPT
from icp_agent.agent.tools import DISPATCH, TOOLS
from icp_agent.pipeline import SYNTHESIS_MODEL

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_TURN = 2048
MAX_TURNS = 20  # Safety cap so a misbehaving model can't loop forever.


def run_agent(intake_id_arg: str, host: str, port: int) -> int:
    """Drive the orchestrator agent end-to-end.

    Args:
        intake_id_arg: The CLI ``--intake-id`` value, possibly ``"latest"``.
        host: Bind host for the chat web server.
        port: Bind port for the chat web server.

    Returns:
        Process exit code. 0 on clean termination (chat server stopped or
        the model finished without errors). 1 if the safety cap on turns
        was hit.
    """
    client = anthropic.Anthropic()

    initial_user = (
        f"Run the pipeline for intake_id={intake_id_arg!r}. "
        f"When you reach launch_chat_server, use host={host!r} port={port}."
    )
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_user}
    ]

    for turn in range(1, MAX_TURNS + 1):
        logger.debug("=== agent turn %d ===", turn)
        final_message = _stream_one_turn(client, messages)

        # Append the assistant turn to history. Pass the SDK's content
        # blocks through as-is — model_dump() would include SDK-internal
        # fields (e.g. text.parsed_output) that the API rejects on input.
        messages.append(
            {"role": "assistant", "content": final_message.content}
        )

        if final_message.stop_reason != "tool_use":
            logger.debug(
                "Agent finished with stop_reason=%s", final_message.stop_reason
            )
            return 0

        tool_results = _execute_tool_uses(final_message.content)
        messages.append({"role": "user", "content": tool_results})

    logger.error(
        "Agent loop hit MAX_TURNS=%d without terminating; aborting.",
        MAX_TURNS,
    )
    print(
        f"\n✗ Agent exceeded the {MAX_TURNS}-turn safety cap without "
        "completing the pipeline. See the agent log for details."
    )
    return 1


def _stream_one_turn(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
) -> Any:
    """Stream one assistant turn to stdout and return the final message.

    Text content is written to stdout token-by-token so the user sees the
    model talking in real time. The stream is fully consumed before
    returning so subsequent tool-use blocks can be inspected.
    """
    with client.messages.stream(
        model=SYNTHESIS_MODEL,
        max_tokens=MAX_TOKENS_PER_TURN,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        for event in stream:
            etype = getattr(event, "type", None)
            if etype == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta is not None and getattr(delta, "type", None) == "text_delta":
                    text = getattr(delta, "text", "")
                    if text:
                        print(text, end="", flush=True)
        final = stream.get_final_message()

    print()  # newline after the streamed prose
    return final


def _execute_tool_uses(
    assistant_content: list[Any],
) -> list[dict[str, Any]]:
    """Execute every tool_use block in an assistant turn, return tool_result blocks."""
    results: list[dict[str, Any]] = []
    for block in assistant_content:
        if getattr(block, "type", None) != "tool_use":
            continue
        name = block.name
        tool_input = dict(block.input or {})
        logger.debug("tool_use name=%s input=%s", name, tool_input)

        fn = DISPATCH.get(name)
        if fn is None:
            result: dict[str, Any] = {
                "status": "error",
                "message": f"Unknown tool '{name}'.",
            }
        else:
            try:
                result = fn(**tool_input)
            except TypeError as exc:
                # Bad arguments from the model — surface as a tool_result so
                # the model can correct itself rather than crashing the loop.
                result = {
                    "status": "error",
                    "message": f"Bad arguments to {name}: {exc}",
                }
            except Exception as exc:  # noqa: BLE001 — see module docstring
                logger.exception("tool %s raised", name)
                result = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }

        logger.debug("tool_result name=%s result=%s", name, result)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            }
        )
    return results
