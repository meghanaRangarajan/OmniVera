"""CLI chat entry point.

Usage:
    python scripts/run_chat.py --intake-id <intake_id>
    python scripts/run_chat.py --intake-id latest

Assumes the synthesis pipeline has already produced an ICPDocument for the
given intake. Prompts the user to pick either the composite Full ICP persona
or one of the synthesized sub-personas, then enters a command-driven chat
loop with per-turn RAG and streamed Sonnet output.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icp_agent.chat import (  # noqa: E402
    build_chat_system_prompt,
    decompose_and_retrieve,
    process_input,
    stream_response,
)
from icp_agent.intake import load_registry  # noqa: E402
from icp_agent.models import ICPDocument, SubPersona, load_icp  # noqa: E402
from icp_agent.pipeline import setup_logging  # noqa: E402

load_dotenv(ROOT / ".env")

PERSIST_DIR = ROOT / "data" / "processed" / "chroma"
ICP_DIR = ROOT / "data" / "icp"
CHAT_DIR = ROOT / "data" / "chat"
LOG_DIR = ROOT / "data" / "logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the synthesized ICP for a given intake.",
    )
    parser.add_argument(
        "--intake-id",
        required=True,
        help="Registered intake_id, or 'latest' to use the most recent intake.",
    )
    return parser.parse_args()


def _resolve_intake_id(intake_id: str) -> str | None:
    if intake_id != "latest":
        return intake_id
    registry = load_registry()
    return registry.latest_intake_id


def _find_latest_icp(intake_id: str) -> Path | None:
    intake_dir = ICP_DIR / intake_id
    if not intake_dir.exists():
        return None
    icp_files = sorted(intake_dir.glob("icp_*.json"))
    icp_files = [p for p in icp_files if p.name != "icp_partial.json"]
    return icp_files[-1] if icp_files else None


def _build_full_icp_persona(icp_doc: ICPDocument) -> SubPersona:
    """Synthetic composite persona built from the ICPDocument itself."""
    return SubPersona(
        name=f"Full ICP — {icp_doc.product_name}",
        description="The composite customer profile across all archetypes",
        key_traits=[c.claim for c in icp_doc.pains[:3]] or ["composite view"],
        motivations=[c.claim for c in icp_doc.gains[:3]] or ["composite view"],
        objections=[c.claim for c in icp_doc.objections[:3]] or ["composite view"],
        # evidence_chunk_ids must be non-empty per validator; use a union sample
        evidence_chunk_ids=_union_chunk_ids(icp_doc)[:10] or ["composite"],
    )


def _union_chunk_ids(icp_doc: ICPDocument) -> list[str]:
    ids: list[str] = []
    for section in (
        icp_doc.pains,
        icp_doc.gains,
        icp_doc.objections,
        icp_doc.demographics,
        icp_doc.jobs_to_be_done,
        icp_doc.watering_holes,
    ):
        for claim in section:
            ids.extend(claim.chunk_ids)
    # Preserve order, drop duplicates
    seen: set[str] = set()
    uniq: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


def _render_persona_menu(icp_doc: ICPDocument) -> None:
    print("══════════════════════════════════════════")
    print(f"ICP Chat — {icp_doc.product_name}")
    print("══════════════════════════════════════════")
    print("Select a persona to chat with:")
    print()
    print("  [1] Full ICP (composite — all personas)")
    for i, persona in enumerate(icp_doc.sub_personas, start=2):
        desc = persona.description[:80]
        ellipsis = "..." if len(persona.description) > 80 else ""
        print(f"  [{i}] {persona.name}")
        print(f"      {desc}{ellipsis}")
    print()


def _select_persona(icp_doc: ICPDocument) -> SubPersona:
    _render_persona_menu(icp_doc)
    max_choice = 1 + len(icp_doc.sub_personas)
    while True:
        raw = input("Enter number (or press Enter for Full ICP): ").strip()
        if raw == "":
            return _build_full_icp_persona(icp_doc)
        if raw.isdigit():
            choice = int(raw)
            if choice == 1:
                return _build_full_icp_persona(icp_doc)
            if 2 <= choice <= max_choice:
                return icp_doc.sub_personas[choice - 2]
        print(f"Please enter a number between 1 and {max_choice}, or press Enter.")


def _print_chat_banner(active_persona: SubPersona) -> None:
    print("──────────────────────────────────────────")
    print(f"Chatting as: {active_persona.name}")
    print("Type your message and press Enter.")
    print("Commands:")
    print("  /persona  — switch persona mid-conversation")
    print("  /history  — show message count and token estimate")
    print("  /save     — save session transcript now")
    print("  /quit     — end session and save transcript")
    print("──────────────────────────────────────────")


def _save_session(
    intake_id: str,
    history: list[dict],
    active_persona: SubPersona,
) -> None:
    """Persist the chat transcript to data/chat/{intake_id}/session_{ts}.json."""
    out_dir = CHAT_DIR / intake_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"session_{timestamp}.json"

    payload = {
        "intake_id": intake_id,
        "persona": active_persona.name,
        "saved_at": datetime.now().isoformat(),
        "message_count": len(history),
        "messages": history,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Session saved to {out_path}")


def main() -> int:
    args = parse_args()
    setup_logging(LOG_DIR)
    # Chat UX: keep the terminal clean — file handler still gets DEBUG/INFO,
    # but the console should only surface WARNING and above.
    _root = logging.getLogger("icp_agent")
    for _h in _root.handlers:
        if isinstance(_h, logging.StreamHandler) and not isinstance(
            _h, logging.FileHandler
        ):
            _h.setLevel(logging.WARNING)
    log = logging.getLogger("icp_agent.run_chat")

    intake_id = _resolve_intake_id(args.intake_id)
    if intake_id is None:
        print("✗ No intake found. Run the intake form first.")
        return 1

    icp_path = _find_latest_icp(intake_id)
    if icp_path is None:
        print(
            f"✗ No ICP found for {intake_id}. Run: "
            f"python scripts/run_pipeline.py --intake-id {intake_id}"
        )
        return 1

    icp_doc = load_icp(icp_path)
    log.debug("Loaded ICP %s (%s)", icp_doc.icp_id, icp_path)

    active_persona = _select_persona(icp_doc)
    system_prompt = build_chat_system_prompt(icp_doc, active_persona)
    _print_chat_banner(active_persona)

    history: list[dict[str, str]] = []

    try:
        while True:
            try:
                raw = input("You: ")
            except EOFError:
                print()
                _save_session(intake_id, history, active_persona)
                return 0

            stripped = raw.strip()
            if not stripped:
                continue

            if stripped == "/quit":
                _save_session(intake_id, history, active_persona)
                print("Goodbye.")
                return 0

            if stripped == "/save":
                _save_session(intake_id, history, active_persona)
                continue

            if stripped == "/history":
                print(
                    f"Messages in history: {len(history)} | "
                    f"Est. tokens: ~{len(history) * 150}"
                )
                continue

            if stripped == "/persona":
                active_persona = _select_persona(icp_doc)
                system_prompt = build_chat_system_prompt(icp_doc, active_persona)
                print(f"Switched to: {active_persona.name}")
                continue

            try:
                processed = process_input(text=stripped)
            except ValueError as exc:
                print(f"✗ {exc}")
                continue

            evidence_block = decompose_and_retrieve(
                processed, active_persona, PERSIST_DIR
            )

            print(f"\n{active_persona.name}: ", end="", flush=True)
            stream_response(stripped, evidence_block, history, system_prompt)
    except KeyboardInterrupt:
        print("\n\nSession interrupted.")
        _save_session(intake_id, history, active_persona)
        return 0


if __name__ == "__main__":
    sys.exit(main())
