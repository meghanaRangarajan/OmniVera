"""Tool implementations and JSON schemas for the orchestrator agent.

Four tools, in the order the agent must call them:

    1. inspect_state            — resolve intake_id, count chunks, list ICPs (read-only)
    2. build_processing_index   — rebuild ChromaDB from data/processed/transcripts.json
    3. run_synthesis            — full Layer 3 pipeline; saves data/icp/{intake}/icp_*.json
    4. launch_chat_server       — terminal handoff; blocks forever on success

Every tool returns a JSON-serialisable dict. Failures are surfaced as
``{"status": "error", "message": str}`` rather than propagated, so the
agent loop can hand them back to the model as a tool_result.
"""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import chromadb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool 1 — inspect_state
# ---------------------------------------------------------------------------


def inspect_state(intake_id: str) -> dict[str, Any]:
    """Read the current state of the pipeline for ``intake_id``. Read-only.

    Resolves ``"latest"`` to the registry's latest intake, confirms the
    intake.json file is on disk and parses, counts ChromaDB chunks by
    source_type, and lists any existing ICP JSON files for that intake.

    Returns:
        On success::

            {
                "status": "ok",
                "intake_id": str,
                "product_name": str,
                "intake_json_exists": bool,
                "chunk_counts": {
                    "total": int,
                    "transcript": int,
                    "deep_research": int,
                    "reddit": int,
                },
                "existing_icp_files": [str, ...],
            }

        On failure: ``{"status": "error", "message": str}``.
    """
    try:
        # Lazy imports so test fixtures can patch chat_web before the dispatch
        # table resolves these symbols.
        from icp_agent.chat_web import ICP_DIR, PERSIST_DIR, _resolve_intake_id
        from icp_agent.intake import load_intake, load_registry
        from icp_agent.pipeline import COLLECTION_NAME

        resolved = _resolve_intake_id(intake_id)
        if resolved is None:
            return {
                "status": "error",
                "message": (
                    "No intakes registered. Run the intake form first "
                    "(scripts/run_web_intake.py)."
                ),
            }

        registry = load_registry()
        entry = next((e for e in registry.intakes if e.intake_id == resolved), None)
        if entry is None:
            available = [e.intake_id for e in registry.intakes]
            return {
                "status": "error",
                "message": (
                    f"Intake '{resolved}' not in registry. "
                    f"Available: {available}"
                ),
            }

        intake_dir = entry.intake_dir
        intake_json = intake_dir / "intake.json"
        if not intake_json.exists():
            return {
                "status": "error",
                "message": (
                    f"intake.json missing on disk for '{resolved}': {intake_json}"
                ),
            }

        intake = load_intake(intake_dir)

        chunk_counts = _count_chunks(PERSIST_DIR, COLLECTION_NAME)

        icp_subdir = ICP_DIR / resolved
        if icp_subdir.exists():
            icp_files = sorted(
                p.name
                for p in icp_subdir.glob("icp_*.json")
                if p.name != "icp_partial.json"
            )
        else:
            icp_files = []

        return {
            "status": "ok",
            "intake_id": resolved,
            "product_name": intake.product_name,
            "intake_json_exists": True,
            "chunk_counts": chunk_counts,
            "existing_icp_files": icp_files,
        }
    except Exception as exc:
        logger.debug("inspect_state failed:\n%s", traceback.format_exc())
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }


def _count_chunks(persist_dir: Path, collection_name: str) -> dict[str, int]:
    """Count chunks in the icp_evidence collection grouped by source_type.

    Mirrors :func:`icp_agent.pipeline._count_chunks_by_source` but does not
    raise if the directory is missing — returns all-zero counts instead.
    """
    counts: dict[str, int] = {
        "total": 0,
        "transcript": 0,
        "deep_research": 0,
        "reddit": 0,
    }

    if not Path(persist_dir).exists():
        return counts

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    total = collection.count()
    counts["total"] = total
    if total == 0:
        return counts

    result = collection.get(include=["metadatas"])
    for meta in result["metadatas"]:
        src = meta.get("source_type")
        if src in counts:
            counts[src] += 1
    return counts


# ---------------------------------------------------------------------------
# Tool 2 — build_processing_index
# ---------------------------------------------------------------------------


def build_processing_index() -> dict[str, Any]:
    """Rebuild the ChromaDB index from data/processed/transcripts.json.

    Wraps the ``build_index_from_disk`` callable extracted in
    ``scripts/build_index.py``. Always rebuilds from scratch.

    Returns:
        ``{"status": "ok", "chunks": int, "persist_dir": str}`` or
        ``{"status": "error", "message": str}``.
    """
    try:
        # ChromaDB caches one PersistentClient per path as a process-wide
        # singleton. If inspect_state (or any earlier code in this process)
        # opened a client at the same path, the singleton holds stale SQLite
        # handles after we shutil.rmtree the directory, and the rebuild
        # fails with "attempt to write a readonly database". Clearing the
        # system cache forces a fresh client on the next open.
        _clear_chromadb_cache()

        # scripts/ is not on sys.path by default; the orchestrator entry point
        # ensures it is, but we also add it defensively for tests.
        scripts_dir = _project_root() / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import build_index as build_index_script  # type: ignore[import-not-found]

        result = build_index_script.build_index_from_disk(
            build_index_script.TRANSCRIPTS_JSON,
            build_index_script.CHROMA_DIR,
        )
        return {
            "status": "ok",
            "chunks": int(result["chunks"]),
            "persist_dir": str(result["persist_dir"]),
        }
    except Exception as exc:
        logger.debug("build_processing_index failed:\n%s", traceback.format_exc())
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }


def _project_root() -> Path:
    # src/icp_agent/agent/tools.py → project root is three levels up.
    return Path(__file__).resolve().parents[3]


def _clear_chromadb_cache() -> None:
    """Drop chromadb's per-path PersistentClient singleton cache.

    Safe no-op if the SDK doesn't expose the helper (older versions): the
    rebuild may then collide with stale handles, but we'd rather fail loudly
    than silently pin a wrong chromadb version.
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001
        logger.debug("chromadb cache clear unavailable", exc_info=True)


# ---------------------------------------------------------------------------
# Tool 3 — run_synthesis
# ---------------------------------------------------------------------------


def run_synthesis(intake_id: str) -> dict[str, Any]:
    """Run the full ICP synthesis pipeline for ``intake_id``.

    Mirrors the orchestration in ``scripts/run_pipeline.py``: verify inputs,
    decompose queries, retrieve evidence, synthesize each of the 7 sections
    with per-section error isolation, identify sub-personas, assemble and
    save the ICP. Pipeline ``print`` output is redirected to the agent log
    so it does not interleave with the model's streamed narration on stdout.

    Per-section contract: a section that raises is recorded in
    ``partial_failures`` and synthesis continues. The overall call still
    returns ``"ok"`` as long as verify / decompose / retrieve / assemble
    succeed.

    Returns:
        On success::

            {
                "status": "ok",
                "icp_path": str,
                "sections_completed": int,    # sections with >= 1 claim
                "personas": int,
                "partial_failures": [str, ...],
            }

        On failure (verify, decompose, retrieve, or assemble crash)::

            {"status": "error", "message": str}
    """
    from icp_agent.models import EvidencedClaim
    from icp_agent.pipeline import (
        SYNTHESIS_SECTIONS,
        _save_partial,
        assemble_and_save_icp,
        build_synthesis_queries,
        retrieve_evidence_for_synthesis,
        synthesize_section,
        synthesize_sub_personas,
        verify_pipeline_inputs,
    )

    persist_dir = _project_root() / "data" / "processed" / "chroma"

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            verification = verify_pipeline_inputs(intake_id, persist_dir)
            intake = verification["intake"]

            queries = build_synthesis_queries(intake)
            evidence = retrieve_evidence_for_synthesis(queries, persist_dir)

            completed_sections: dict[str, list[EvidencedClaim]] = {}
            partial_failures: list[str] = []
            for section in SYNTHESIS_SECTIONS:
                try:
                    claims = synthesize_section(
                        section, evidence.get(section, []), intake
                    )
                except Exception as exc:
                    logger.error(
                        "Section %s synthesis crashed: %s", section, exc
                    )
                    logger.debug("Traceback:\n%s", traceback.format_exc())
                    claims = []
                    partial_failures.append(section)
                completed_sections[section] = claims
                _save_partial(intake.intake_id, completed_sections)

            try:
                sub_personas = synthesize_sub_personas(
                    evidence, completed_sections, intake
                )
            except Exception as exc:
                logger.error("Sub-persona synthesis crashed: %s", exc)
                logger.debug("Traceback:\n%s", traceback.format_exc())
                sub_personas = []

            icp = assemble_and_save_icp(
                intake=intake,
                completed_sections=completed_sections,
                sub_personas=sub_personas,
                all_evidence=evidence,
            )
    except Exception as exc:
        # Always preserve any captured output before exiting.
        _flush_captured("run_synthesis (failed)", captured)
        logger.debug("run_synthesis failed:\n%s", traceback.format_exc())
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }

    _flush_captured("run_synthesis", captured)

    icp_path = (
        _project_root()
        / "data"
        / "icp"
        / icp.intake_id
        / f"icp_{icp.icp_id}.json"
    )
    sections_completed = sum(
        1 for claims in completed_sections.values() if claims
    )
    return {
        "status": "ok",
        "icp_path": str(icp_path),
        "sections_completed": sections_completed,
        "personas": len(sub_personas),
        "partial_failures": partial_failures,
    }


def _flush_captured(label: str, buf: io.StringIO) -> None:
    """Emit a multi-line string captured from a redirected stdout to the log."""
    text = buf.getvalue()
    if not text.strip():
        return
    logger.info("[%s stdout]\n%s", label, text.rstrip("\n"))


# ---------------------------------------------------------------------------
# Tool 4 — launch_chat_server
# ---------------------------------------------------------------------------


def launch_chat_server(intake_id: str, host: str, port: int) -> dict[str, Any]:
    """Start the Omni-Vera chat web server for ``intake_id``. Terminal handoff.

    On success this call NEVER returns — ``chat_web.run`` blocks the process
    serving Flask until SIGINT. A returned dict therefore always indicates
    a pre-launch failure.

    Returns (only on early failure):
        ``{"status": "error", "message": str}``.
    """
    try:
        from icp_agent import chat_web
        from icp_agent.chat_web import (
            _find_latest_icp,
            _list_personas,
            _print_banner,
            _resolve_intake_id,
            launch_browser_async,
        )
        from icp_agent.models import load_icp

        resolved = _resolve_intake_id(intake_id)
        if resolved is None:
            return {
                "status": "error",
                "message": "No intakes registered — cannot launch chat server.",
            }

        icp_path = _find_latest_icp(resolved)
        if icp_path is None:
            return {
                "status": "error",
                "message": (
                    f"No ICP found for intake '{resolved}'. "
                    "Synthesis must succeed first."
                ),
            }

        icp_doc = load_icp(icp_path)
        personas = _list_personas(icp_doc)
        url = f"http://{host}:{port}?intake_id={resolved}"

        _print_banner(
            product_name=icp_doc.product_name,
            intake_id=resolved,
            persona_count=len(personas),
            url=url,
        )
        launch_browser_async(url)

        chat_web.run(host=host, port=port, debug=False)

        # If run() returns (clean shutdown via SIGINT), treat it as ok-no-op
        # so the agent loop can exit cleanly.
        return {
            "status": "ok",
            "message": "Chat server stopped.",
        }
    except Exception as exc:
        logger.debug("launch_chat_server failed:\n%s", traceback.format_exc())
        return {
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# JSON schemas (Anthropic tool-use format) and dispatch table
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_state",
        "description": (
            "Inspect the current state of the pipeline for a given intake. "
            "Resolves 'latest' to a real intake_id, confirms intake.json "
            "exists on disk, counts ChromaDB chunks by source type, and "
            "lists any existing ICP files. Read-only. Always call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intake_id": {
                    "type": "string",
                    "description": (
                        "Registered intake_id, or the literal string "
                        "'latest' to use the most recent intake."
                    ),
                }
            },
            "required": ["intake_id"],
        },
    },
    {
        "name": "build_processing_index",
        "description": (
            "Build the ChromaDB index from data/processed/transcripts.json. "
            "Always rebuilds from scratch — any existing index is removed "
            "first. Returns the resulting chunk count. Must succeed before "
            "run_synthesis, which hard-fails if zero transcript chunks exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_synthesis",
        "description": (
            "Run the full ICP synthesis pipeline: verify inputs, decompose "
            "queries, retrieve evidence, synthesize each of the 7 sections "
            "(per-section error-isolated), identify sub-personas, assemble "
            "and save the ICP. Returns the saved ICP path plus a "
            "'partial_failures' list naming any sections that crashed — "
            "narrate those to the user honestly if non-empty. Requires "
            "build_processing_index to have succeeded first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intake_id": {
                    "type": "string",
                    "description": (
                        "The resolved intake_id (NOT 'latest' — use the "
                        "value returned by inspect_state)."
                    ),
                }
            },
            "required": ["intake_id"],
        },
    },
    {
        "name": "launch_chat_server",
        "description": (
            "TERMINAL HANDOFF: starts the Omni-Vera chat web server. This call "
            "BLOCKS FOREVER under normal operation — the Python process "
            "becomes the web server. If it returns a tool_result, that "
            "means it failed before the server started. Requires a saved "
            "ICP for the given intake."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intake_id": {
                    "type": "string",
                    "description": "The resolved intake_id.",
                },
                "host": {
                    "type": "string",
                    "description": "Bind host, e.g. 127.0.0.1.",
                },
                "port": {
                    "type": "integer",
                    "description": "Bind port, e.g. 5001.",
                },
            },
            "required": ["intake_id", "host", "port"],
        },
    },
]


DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "inspect_state": inspect_state,
    "build_processing_index": build_processing_index,
    "run_synthesis": run_synthesis,
    "launch_chat_server": launch_chat_server,
}
