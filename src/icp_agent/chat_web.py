"""Flask backend for the Omni-Vera chat web UI.

Serves a single-page chat experience backed by the existing
:mod:`icp_agent.chat` pipeline (input normalization, query decomposition,
RAG retrieval, Sonnet streaming). Streaming responses are delivered to the
browser over Server-Sent Events so tokens render in real time.

Run with: python scripts/run_chat_server.py
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    session,
    stream_with_context,
)
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)

from icp_agent.chat import (
    CHAT_MODEL,
    build_chat_system_prompt,
    decompose_and_retrieve,
    process_input,
)
from icp_agent.intake import load_registry
from icp_agent.models import ICPDocument, SubPersona, load_icp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths / app setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ICP_DIR = PROJECT_ROOT / "data" / "icp"
CHAT_DIR = PROJECT_ROOT / "data" / "chat"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
PERSIST_DIR = PROJECT_ROOT / "data" / "processed" / "chroma"

# Per-process cache of loaded ICPDocument objects keyed by intake_id so we
# don't re-read the JSON on every request. The Flask session only stores the
# intake_id + persona index + sid, which keeps session cookies small and safe.
_ICP_CACHE: dict[str, tuple[ICPDocument, Path]] = {}

# Conversation histories keyed by the per-browser session id ``sid``. This
# lives in-process (not in the Flask cookie) because streaming responses
# flush headers before the generator runs, so cookie-session writes inside
# the generator never reach the client.
_CHAT_HISTORIES: dict[str, list[dict[str, str]]] = {}

# Active uploaded attachment per browser sid. Extracted text can easily
# exceed the 4KB signed-cookie limit, so this must stay server-side
# rather than living in ``session[...]`` directly.
_ACTIVE_ATTACHMENTS: dict[str, dict[str, str]] = {}


def _session_id() -> str:
    """Return the session id for this browser, minting one if needed."""
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_urlsafe(16)
        session["sid"] = sid
    return sid


def _get_history() -> list[dict[str, str]]:
    return _CHAT_HISTORIES.setdefault(_session_id(), [])

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=None,
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB cap on message payload

_secret = os.environ.get("FLASK_SECRET_KEY")
if _secret:
    app.secret_key = _secret
else:
    app.secret_key = secrets.token_hex(32)
    logger.warning(
        "FLASK_SECRET_KEY not set — generated an ephemeral secret. "
        "Sessions will not survive a server restart."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_intake_id(intake_id: str) -> str | None:
    """Resolve the magic value ``latest`` to the registry's latest_intake_id."""
    if intake_id and intake_id != "latest":
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


def _load_icp_for(intake_id: str) -> tuple[ICPDocument, Path]:
    """Load + cache the ICPDocument for a given intake_id.

    Raises FileNotFoundError if no ICP exists yet for that intake.
    """
    cached = _ICP_CACHE.get(intake_id)
    if cached is not None:
        return cached

    icp_path = _find_latest_icp(intake_id)
    if icp_path is None:
        raise FileNotFoundError(
            f"No ICP found for intake '{intake_id}'. "
            f"Run the synthesis pipeline first."
        )
    icp_doc = load_icp(icp_path)
    _ICP_CACHE[intake_id] = (icp_doc, icp_path)
    return icp_doc, icp_path


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
    seen: set[str] = set()
    uniq: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


def _build_full_icp_persona(icp_doc: ICPDocument) -> SubPersona:
    """Composite persona synthesized from the top-level ICPDocument."""
    return SubPersona(
        name="Full ICP — composite",
        description=(
            f"Composite customer profile for {icp_doc.product_name} across "
            "all archetypes"
        ),
        key_traits=[c.claim for c in icp_doc.pains[:3]] or ["composite view"],
        motivations=[c.claim for c in icp_doc.gains[:3]] or ["composite view"],
        objections=[c.claim for c in icp_doc.objections[:3]] or ["composite view"],
        evidence_chunk_ids=_union_chunk_ids(icp_doc)[:10] or ["composite"],
    )


def _list_personas(icp_doc: ICPDocument) -> list[SubPersona]:
    """Ordered persona list used by the UI. Index 0 is always the composite."""
    return [_build_full_icp_persona(icp_doc), *icp_doc.sub_personas]


def _get_active_persona() -> tuple[ICPDocument, SubPersona]:
    """Re-derive the active persona from the session state. Raises on failure."""
    intake_id = session.get("intake_id")
    idx = session.get("active_persona_index", 0)
    if not intake_id:
        raise RuntimeError("No active chat session — call /api/chat/init first.")
    icp_doc, _ = _load_icp_for(intake_id)
    personas = _list_personas(icp_doc)
    if idx < 0 or idx >= len(personas):
        idx = 0
    return icp_doc, personas[idx]


def _persist_error_log(exc: BaseException) -> Path:
    """Dump a traceback to data/logs/chat_{ts}.log and return the path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"chat_{ts}.log"
    log_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )
    return log_path


def _ext_for(file_name: str, file_type: str | None) -> str:
    """Pick a filesystem extension for the temp file based on name or MIME."""
    if file_name and "." in file_name:
        return "." + file_name.rsplit(".", 1)[-1].lower()
    if file_type:
        # Accept either MIME (image/png) or bare extension
        if "/" in file_type:
            return "." + file_type.split("/", 1)[-1].lower()
        return "." + file_type.lstrip(".").lower()
    return ".bin"


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_FILE_EXTS = {".pdf", ".docx", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    return render_template("chat.html")


@app.route("/api/chat/init")
def init_session() -> Response:
    """Initialize (or re-initialize) the chat session for a given intake_id."""
    raw_id = request.args.get("intake_id", "latest")
    try:
        intake_id = _resolve_intake_id(raw_id)
        if intake_id is None:
            return jsonify({
                "ok": False,
                "error": "No intakes found. Run the intake form first.",
            }), 400

        icp_doc, icp_path = _load_icp_for(intake_id)
        personas = _list_personas(icp_doc)

        session["intake_id"] = intake_id
        session["active_persona_index"] = 0
        session["icp_doc_path"] = str(icp_path)
        sid = _session_id()
        _CHAT_HISTORIES[sid] = []

        active = _ACTIVE_ATTACHMENTS.get(sid)
        active_payload = (
            {"filename": active["filename"], "type": active["type"]}
            if active else None
        )

        return jsonify({
            "ok": True,
            "intake_id": intake_id,
            "product_name": icp_doc.product_name,
            "personas": [
                {
                    "index": i,
                    "name": p.name,
                    "description": p.description,
                }
                for i, p in enumerate(personas)
            ],
            "active_persona_index": 0,
            "active_attachment": active_payload,
        })

    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("init_session failed — log at %s", log_path)
        return jsonify({
            "ok": False,
            "error": "Failed to initialize chat. See server logs.",
        }), 500


@app.route("/api/chat/message", methods=["POST"])
def post_message() -> Response:
    """Handle a user turn. Returns an SSE stream of the persona's response."""
    try:
        data = request.get_json(silent=True) or {}
        user_text: str = (data.get("message") or "").strip()
        file_data: str | None = data.get("file_data")
        file_name: str = data.get("file_name") or ""
        file_type: str | None = data.get("file_type")

        if not user_text and not file_data:
            return jsonify({"ok": False, "error": "Empty message."}), 400

        icp_doc, active_persona = _get_active_persona()
        sid = _session_id()
        history = _get_history()
        messages_history_snapshot = list(history)

        # ---- If a new file was uploaded, extract + replace active attachment ----
        new_upload_filename: str | None = None
        attachment_replaced_existing = False
        if file_data:
            extracted_text, att_type = _extract_uploaded(
                file_data, file_name, file_type
            )
            attachment_replaced_existing = sid in _ACTIVE_ATTACHMENTS
            _ACTIVE_ATTACHMENTS[sid] = {
                "filename": file_name,
                "extracted_text": extracted_text,
                "type": att_type,
            }
            new_upload_filename = file_name

        # ---- Build the text that goes to retrieval + the model ----
        active = _ACTIVE_ATTACHMENTS.get(sid)
        if active:
            question = user_text if user_text else "Please analyze the attached file."
            processed_text = (
                f"[ATTACHMENT CONTEXT — {active['filename']}]:\n"
                f"{active['extracted_text']}\n\n"
                f"[USER QUESTION]: {question}"
            )
        else:
            processed_text = process_input(text=user_text)

        # ---- Retrieval ----
        evidence_block = decompose_and_retrieve(
            processed_text, active_persona, PERSIST_DIR
        )

        # ---- Stream Sonnet response over SSE ----
        system_prompt = build_chat_system_prompt(icp_doc, active_persona)
        enriched_user = (
            f"{processed_text}\n\n"
            f"RELEVANT EVIDENCE:\n{evidence_block}\n\n"
            "Answer from your persona's perspective. Cite sources inline after "
            "each claim as [transcript: name], [reddit: username], or "
            "[research: section]. If evidence doesn't cover the question, "
            "say so explicitly."
        )
        messages_for_api = messages_history_snapshot + [
            {"role": "user", "content": enriched_user}
        ]

        persona_name = active_persona.name
        user_display = user_text if user_text else f"(file: {file_name})"

        def generate():
            if new_upload_filename is not None:
                yield _sse_event({
                    "type": "attachment_set",
                    "filename": new_upload_filename,
                    "replaced": attachment_replaced_existing,
                })
            yield _sse_event({"type": "start", "persona": persona_name})
            response_parts: list[str] = []
            try:
                client = anthropic.Anthropic()
                with client.messages.stream(
                    model=CHAT_MODEL,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=messages_for_api,
                ) as stream:
                    for text in stream.text_stream:
                        response_parts.append(text)
                        yield _sse_event({"type": "token", "text": text})
            except Exception as stream_exc:
                log_path = _persist_error_log(stream_exc)
                logger.exception(
                    "Streaming error — log at %s", log_path,
                )
                yield _sse_event({
                    "type": "error",
                    "message": "The model had trouble responding. Please try again.",
                })
                return

            full_text = "".join(response_parts)

            # Append to in-process history keyed by sid. The sid was set
            # in /api/chat/init before any streaming started, so it is
            # already in the browser's cookie.
            history.append({"role": "user", "content": user_display})
            history.append({"role": "assistant", "content": full_text})

            yield _sse_event({"type": "done", "full_text": full_text})

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("post_message failed — log at %s", log_path)
        return jsonify({
            "ok": False,
            "error": "Server error handling message. See server logs.",
        }), 500


@app.route("/api/chat/persona", methods=["POST"])
def switch_persona() -> Response:
    try:
        data = request.get_json(silent=True) or {}
        idx_raw = data.get("persona_index")
        if not isinstance(idx_raw, int):
            return jsonify({"ok": False, "error": "persona_index must be an integer"}), 400

        intake_id = session.get("intake_id")
        if not intake_id:
            return jsonify({"ok": False, "error": "No active session."}), 400

        icp_doc, _ = _load_icp_for(intake_id)
        personas = _list_personas(icp_doc)
        if idx_raw < 0 or idx_raw >= len(personas):
            return jsonify({"ok": False, "error": "persona_index out of range"}), 400

        session["active_persona_index"] = idx_raw
        persona = personas[idx_raw]
        return jsonify({
            "ok": True,
            "persona_name": persona.name,
            "description": persona.description,
        })

    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("switch_persona failed — log at %s", log_path)
        return jsonify({"ok": False, "error": "Server error."}), 500


@app.route("/api/chat/save", methods=["POST"])
def save_session() -> Response:
    try:
        intake_id = session.get("intake_id")
        if not intake_id:
            return jsonify({"ok": False, "error": "No active session."}), 400

        _, active_persona = _get_active_persona()
        history = _get_history()

        out_dir = CHAT_DIR / intake_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"session_{ts}.json"
        payload = {
            "intake_id": intake_id,
            "persona": active_persona.name,
            "saved_at": datetime.now().isoformat(),
            "message_count": len(history),
            "messages": history,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return jsonify({"ok": True, "path": str(out_path)})

    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("save_session failed — log at %s", log_path)
        return jsonify({"ok": False, "error": "Server error saving session."}), 500


@app.route("/api/chat/attachment", methods=["DELETE"])
def clear_attachment() -> Response:
    """Drop the currently-active uploaded attachment for this browser."""
    try:
        sid = _session_id()
        _ACTIVE_ATTACHMENTS.pop(sid, None)
        return jsonify({"ok": True})
    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("clear_attachment failed — log at %s", log_path)
        return jsonify({"ok": False, "error": "Server error."}), 500


@app.route("/api/chat/export", methods=["POST"])
def export_pdf() -> Response:
    """Render the current chat to a PDF on disk and return it as a download."""
    try:
        data = request.get_json(silent=True) or {}
        messages = data.get("messages") or []
        persona_name = (data.get("persona_name") or "Persona").strip() or "Persona"
        product_name = (data.get("product_name") or "Product").strip() or "Product"
        intake_id = (data.get("intake_id") or "").strip()
        if not intake_id:
            return jsonify({"ok": False, "error": "intake_id is required."}), 400
        if not isinstance(messages, list):
            return jsonify({"ok": False, "error": "messages must be a list."}), 400

        out_dir = CHAT_DIR / intake_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"export_{ts}.pdf"

        _build_export_pdf(
            out_path=out_path,
            messages=messages,
            persona_name=persona_name,
            product_name=product_name,
            intake_id=intake_id,
        )

        safe_product = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in product_name
        ) or "chat"
        download_name = f"omni_vera_chat_{safe_product}_{ts}.pdf"
        return send_file(
            str(out_path),
            as_attachment=True,
            download_name=download_name,
            mimetype="application/pdf",
        )

    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("export_pdf failed — log at %s", log_path)
        return jsonify({"ok": False, "error": "Server error exporting PDF."}), 500


@app.route("/api/chat/history")
def history_stats() -> Response:
    try:
        history = _get_history()
        total_chars = sum(len(m.get("content", "")) for m in history)
        # Rough Sonnet-ish estimate: ~4 chars/token.
        token_estimate = total_chars // 4
        return jsonify({
            "message_count": len(history),
            "token_estimate": token_estimate,
        })
    except Exception as e:
        log_path = _persist_error_log(e)
        logger.exception("history_stats failed — log at %s", log_path)
        return jsonify({"ok": False, "error": "Server error."}), 500


# ---------------------------------------------------------------------------
# Internal: prepare user input
# ---------------------------------------------------------------------------


def _extract_uploaded(
    file_data: str,
    file_name: str,
    file_type: str | None,
) -> tuple[str, str]:
    """Decode a base64 upload, run process_input on it, return (text, type).

    The returned ``type`` is "image" for jpg/png/webp, else "file".
    """
    ext = _ext_for(file_name, file_type)
    if ext not in _IMAGE_EXTS and ext not in _FILE_EXTS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Accepted: "
            f"{sorted(_IMAGE_EXTS | _FILE_EXTS)}"
        )

    try:
        raw = base64.b64decode(file_data)
    except Exception as exc:
        raise ValueError(f"Malformed file payload: {exc}") from exc

    tmp_dir = Path(tempfile.mkdtemp(prefix="omni_vera_upload_"))
    tmp_path = tmp_dir / (file_name or f"upload{ext}")
    tmp_path.write_bytes(raw)

    try:
        if ext in _IMAGE_EXTS:
            processed = process_input(image_path=tmp_path)
            att_type = "image"
        else:
            processed = process_input(file_path=tmp_path)
            att_type = "file"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass

    return processed, att_type


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Internal: PDF export
# ---------------------------------------------------------------------------

_PDF_BG = rl_colors.HexColor("#1a1f2e")
_PDF_TEXT = rl_colors.HexColor("#e5e7eb")
_PDF_ACCENT = rl_colors.HexColor("#2dd4bf")
_PDF_DIM = rl_colors.HexColor("#9ca3af")
_PDF_SEP = rl_colors.HexColor("#2a3142")


class _FooterCanvas(Canvas):
    """Canvas that defers showPage so we can stamp 'Page X of Y' on every page."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Canvas.__init__(self, *args, **kwargs)
        self._saved_pages: list[dict[str, Any]] = []

    def showPage(self) -> None:  # type: ignore[override]
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:  # type: ignore[override]
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self._draw_footer(total)
            Canvas.showPage(self)
        Canvas.save(self)

    def _draw_footer(self, total: int) -> None:
        self.saveState()
        self.setFillColor(_PDF_DIM)
        self.setFont("Helvetica", 8)
        self.drawCentredString(
            letter[0] / 2,
            0.4 * inch,
            f"Page {self._pageNumber} of {total} · Omni-Vera",
        )
        self.restoreState()


def _pdf_draw_background(canvas: Canvas, _doc: Any) -> None:
    canvas.saveState()
    canvas.setFillColor(_PDF_BG)
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.restoreState()


def _pdf_escape(s: Any) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _build_export_pdf(
    out_path: Path,
    messages: list[dict[str, Any]],
    persona_name: str,
    product_name: str,
    intake_id: str,
) -> None:
    """Render a cover page plus the conversation into ``out_path``."""
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.8 * inch,
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([
        PageTemplate(id="main", frames=frame, onPage=_pdf_draw_background)
    ])

    title_style = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=30,
        textColor=_PDF_TEXT, alignment=1, leading=34, spaceAfter=6,
    )
    tagline_style = ParagraphStyle(
        "Tagline", fontName="Helvetica-Oblique", fontSize=12,
        textColor=_PDF_ACCENT, alignment=1, leading=16, spaceAfter=40,
    )
    meta_style = ParagraphStyle(
        "Meta", fontName="Helvetica", fontSize=11,
        textColor=_PDF_TEXT, alignment=1, leading=18, spaceAfter=6,
    )
    user_label_style = ParagraphStyle(
        "UserLabel", fontName="Helvetica-Bold", fontSize=10,
        textColor=_PDF_ACCENT, alignment=2, leading=12, spaceAfter=3,
    )
    omni_vera_label_style = ParagraphStyle(
        "OmniVeraLabel", fontName="Helvetica-Bold", fontSize=10,
        textColor=_PDF_ACCENT, alignment=0, leading=12, spaceAfter=3,
    )
    user_body_style = ParagraphStyle(
        "UserBody", fontName="Helvetica", fontSize=11,
        textColor=_PDF_TEXT, alignment=2, leading=16, spaceAfter=8,
    )
    omni_vera_body_style = ParagraphStyle(
        "OmniVeraBody", fontName="Helvetica", fontSize=11,
        textColor=_PDF_TEXT, alignment=0, leading=16, spaceAfter=8,
    )

    story: list[Any] = []

    # ---- Cover ----
    story.append(Spacer(1, 2.4 * inch))
    story.append(Paragraph("OMNI-VERA", title_style))
    story.append(Paragraph("The truth behind your customer", tagline_style))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(
        f"<b>Product:</b> {_pdf_escape(product_name)}", meta_style
    ))
    story.append(Paragraph(
        f"<b>Persona:</b> {_pdf_escape(persona_name)}", meta_style
    ))
    story.append(Paragraph(f"<b>Exported:</b> {now_str}", meta_style))
    story.append(Paragraph(
        f"<b>Intake ID:</b> {_pdf_escape(intake_id)}", meta_style
    ))
    story.append(PageBreak())

    # ---- Conversation ----
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            story.append(Paragraph("You", user_label_style))
            story.append(Paragraph(_pdf_escape(content), user_body_style))
        elif role == "assistant":
            label = msg.get("persona") or persona_name
            story.append(Paragraph(_pdf_escape(label), omni_vera_label_style))
            story.append(Paragraph(_pdf_escape(content), omni_vera_body_style))
        else:
            continue
        story.append(HRFlowable(
            width="100%", thickness=0.4, color=_PDF_SEP,
            spaceBefore=2, spaceAfter=8,
        ))

    doc.build(story, canvasmaker=_FooterCanvas)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_banner(
    product_name: str,
    intake_id: str,
    persona_count: int,
    url: str,
) -> None:
    """Print the chat-server startup banner to stdout."""
    line = "═" * 44
    print()
    print(line)
    print("Omni-Vera — The truth behind your customer")
    print(line)
    print(f"Intake:   {product_name} ({intake_id})")
    print(f"Personas: {persona_count} available")
    print(f"URL:      {url}")
    print(line)
    print()


def _open_browser_after_delay(url: str, delay_seconds: float = 1.5) -> None:
    """Sleep briefly so the Flask server is up, then open the URL in a browser."""
    time.sleep(delay_seconds)
    webbrowser.open(url)


def launch_browser_async(url: str, delay_seconds: float = 1.5) -> None:
    """Spawn a daemon thread that opens ``url`` after a short delay."""
    threading.Thread(
        target=_open_browser_after_delay,
        args=(url, delay_seconds),
        daemon=True,
    ).start()


def run(host: str = "127.0.0.1", port: int = 5001, debug: bool = False) -> None:
    """Launch the Flask server. Called by scripts/run_chat_server.py."""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run(debug=True)
