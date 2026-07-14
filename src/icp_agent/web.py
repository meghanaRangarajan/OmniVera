"""
Flask backend for the ICP Agent web intake.

Serves the single-page intake form and handles form submission,
integrating with the existing src/icp_agent/intake.py module.

Run with: python scripts/run_web_intake.py
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError

from icp_agent.intake import (
    Intake,
    IntakeStatus,
    ResearchGoal,
    ingest_transcripts,
    register_intake,
    save_intake,
)
from icp_agent.transcripts import load_transcripts, save_transcripts

logger = logging.getLogger(__name__)


def _visible_len(s: str) -> int:
    return len(re.sub(r'\033\[[0-9;]*m', '', s))


def _pad_line(content: str, width: int = 60) -> str:
    return content + " " * max(0, width - _visible_len(content))


def _print_success_banner(
    intake_id: str,
    product_name: str,
    transcript_count: int,
    turn_count: int,
    intake_dir: str,
) -> None:
    """Print a visible success box to stdout after an intake is saved."""
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    line = "─" * 60
    print()
    print(f"{GREEN}┌{line}┐{RESET}")
    print(f"{GREEN}│{RESET}" + _pad_line(f"  {BOLD}{GREEN}✓ Intake saved successfully{RESET}") + f"{GREEN}│{RESET}")
    print(f"{GREEN}├{line}┤{RESET}")
    print(f"{GREEN}│{RESET}" + _pad_line(f"  Intake ID:    {intake_id[:42]}") + f"{GREEN}│{RESET}")
    print(f"{GREEN}│{RESET}" + _pad_line(f"  Product:      {product_name[:42]}") + f"{GREEN}│{RESET}")
    print(f"{GREEN}│{RESET}" + _pad_line(f"  Transcripts:  {transcript_count} parsed ({turn_count} turns)") + f"{GREEN}│{RESET}")
    print(f"{GREEN}│{RESET}" + _pad_line(f"  Location:     {intake_dir[:42]}") + f"{GREEN}│{RESET}")
    print(f"{GREEN}└{line}┘{RESET}")
    print()
    print("Server will shut down in 2 seconds. You can close the browser tab.")
    print()


def _shutdown_after_delay(delay_seconds: float = 2.0) -> None:
    """Schedule the Flask server to stop after a short delay.

    The delay lets the HTTP response finish flushing to the browser
    so the success screen renders before the server dies.
    """
    import os
    import signal
    import threading
    import time

    def _killer():
        time.sleep(delay_seconds)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=_killer, daemon=True).start()

# Flask app configured to find templates at <project_root>/templates
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=None,
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

BASE_DIR = PROJECT_ROOT / "data" / "inputs"


@app.route("/")
def index():
    """Serve the intake form."""
    return render_template("intake.html")


@app.route("/api/intake", methods=["POST"])
def create_intake():
    """Accept form submission, create intake directory, save files, parse transcripts."""
    try:
        # Parse JSON-encoded list fields
        competitors = json.loads(request.form.get("competitors", "[]"))
        goals_raw = json.loads(request.form.get("research_goals", "[]"))

        # Convert goal strings to enum values
        research_goals = [ResearchGoal(g) for g in goals_raw]

        # Build intake metadata
        intake_id = f"intake_{datetime.now():%Y%m%d_%H%M%S}"
        intake_dir = BASE_DIR / intake_id
        intake_dir.mkdir(parents=True, exist_ok=True)
        transcripts_subdir = intake_dir / "transcripts"
        transcripts_subdir.mkdir(exist_ok=True)

        # Save uploaded transcripts to a temp location, then ingest
        uploaded = request.files.getlist("transcripts")
        temp_paths: list[Path] = []
        for f in uploaded:
            if not f.filename:
                continue
            temp_path = transcripts_subdir / f.filename
            f.save(temp_path)
            temp_paths.append(temp_path)

        # ingest_transcripts normalizes paths and skips unsupported files
        # Since we saved directly to the intake folder, we just validate
        copied = [p for p in temp_paths if _is_supported(p)]

        # Parse transcripts immediately so we can return counts to the user
        parsed_path: Path | None = None
        transcript_count = 0
        turn_count = 0

        if copied:
            try:
                transcripts = load_transcripts(transcripts_subdir)
                if transcripts:
                    parsed_path = intake_dir / "parsed_transcripts.json"
                    save_transcripts(transcripts, parsed_path)
                    transcript_count = len(transcripts)
                    turn_count = sum(len(t.turns) for t in transcripts)
            except Exception as e:
                logger.warning(f"Transcript parsing failed: {e}")
                # Continue without parsed transcripts

        # Build and validate Intake object
        now = datetime.now()
        intake = Intake(
            product_name=request.form["product_name"].strip(),
            product_description=request.form["product_description"].strip(),
            company_name=request.form.get("company_name", "").strip() or None,
            icp_hypothesis=request.form["icp_hypothesis"].strip(),
            competitors=competitors,
            research_goals=research_goals,
            target_geography=request.form.get("target_geography", "").strip() or None,
            intake_id=intake_id,
            intake_dir=intake_dir,
            transcript_files=copied,
            parsed_transcripts_path=parsed_path,
            created_at=now,
            updated_at=now,
            version=1,
        )

        # Save intake.json and update registry
        save_intake(intake)
        register_intake(intake, status=IntakeStatus.INTAKE_COMPLETE, base_dir=BASE_DIR)

        logger.info(
            f"Created intake {intake_id}: {transcript_count} transcripts, "
            f"{turn_count} turns"
        )

        _print_success_banner(
            intake_id=intake_id,
            product_name=intake.product_name,
            transcript_count=transcript_count,
            turn_count=turn_count,
            intake_dir=str(intake_dir),
        )
        _shutdown_after_delay()

        return jsonify({
            "ok": True,
            "intake_id": intake_id,
            "intake_dir": str(intake_dir),
            "transcript_count": transcript_count,
            "turn_count": turn_count,
        })

    except ValidationError as e:
        # Convert pydantic errors to user-friendly messages
        msg = _friendly_validation_error(e)
        return jsonify({"ok": False, "error": msg}), 400

    except Exception as e:
        logger.exception("Intake creation failed")
        return jsonify({"ok": False, "error": f"Server error: {e}"}), 500


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in {".txt", ".md", ".pdf", ".docx", ".vtt", ".srt"}


def _friendly_validation_error(err: ValidationError) -> str:
    """Turn pydantic validation errors into one human-readable string."""
    first = err.errors()[0] if err.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", []))
    msg = first.get("msg", "Invalid input")
    return f"{field}: {msg}" if field else msg


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Entry point — called by scripts/run_web_intake.py."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run(debug=True)
