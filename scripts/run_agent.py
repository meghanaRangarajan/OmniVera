"""Single CLI entry point for the ICP Agent v1 orchestrator.

Wraps Layer 2 (Processing) and Layer 3 (Synthesis & Activation) as a Claude
tool-use loop. Drives the pipeline end-to-end and hands off to the chat web
server in the same process.

Usage:
    python scripts/run_agent.py
    python scripts/run_agent.py --intake-id intake_20260418_192443
    python scripts/run_agent.py --intake-id latest --port 5001
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from icp_agent.agent.loop import run_agent  # noqa: E402

LOG_DIR = ROOT / "data" / "logs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the full ICP Agent v1 pipeline (index build → synthesis → "
            "chat web server) as a single Claude tool-use loop."
        ),
    )
    parser.add_argument(
        "--intake-id",
        default="latest",
        help="Registered intake_id, or 'latest' (default).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the Flask chat server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port for the Flask chat server (default: 5001).",
    )
    return parser.parse_args()


def _setup_logging(log_dir: Path) -> Path:
    """Attach a DEBUG file handler and an INFO stderr handler to ``icp_agent``.

    Mirrors the format used by ``icp_agent.pipeline.setup_logging`` but writes
    to ``agent_{ts}.log`` and emits INFO to stderr (not stdout) so it does not
    interleave with the model's streamed prose. Idempotent across re-invocations
    within the same Python process.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"agent_{timestamp}.log"

    root = logging.getLogger("icp_agent")
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        )
    )

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s  %(message)s")
    )

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.propagate = False

    return log_path


def main() -> int:
    args = _parse_args()
    log_path = _setup_logging(LOG_DIR)
    logging.getLogger("icp_agent.run_agent").info(
        "Agent log: %s", log_path
    )

    try:
        return run_agent(args.intake_id, args.host, args.port)
    except KeyboardInterrupt:
        print("\n  Agent interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
