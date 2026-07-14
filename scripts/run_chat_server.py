"""Entry point for the Omni-Vera chat web server.

Usage:
    python scripts/run_chat_server.py
    python scripts/run_chat_server.py --intake-id intake_20260418_192443
    python scripts/run_chat_server.py --port 5001

On startup, resolves the intake id (defaulting to ``latest``), verifies
the ICP synthesis output exists, prints a startup banner, opens the
browser, and starts a threaded Flask server with debug disabled.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from icp_agent.chat_web import (  # noqa: E402
    _find_latest_icp,
    _list_personas,
    _print_banner,
    _resolve_intake_id,
    app,
    launch_browser_async,
    run,
)
from icp_agent.models import load_icp  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("icp_agent.run_chat_server")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Omni-Vera chat web UI for a given intake.",
    )
    parser.add_argument(
        "--intake-id",
        default="latest",
        help="Registered intake_id, or 'latest' (default).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port for the Flask server (default: 5001).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the Flask server (default: 127.0.0.1).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    intake_id = _resolve_intake_id(args.intake_id)
    if intake_id is None:
        print("✗ No intakes registered. Run: python scripts/run_web_intake.py")
        return 1

    icp_path = _find_latest_icp(intake_id)
    if icp_path is None:
        print(
            f"✗ No ICP found for intake '{intake_id}'.\n"
            f"  Run: python scripts/run_pipeline.py --intake-id {intake_id}"
        )
        return 1

    icp_doc = load_icp(icp_path)
    personas = _list_personas(icp_doc)

    url = f"http://{args.host}:{args.port}?intake_id={intake_id}"
    _print_banner(
        product_name=icp_doc.product_name,
        intake_id=intake_id,
        persona_count=len(personas),
        url=url,
    )

    launch_browser_async(url)

    try:
        run(host=args.host, port=args.port, debug=False)
    except (KeyboardInterrupt, SystemExit):
        print("\n  Chat server stopped.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
