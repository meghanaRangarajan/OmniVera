"""
Entry point for the web-based intake.

Usage:
    python scripts/run_web_intake.py
"""
from __future__ import annotations

import logging
import sys
import threading
import time
import webbrowser

from icp_agent.web import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _open_browser_after_delay(url: str, delay_seconds: float = 1.5):
    """Open the browser a moment after the server starts."""
    time.sleep(delay_seconds)
    webbrowser.open(url)


def main():
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"

    print()
    print("  ICP Agent — Web Intake")
    print(f"  Server: {url}")
    print("  Submit one intake. The server will shut down automatically after submission.")
    print()

    # Open browser in a background thread
    threading.Thread(
        target=_open_browser_after_delay,
        args=(url,),
        daemon=True,
    ).start()

    try:
        run(host=host, port=port, debug=False)
    except (KeyboardInterrupt, SystemExit):
        print("  Server stopped. Intake complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
