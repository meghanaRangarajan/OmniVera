"""Smoke test: verify environment and Anthropic connectivity."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent.parent / ".env")

# 1. Anthropic key
if os.environ.get("ANTHROPIC_API_KEY"):
    print("✓ Anthropic key present")
else:
    print("✗ ANTHROPIC_API_KEY not set")

# 2. Reddit credentials
PLACEHOLDERS = {"replace-me", "your_client_id", "your_client_secret", "your_username", "your_password", ""}
reddit_vars = {
    "REDDIT_CLIENT_ID": os.environ.get("REDDIT_CLIENT_ID", ""),
    "REDDIT_CLIENT_SECRET": os.environ.get("REDDIT_CLIENT_SECRET", ""),
}
all_real = all(v not in PLACEHOLDERS for v in reddit_vars.values())
if all_real:
    print("✓ Reddit creds present")
else:
    print("⚠ Reddit creds still placeholder — expected at this stage")

# 3. Python version
print(f"Python {sys.version}")

# 4. Real Anthropic API call
client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=16,
    messages=[{"role": "user", "content": "Reply with exactly the word: PONG"}],
)
print(f"Anthropic response: {message.content[0].text}")
