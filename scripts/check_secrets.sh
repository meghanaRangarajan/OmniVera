#!/usr/bin/env bash
#
# Pre-commit guard. Blocks a commit if secrets, private notes, or research
# data are staged.
#
# Install:
#   ln -sf ../../scripts/check_secrets.sh .git/hooks/pre-commit
#
# Optional: create a gitignored `.private-terms` file in the repo root with one
# term per line (client names, confidential case studies, internal codenames).
# Any staged line containing one of them blocks the commit.
#
set -u
fail=0

# 1. Files that must never be committed.
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    .env|*/.env)
      echo "BLOCKED FILE: $f"; fail=1 ;;
  esac
  case "$f" in
    data/raw/*|data/processed/*|data/icp/*|data/chat/*|data/logs/*|data/inputs/*)
      case "$f" in
        *.gitkeep) ;;
        *) echo "RESEARCH DATA: $f"; fail=1 ;;
      esac ;;
  esac
done <<< "$(git diff --cached --name-only)"

# 2. Secret-shaped strings in staged content.
if git diff --cached -U0 | grep -nEi '^\+.*(sk-ant-[A-Za-z0-9_-]{15,}|sk-[A-Za-z0-9]{25,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'; then
  echo "SECRET-SHAPED STRING in staged diff (above)"
  fail=1
fi

# 3. Private terms, read from a gitignored .private-terms file.
if [ -f .private-terms ]; then
  while IFS= read -r term; do
    [ -z "$term" ] && continue
    case "$term" in \#*) continue ;; esac
    if git diff --cached -U0 | grep -niE "^\+.*${term}" > /dev/null; then
      echo "PRIVATE TERM in staged diff: ${term}"
      fail=1
    fi
  done < .private-terms
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "Commit blocked. Unstage the files above, or use --no-verify if you are certain."
  exit 1
fi

echo "check_secrets: clean"
