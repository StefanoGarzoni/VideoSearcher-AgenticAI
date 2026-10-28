#!/usr/bin/env bash
# Requires GitHub CLI (gh) and an authenticated user: gh auth login
set -euo pipefail
cd "$(dirname "$0")/.."
command -v gh >/dev/null || { echo "Install GitHub CLI: https://cli.github.com/" >&2; exit 1; }
[[ -d .git ]] || { echo "Use the prepared Git repository archive with .git metadata." >&2; exit 1; }
gh auth status
gh repo create StefanoGarzoni/VideoSearcher-AgenticAI --private --source . --remote origin --push \
  --description "Agentic video understanding, MCP tools and energy-aware local-LLM benchmarking (academic research)"
printf 'Repository URL: https://github.com/StefanoGarzoni/VideoSearcher-AgenticAI\n'
