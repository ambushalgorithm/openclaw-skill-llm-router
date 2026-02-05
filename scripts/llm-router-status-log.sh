#!/usr/bin/env bash
# Periodically log llm-router status snapshots with bounded disk growth.
#
# Env overrides:
#   LLM_ROUTER_STATUS_LOG_PATH   (default: /home/clawdbot/llm-router-status.log)
#   LLM_ROUTER_STATUS_LOG_MAX_BYTES (default: 5000000)
#   LLM_ROUTER_STATUS_LOG_KEEP_LINES (default: 5000)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${LLM_ROUTER_STATUS_LOG_PATH:-/home/clawdbot/llm-router-status.log}"
MAX_BYTES="${LLM_ROUTER_STATUS_LOG_MAX_BYTES:-5000000}"
KEEP_LINES="${LLM_ROUTER_STATUS_LOG_KEEP_LINES:-5000}"

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "==== $(date -u +"%Y-%m-%dT%H:%M:%SZ") ===="
  "$ROOT_DIR/llm-router-status.sh"
  echo
} >> "$LOG_PATH"

# Truncate the log if it grows too large.
if [[ -f "$LOG_PATH" ]]; then
  size=$(wc -c < "$LOG_PATH" | tr -d ' ')
  if [[ "$size" -gt "$MAX_BYTES" ]]; then
    tmp="${LOG_PATH}.tmp"
    tail -n "$KEEP_LINES" "$LOG_PATH" > "$tmp" || true
    mv "$tmp" "$LOG_PATH"
  fi
fi
