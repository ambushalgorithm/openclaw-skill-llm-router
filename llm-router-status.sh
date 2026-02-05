#!/usr/bin/env bash
# llm-router-status.sh - helper for inspecting router budgets/usage
#
# Usage:
#   ./llm-router-status.sh           # table view (per-category)
#   ./llm-router-status.sh --raw     # raw JSON from status mode
#
# This script assumes it is run from the skill repo root or that the
# current working directory contains src/main.py.

set -euo pipefail

# Determine repo root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if [[ "${1-}" == "--raw" ]]; then
  python3 -m src.main --status
  exit 0
fi

python3 -m src.main --status \
  | jq -r \
    '.modes.OpenClaw | to_entries[] | select(.value.limit_usd > 0)
     | [.key, .value.used_usd, .value.limit_usd, (.value.limit_usd - .value.used_usd), (100 * (.value.used_usd / .value.limit_usd))]
     | @tsv' \
  | awk 'BEGIN {
           printf "%-18s %-12s %-12s %-14s %s\n", "Category", "$ Used", "$ Limit", "$ Remaining", "%"
         }
         {
           used=$2+0; limit=$3+0; rem=$4+0; pct=$5+0;
           printf "%-18s $%-11.3f $%-11.3f $%-13.3f %.3f%%\n", $1, used, limit, rem, pct
         }'
