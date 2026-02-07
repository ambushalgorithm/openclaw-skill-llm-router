#!/bin/bash
# llm-router-dashboard: Comprehensive LLM router status check
# Runs in order: sync → ledger → status-log → status

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Separator matching widest content (ledger table = 92 chars)
SEP="────────────────────────────────────────────────────────────────────────────────────────────"

echo "╔═══════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    LLM ROUTER DASHBOARD - Daily Check                                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Sync transcript usage
echo "▶ Syncing transcript usage..."
cd "$ROOT_DIR" && python3 -m src.main --import-openclaw-usage --stdin < /dev/null 2>&1 | jq -c '{status, category, stats}'
echo ""

# 2. Ledger entries
echo "▶ Recent ledger entries (last 10):"
echo "$SEP"
"$ROOT_DIR/llm-router-status-ledger.sh" 10
echo ""

# 3. Status snapshots
echo "▶ Status history (last 3 snapshots):"
echo "$SEP"
"$ROOT_DIR/llm-router-status-log.sh" 3
echo ""

# 4. Current status table
echo "▶ Current budget status:"
echo "$SEP"
"$ROOT_DIR/llm-router-status.sh"
echo ""

echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "           Done. Run 'llm-router-dashboard' anytime for this full view."
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
