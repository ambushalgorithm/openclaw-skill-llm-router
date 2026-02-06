#!/usr/bin/env bash
# llm-router-status-log: View recent ledger entries with pretty formatting
# Usage: llm-router-status-log [N]  (default: last 20 entries)

LEDGER_FILE="${LLM_ROUTER_LEDGER_PATH:-$HOME/.llm-router-ledger.jsonl}"
NUM_ENTRIES="${1:-20}"

if [ ! -f "$LEDGER_FILE" ]; then
    echo "No ledger found at $LEDGER_FILE"
    exit 1
fi

# Show last N entries with pretty formatting
tail -n "$NUM_ENTRIES" "$LEDGER_FILE" | while read -r line; do
    echo "$line" | jq -r '[.ts_ms, .category, .provider, .model, .cost_usd, .is_estimate] | @tsv' 2>/dev/null || echo "$line"
done | awk -F'\t' 'BEGIN { 
    printf "%-20s %-12s %-12s %-25s %-10s %s\n", "Timestamp", "Category", "Provider", "Model", "Cost(USD)", "Estimate"
    printf "%s\n", "────────────────────────────────────────────────────────────────────────────────────────────"
}
{
    ts=$1; category=$2; provider=$3; model=$4; cost=$5; is_est=$6
    if (is_est == "true") est="✓"; else est=""
    printf "%-20s %-12s %-12s %-25s %-10s %s\n", ts, category, provider, model, cost, est
}'

# Show total
TOTAL=$(tail -n "$NUM_ENTRIES" "$LEDGER_FILE" | jq -s 'map(.cost_usd) | add' 2>/dev/null)
echo ""
echo "Total cost (last $NUM_ENTRIES entries): \$${TOTAL:-0}"
