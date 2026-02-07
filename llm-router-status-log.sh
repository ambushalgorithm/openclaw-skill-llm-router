#!/bin/bash
# llm-router-status-log: Show status snapshots over time with formatted headers
# Usage: llm-router-status-log [N]  (default: last 10 snapshots)

LOG_FILE="${LLM_ROUTER_STATUS_LOG:-$HOME/.llm-router-status.log}"
NUM_SNAPSHOTS="${1:-10}"

# If log doesn't exist, create initial snapshot
if [ ! -f "$LOG_FILE" ] || [ ! -s "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Initializing status log..." >&2
    
    # Import and capture current status
    cd ~/Projects/openclaw-skill-llm-router
    python3 -m src.main --import-openclaw-usage </dev/null >/dev/null 2>&1
    
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" > "$LOG_FILE"
    ./llm-router-status.sh >> "$LOG_FILE" 2>/dev/null
    echo "" >> "$LOG_FILE"
fi

# Count total snapshots in file
TOTAL_SNAPSHOTS=$(grep -c '^====' "$LOG_FILE" 2>/dev/null || echo 0)

# Calculate which snapshot to start from (show last N)
START_SNAPSHOT=$((TOTAL_SNAPSHOTS - NUM_SNAPSHOTS + 1))
if [ "$START_SNAPSHOT" -lt 1 ]; then
    START_SNAPSHOT=1
fi

# Print from the calculated start point
current=0
while IFS= read -r line; do
    if [[ "$line" =~ ^==== ]]; then
        ((current++))
    fi
    if [ "$current" -ge "$START_SNAPSHOT" ]; then
        echo "$line"
    fi
done < "$LOG_FILE"
