#!/bin/bash
# llm-router-cron-task: Sequential import → snapshot for data consistency
# This ensures snapshots always reflect the latest imported data

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${LLM_ROUTER_CRON_LOG:-$HOME/llm-router-cron.log}"

exec >> "$LOG_FILE" 2>&1

echo "[$(date -Iseconds)] Starting llm-router cron task"

# Step 1: Import OpenClaw transcripts into ledger
echo "[$(date -Iseconds)] Step 1: Importing transcripts..."
cd "$ROOT_DIR"
IMPORT_RESULT=$(echo '{"max_files":200}' | python3 -m src.main --import-openclaw-usage --stdin 2>&1)
echo "$IMPORT_RESULT" | jq -c '{status, stats}' || echo "Import raw: $IMPORT_RESULT"

# Check if import succeeded
if echo "$IMPORT_RESULT" | jq -e '.status == "ok"' > /dev/null 2>&1; then
    IMPORTED=$(echo "$IMPORT_RESULT" | jq -r '.stats.events_appended // 0')
    echo "[$(date -Iseconds)] Import complete: $IMPORTED events appended"
else
    echo "[$(date -Iseconds)] Import failed or no new events"
fi

# Step 2: Capture status snapshot (always runs, even if import had issues)
echo "[$(date -Iseconds)] Step 2: Capturing status snapshot..."
"$ROOT_DIR/scripts/llm-router-status-log.sh"
echo "[$(date -Iseconds)] Snapshot complete"

echo "[$(date -Iseconds)] Cron task finished"
echo "---"
