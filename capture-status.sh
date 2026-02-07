#!/bin/bash
# Capture llm-router status snapshot (for cron)
LOG_FILE="${LLM_ROUTER_STATUS_LOG:-$HOME/.llm-router-status.log}"
mkdir -p "$(dirname "$LOG_FILE")"
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" >> "$LOG_FILE"
cd ~/Projects/openclaw-skill-llm-router && python3 -m src.main --import-openclaw-usage 2>/dev/null | tail -n +13 >> "$LOG_FILE" 2>/dev/null || ./llm-router-status.sh >> "$LOG_FILE" 2>/dev/null
echo "" >> "$LOG_FILE"

# Keep log from growing too large (keep last 500 entries ~5000 lines)
tail -n 5000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
