#!/usr/bin/env python3
"""Send loop-proof cost receipts to Discord using OpenClaw CLI.

This script is designed to run from cron with **no LLM calls**.

It:
- Reads new events appended to the llm-router ledger (JSONL)
- Aggregates the delta cost since the last run
- Sends a short "receipt" message via `openclaw message send`
- Writes a state cursor so it won't re-send

Loop-proofing:
- Receipt messages are posted under Router: Category=Heartbeat and include a
  [COST_RECEIPT] sentinel.
- The script ignores any ledger events whose category is "Heartbeat".

Env vars:
- LLM_ROUTER_LEDGER_PATH (default: ~/.llm-router-ledger.jsonl)
- LLM_ROUTER_RECEIPT_STATE_PATH (default: ~/.llm-router-receipt-state.json)
- LLM_ROUTER_RECEIPT_CHANNEL (default: discord)
- LLM_ROUTER_RECEIPT_TARGET  (required; e.g. Discord channel id)
- OPENCLAW_CLI (default: /home/clawdbot/.local/share/pnpm/openclaw)
- LLM_ROUTER_RECEIPT_MIN_DELTA_USD (default: 0.000)
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ledger_path() -> Path:
    return Path(os.getenv("LLM_ROUTER_LEDGER_PATH", "~/.llm-router-ledger.jsonl")).expanduser()


def _state_path() -> Path:
    return Path(os.getenv("LLM_ROUTER_RECEIPT_STATE_PATH", "~/.llm-router-receipt-state.json")).expanduser()


def _openclaw_cli() -> str:
    return os.getenv("OPENCLAW_CLI", "/home/clawdbot/.local/share/pnpm/openclaw")


@dataclass
class State:
    offset: int = 0


def load_state(path: Path) -> State:
    if not path.exists():
        return State()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            off = raw.get("offset")
            if isinstance(off, int) and off >= 0:
                return State(offset=off)
    except Exception:
        pass
    return State()


def save_state(path: Path, state: State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"offset": state.offset}, indent=2) + "\n", encoding="utf-8")


def read_new_events(ledger: Path, offset: int) -> Tuple[List[Dict[str, Any]], int]:
    if not ledger.exists():
        return ([], 0)

    size = ledger.stat().st_size
    if offset > size:
        offset = 0

    events: List[Dict[str, Any]] = []
    with ledger.open("rb") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                break
            offset = f.tell()
            try:
                obj = json.loads(line.decode("utf-8"))
                if isinstance(obj, dict):
                    events.append(obj)
            except Exception:
                continue

    return (events, offset)


def format_usd(x: float) -> str:
    return f"${x:.3f}"


def main() -> int:
    target = (os.getenv("LLM_ROUTER_RECEIPT_TARGET") or "").strip()
    if not target:
        # Nothing to do without a destination.
        return 0

    channel = (os.getenv("LLM_ROUTER_RECEIPT_CHANNEL") or "discord").strip()
    min_delta = float(os.getenv("LLM_ROUTER_RECEIPT_MIN_DELTA_USD", "0") or 0.0)

    ledger = _ledger_path()
    state_path = _state_path()
    state = load_state(state_path)

    events, new_offset = read_new_events(ledger, state.offset)

    # Filter to non-heartbeat, positive costs.
    deltas: Dict[str, float] = {}
    total = 0.0
    count = 0
    for ev in events:
        try:
            cat = str(ev.get("category") or "").strip() or "Unknown"
            if cat == "Heartbeat":
                continue
            cost = float(ev.get("cost_usd") or 0.0)
            if cost <= 0:
                continue
        except Exception:
            continue

        deltas[cat] = float(deltas.get(cat, 0.0) + cost)
        total += cost
        count += 1

    # Always advance cursor so we don't reprocess.
    state.offset = new_offset
    save_state(state_path, state)

    if count == 0 or total < min_delta:
        return 0

    # Compose a short receipt.
    # First line is canonical header; second line is human telemetry.
    parts = [
        "Router: Category=Heartbeat",
        f"[COST_RECEIPT] Imported {count} new event(s): total {format_usd(total)}",
    ]

    # Keep it small: show top categories by delta.
    top = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)[:5]
    breakdown = ", ".join([f"{k} +{format_usd(v)}" for k, v in top])
    if breakdown:
        parts.append(f"Breakdown: {breakdown}")

    msg = "\n".join(parts)

    cmd = [
        _openclaw_cli(),
        "message",
        "send",
        "--channel",
        channel,
        "--target",
        target,
        "--message",
        msg,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # Don't crash cron; state already advanced.
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
