"""Import OpenClaw transcript usage into llm-router unified ledger.

This is a *local-only* importer: it reads OpenClaw session transcript JSONL files
and appends UsageEvent records into the router ledger.

Design goals:
- No LLM calls.
- Idempotent incremental imports via byte-offset cursor per file.
- Default category for imported events: Brain.

OpenClaw transcripts live under:
  ~/.openclaw/agents/<agentId>/sessions/*.jsonl
(or $OPENCLAW_STATE_DIR/agents/<agentId>/sessions/*.jsonl)

We only import assistant messages that include provider/model + usage + cost.total.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
import re

from . import router_core


DEFAULT_CATEGORY = "Brain"
PRIMARY_LLM_CATEGORY = "Primary LLM"


def _openclaw_state_dir() -> Path:
    override = os.getenv("OPENCLAW_STATE_DIR") or os.getenv("CLAWDBOT_STATE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".openclaw"


def _default_cursor_path() -> Path:
    override = os.getenv("LLM_ROUTER_OPENCLAW_IMPORT_STATE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".llm-router-openclaw-import-state.json"


@dataclass
class ImportStats:
    files_scanned: int = 0
    events_appended: int = 0
    lines_read: int = 0
    lines_parsed: int = 0
    lines_skipped: int = 0


def _load_cursor(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out: Dict[str, int] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, int) and v >= 0:
                    out[k] = v
            return out
    except Exception:
        return {}
    return {}


def _save_cursor(path: Path, cursor: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_transcript_files(state_dir: Path, *, max_files: Optional[int] = None) -> Iterable[Path]:
    agents_dir = state_dir / "agents"
    if not agents_dir.exists():
        return []

    def _gather() -> list[Path]:
        files: list[Path] = []
        for agent_dir in agents_dir.iterdir():
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.exists() or not sessions_dir.is_dir():
                continue
            for p in sessions_dir.iterdir():
                if not p.is_file():
                    continue
                # Only real session transcripts
                if p.name.endswith(".jsonl") and ".deleted." not in p.name:
                    files.append(p)
        # Prefer newest first to make partial runs still useful.
        try:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        except Exception:
            pass
        if max_files is not None:
            return files[: max(0, int(max_files))]
        return files

    return _gather()


# Header formats we treat as canonical (first line of assistant message):
#   Router: Category=<CategoryName>
#   Direct (no router): Category=Primary LLM
_ROUTER_CAT_RE = re.compile(r"(?m)^Router:\s*Category\s*=\s*([^\n|]+)")
_DIRECT_PRIMARY_RE = re.compile(
    r"(?mi)^Direct\s*\(no router\):\s*Category\s*=\s*Primary LLM\s*$",
)
_DIRECT_NO_ROUTER_RE = re.compile(r"\bDirect\s*\(no router\)\b", re.IGNORECASE)

# Heuristic patterns for agent_id -> category matching (lowest priority fallback)
_HEURISTIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^coding|codex|cline|roo|claude.?code", re.IGNORECASE), "Coding"),
    (re.compile(r"^heartbeat|cron|scheduler", re.IGNORECASE), "Heartbeat"),
    (re.compile(r"^image|vision|look.*|see.*", re.IGNORECASE), "Image Understanding"),
    (re.compile(r"^voice|speak|tts|audio", re.IGNORECASE), "Voice"),
    (re.compile(r"^web|search|browse|fetch", re.IGNORECASE), "Web Search"),
    (re.compile(r"^write|content|blog|draft", re.IGNORECASE), "Writing Content"),
    (re.compile(r"^main|primary|default|core", re.IGNORECASE), "Primary LLM"),
]


def _extract_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                # OpenClaw uses {type:"text", text:"..."}
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join([s for s in parts if s])
    return ""


def _load_agent_config(agent_id: str, state_dir: Path) -> dict:
    """Load agent configuration file if it exists.

    Config file location: <state_dir>/agents/<agent_id>/config.json
    Returns empty dict if file doesn't exist or is invalid.
    """
    try:
        config_path = state_dir / "agents" / agent_id / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _get_heuristic_category(agent_id: str) -> Optional[str]:
    """Infer category from agent_id using heuristic pattern matching.

    Returns None if no pattern matches.
    """
    if not agent_id:
        return None
    for pattern, category in _HEURISTIC_PATTERNS:
        if pattern.search(agent_id):
            return category
    return None


def _infer_category(text: str, agent_id: str, state_dir: Path) -> Optional[str]:
    """Determine category using layered precedence:

    1. Explicit header in message text (highest priority)
    2. Agent default category from config file
    3. Heuristic pattern matching on agent_id
    4. None (fallback to caller's default)

    This ensures explicit headers override defaults, and defaults override heuristics.
    """
    # Layer 1: Explicit header in message text
    if text:
        # Canonical direct/no-router header.
        if _DIRECT_PRIMARY_RE.search(text) or _DIRECT_NO_ROUTER_RE.search(text):
            return PRIMARY_LLM_CATEGORY

        m = _ROUTER_CAT_RE.search(text)
        if m:
            cat = m.group(1).strip()
            # normalize common variants
            if cat.lower() in ("web", "websearch", "web search"):
                return "Web Search"
            if cat.lower() in ("writing", "writing content"):
                return "Writing Content"
            if cat.lower() in ("image", "image understanding"):
                return "Image Understanding"
            return cat

    # Layer 2: Agent default category from config file
    agent_config = _load_agent_config(agent_id, state_dir)
    default_cat = agent_config.get("default_category")
    if default_cat and isinstance(default_cat, str):
        return default_cat.strip()

    # Layer 3: Heuristic pattern matching
    return _get_heuristic_category(agent_id)


# Backward compatibility - simple text-only inference for external callers
def _infer_category_from_text(text: str) -> Optional[str]:
    """Legacy function: only checks message headers.

    For full layered categorization, use _infer_category() instead.
    """
    return _infer_category(text, "", Path.home() / ".openclaw")  # dummy values, won't match heuristics


def _extract_usage_cost_total(usage: Any) -> Optional[float]:
    """Extract cost.total from usage dict.

    Returns the cost value (including 0.0 for free/subscription providers).
    Returns None only if cost is missing or invalid.
    """
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    if not isinstance(cost, dict):
        return None
    total = cost.get("total")
    try:
        if total is None:
            return None
        total_f = float(total)
        # Allow $0 costs (subscription providers like Ollama report $0)
        # Only reject negative or non-numeric values
        if total_f < 0:
            return None
        return total_f
    except Exception:
        return None


def _extract_tokens(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(usage, dict):
        return (None, None)

    def _get_int(key: str) -> Optional[int]:
        val = usage.get(key)
        try:
            if val is None:
                return None
            iv = int(val)
            return iv if iv >= 0 else None
        except Exception:
            return None

    # OpenClaw stores normalized usage as input/output/cacheRead/cacheWrite/totalTokens.
    return (_get_int("input"), _get_int("output"))


def _extract_agent_id_from_path(file_path: Path, state_dir: Path) -> str:
    """Extract agent_id from session file path.

    Path format: <state_dir>/agents/<agent_id>/sessions/<session_id>.jsonl
    Returns empty string if path doesn't match expected format.
    """
    try:
        # Get path parts relative to state_dir
        rel_path = file_path.relative_to(state_dir)
        parts = rel_path.parts
        # Expected: ('agents', '<agent_id>', 'sessions', '<file>.jsonl')
        if len(parts) >= 3 and parts[0] == "agents":
            return parts[1]
    except Exception:
        pass
    return ""


def import_openclaw_usage(
    *,
    category: str = DEFAULT_CATEGORY,
    mode: str = router_core.OPENCLAW_MODE,
    state_dir: Optional[Path] = None,
    cursor_path: Optional[Path] = None,
    source_prefix: str = "openclaw-transcript",
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    state_dir = state_dir or _openclaw_state_dir()
    cursor_path = cursor_path or _default_cursor_path()

    cursor = _load_cursor(cursor_path)
    stats = ImportStats()

    for file_path in iter_transcript_files(state_dir, max_files=max_files):
        stats.files_scanned += 1
        file_key = str(file_path)
        agent_id = _extract_agent_id_from_path(file_path, state_dir)
        offset = int(cursor.get(file_key, 0))
        try:
            size = file_path.stat().st_size
            if offset > size:
                offset = 0
        except Exception:
            continue

        try:
            with file_path.open("rb") as bf:
                bf.seek(offset)
                while True:
                    line = bf.readline()
                    if not line:
                        break
                    stats.lines_read += 1
                    offset = bf.tell()
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        stats.lines_parsed += 1
                    except Exception:
                        stats.lines_skipped += 1
                        continue

                    if not isinstance(obj, dict) or obj.get("type") != "message":
                        continue
                    msg = obj.get("message")
                    if not isinstance(msg, dict) or msg.get("role") != "assistant":
                        continue

                    provider = msg.get("provider") if isinstance(msg.get("provider"), str) else None
                    model = msg.get("model") if isinstance(msg.get("model"), str) else None
                    usage = msg.get("usage")
                    cost_total = _extract_usage_cost_total(usage)
                    if not provider or not model or cost_total is None:
                        continue

                    text = _extract_text(msg)
                    # Use layered categorization: header > agent config > heuristic > default
                    inferred_cat = _infer_category(text, agent_id, state_dir)
                    event_category = inferred_cat or category

                    tokens_in, tokens_out = _extract_tokens(usage)

                    # Preserve original OpenClaw timestamp (ms since epoch)
                    # so "today" and monthly accounting are correct.
                    ts_ms = None
                    try:
                        ts_raw = msg.get("timestamp")
                        ts_ms = int(ts_raw) if ts_raw is not None else None
                    except Exception:
                        ts_ms = None

                    # Append to router ledger + legacy accumulator.
                    router_core.log_usage_event(
                        mode=mode,
                        category=event_category,
                        cost_usd=cost_total,
                        provider=provider,
                        model=model,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        is_estimate=False,
                        source=f"{source_prefix}:{file_path.name}",
                        ts_ms=ts_ms,
                    )
                    stats.events_appended += 1

        except Exception:
            continue

        cursor[file_key] = offset

    _save_cursor(cursor_path, cursor)

    return {
        "status": "ok",
        "state_dir": str(state_dir),
        "cursor_path": str(cursor_path),
        "category": category,
        "mode": mode,
        "stats": stats.__dict__,
    }
