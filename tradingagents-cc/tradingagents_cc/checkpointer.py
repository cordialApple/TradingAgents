"""JSON checkpoint support for resumable pipeline runs.

Replaces the parent's LangGraph SqliteSaver with plain per-run JSON files:
one file per (ticker, trade_date) at
``{data_cache_dir}/cc_checkpoints/{TICKER}/{thread_id}.json`` containing
``{"thread_id", "completed", "state"}``. ``thread_id`` derivation is
identical to ``tradingagents/graph/checkpointer.py`` so both projects key
runs the same way. Writes are atomic (temp file + ``os.replace``) so a crash
mid-save never corrupts an existing checkpoint.

Loop stages ("Investment Debate" / "Risk Debate") save state after every
turn but join ``completed`` only when the loop exits; resume re-enters the
loop and the conditional logic picks the next speaker from the saved state.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from tradingagents.dataflows.utils import safe_ticker_component

# Exact stage-name strings recorded in a checkpoint's `completed` list,
# in pipeline order. Frozen by the design contract.
STAGES: tuple[str, ...] = (
    "Market Analyst",
    "Social Analyst",
    "News Analyst",
    "Fundamentals Analyst",
    "Investment Debate",
    "Research Manager",
    "Trader",
    "Risk Debate",
    "Portfolio Manager",
)


def thread_id(ticker: str, trade_date: str) -> str:
    """Deterministic thread ID for a ticker+date pair (parent-identical)."""
    return hashlib.sha256(f"{ticker.upper()}:{trade_date}".encode()).hexdigest()[:16]


# Upper bound accepted for a persisted debate-loop ``count``. Real runs top
# out at 2*max_debate_rounds / 3*max_risk_discuss_rounds (single digits by
# default); anything beyond this ceiling is treated as corruption.
_MAX_LOOP_COUNT = 1000


def _valid_loop_count(state: dict, key: str) -> bool:
    """True when ``state[key]["count"]`` is a sane non-negative int.

    The pipeline's debate loops iterate off these counts, so a corrupted
    value (huge, negative, or non-int) must invalidate the whole checkpoint
    rather than drive the loops into unbounded paid LLM turns or raise an
    uncaught ``TypeError`` mid-run. ``bool`` is rejected explicitly because
    it is an ``int`` subclass but never a legitimately saved count.
    """
    sub = state.get(key)
    if not isinstance(sub, dict):
        return False
    count = sub.get("count")
    return (
        isinstance(count, int)
        and not isinstance(count, bool)
        and 0 <= count <= _MAX_LOOP_COUNT
    )


class RunCheckpointer:
    """Load/save per-run JSON checkpoints under ``data_cache_dir``."""

    def __init__(self, data_cache_dir: str | Path, enabled: bool = True):
        self._root = Path(data_cache_dir) / "cc_checkpoints"
        self.enabled = enabled

    def _path(self, ticker: str, trade_date: str) -> Path:
        # safe_ticker_component raises on values that would escape the
        # checkpoints directory (e.g. "../..") — never silently sanitised.
        safe = safe_ticker_component(ticker).upper()
        return self._root / safe / f"{thread_id(ticker, trade_date)}.json"

    def load(self, ticker: str, trade_date: str) -> dict | None:
        """Return ``{"thread_id", "completed", "state"}`` or None.

        A missing, unreadable, or malformed checkpoint yields None so the
        pipeline falls back to a fresh run instead of crashing an unattended
        scheduled job. Malformed includes the loop-critical fields: both
        ``investment_debate_state["count"]`` and ``risk_debate_state["count"]``
        must be ints within ``0..1000`` (every legitimately saved state carries
        both), so a corrupted checkpoint can never feed the pipeline's debate
        loops a count that burns unbounded credit or raises a ``TypeError``.
        """
        if not self.enabled:
            return None
        path = self._path(ticker, trade_date)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        completed = data.get("completed") if isinstance(data, dict) else None
        state = data.get("state") if isinstance(data, dict) else None
        if (
            not isinstance(completed, list)
            or not all(isinstance(s, str) for s in completed)
            or not isinstance(state, dict)
            or not _valid_loop_count(state, "investment_debate_state")
            or not _valid_loop_count(state, "risk_debate_state")
        ):
            return None
        return {
            "thread_id": thread_id(ticker, trade_date),
            "completed": completed,
            "state": state,
        }

    def save(self, ticker: str, trade_date: str, completed: list[str], state: dict) -> None:
        """Atomically persist a checkpoint. ``state`` must be JSON-serializable."""
        if not self.enabled:
            return
        path = self._path(ticker, trade_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thread_id": thread_id(ticker, trade_date),
            "completed": list(completed),
            "state": state,
        }
        # Serialize before touching the temp file so a non-serializable state
        # raises without leaving a partial file behind.
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def clear(self, ticker: str, trade_date: str) -> None:
        """Remove the checkpoint for one ticker+date (and any stale temp file)."""
        path = self._path(ticker, trade_date)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        # Best-effort tidy: drop the per-ticker directory once empty.
        try:
            path.parent.rmdir()
        except OSError:
            pass

    def clear_all(self) -> int:
        """Remove every checkpoint. Returns the number of checkpoints deleted."""
        if not self._root.exists():
            return 0
        count = sum(1 for _ in self._root.glob("*/*.json"))
        shutil.rmtree(self._root, ignore_errors=True)
        return count
