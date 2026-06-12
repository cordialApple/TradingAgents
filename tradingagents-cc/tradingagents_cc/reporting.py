# Vendored layout from cli/main.py (run_analysis tee + save_report_to_disk) and tradingagents/graph/trading_graph.py (_log_state).
"""Per-run artifacts: live report tee, tool log, consolidated report, state log, decisions.jsonl.

Tree (parent-identical where the parent writes one):

    {results_dir}/{TICKER}/{date}/reports/{section}.md   <- live tee, 7 canonical sections
    {results_dir}/{TICKER}/{date}/message_tool.log       <- "HH:MM:SS [Tool Call] name(args)"
    {results_dir}/{TICKER}/{date}/complete_report.md     <- "## I." .. "## V." consolidated report
    {results_dir}/{TICKER}/TradingAgentsStrategy_logs/full_states_log_{date}.json
    {results_dir}/decisions.jsonl                        <- one JSON line per run (schema_version 1)
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.dataflows.utils import safe_ticker_component

__all__ = [
    "REPORT_SECTIONS",
    "SECTION_HEADERS",
    "RunReporter",
    "append_decision",
]

# The 7 canonical report sections (parent MessageBuffer.REPORT_SECTIONS order).
REPORT_SECTIONS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
)

# Sub-section headers the parent CLI prepends when teeing debate snapshots into
# the composite sections (investment_plan / final_trade_decision). The pipeline
# composes "{header}\n{body}" and passes it to write_section — string-identical
# artifacts to the parent's live tee.
SECTION_HEADERS = {
    "bull": "### Bull Researcher Analysis",
    "bear": "### Bear Researcher Analysis",
    "research_manager": "### Research Manager Decision",
    "aggressive": "### Aggressive Analyst Analysis",
    "conservative": "### Conservative Analyst Analysis",
    "neutral": "### Neutral Analyst Analysis",
    "portfolio_manager": "### Portfolio Manager Decision",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Cap on the decisions.jsonl "error" field so one pathological traceback or
# vendor payload cannot bloat a record (and the single-write append below).
_ERROR_MAX_CHARS = 4000


def _truncate_error(error):
    """Coerce error to str and cap it at _ERROR_MAX_CHARS with a marker; None passes through."""
    if error is None:
        return None
    error = str(error)
    if len(error) <= _ERROR_MAX_CHARS:
        return error
    return error[:_ERROR_MAX_CHARS] + "... [truncated]"


def _safe_date_component(value) -> str:
    """Validate trade_date is a plain YYYY-MM-DD string before any path join."""
    value = str(value)
    if not _DATE_RE.fullmatch(value):
        raise ValueError(f"trade_date must be YYYY-MM-DD, got {value!r}")
    return value


class RunReporter:
    """Writes every per-run artifact for one (ticker, trade_date) analysis.

    Callers pass the already-normalized ticker (strip().upper()); both path
    components are validated here regardless.
    """

    def __init__(self, results_dir, ticker: str, trade_date: str):
        self.results_dir = Path(results_dir)
        self.ticker = safe_ticker_component(ticker)
        self.trade_date = _safe_date_component(trade_date)
        self.run_dir = self.results_dir / self.ticker / self.trade_date
        self.reports_dir = self.run_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.tool_log_path = self.run_dir / "message_tool.log"

    # --- Live tee (parent run_analysis decorators) ---

    def write_section(self, name: str, markdown: str) -> Path:
        """Overwrite reports/{name}.md with the latest snapshot (parent tee semantics).

        For the composite sections, pass f"{SECTION_HEADERS[key]}\\n{body}" so
        debate snapshots carry the original headers.
        """
        if name not in REPORT_SECTIONS:
            raise ValueError(
                f"unknown report section {name!r}; expected one of {REPORT_SECTIONS}"
            )
        path = self.reports_dir / f"{name}.md"
        path.write_text(str(markdown), encoding="utf-8")
        return path

    def write_tool_log(self, lines) -> Path:
        """Append pre-formatted "HH:MM:SS [Tool Call] name(args)" lines to message_tool.log.

        Accepts AgentResult.tool_call_log batches; safe to call once per stage.
        """
        lines = [
            str(line).replace("\r", " ").replace("\n", " ").rstrip()
            for line in lines
        ]
        lines = [line for line in lines if line]
        if lines:
            with open(self.tool_log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        return self.tool_log_path

    # --- Post-run consolidated report (parent save_report_to_disk layout) ---

    def write_complete_report(self, state: dict) -> Path:
        """Write complete_report.md with the parent's "## I." .. "## V." layout."""
        sections = []

        analyst_parts = [
            (title, state.get(key))
            for key, title in (
                ("market_report", "Market Analyst"),
                ("sentiment_report", "Social Analyst"),
                ("news_report", "News Analyst"),
                ("fundamentals_report", "Fundamentals Analyst"),
            )
            if state.get(key)
        ]
        if analyst_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
            sections.append(f"## I. Analyst Team Reports\n\n{content}")

        debate = state.get("investment_debate_state") or {}
        research_parts = [
            (title, debate.get(key))
            for key, title in (
                ("bull_history", "Bull Researcher"),
                ("bear_history", "Bear Researcher"),
                ("judge_decision", "Research Manager"),
            )
            if debate.get(key)
        ]
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

        if state.get("trader_investment_plan"):
            sections.append(
                f"## III. Trading Team Plan\n\n### Trader\n{state['trader_investment_plan']}"
            )

        risk = state.get("risk_debate_state") or {}
        risk_parts = [
            (title, risk.get(key))
            for key, title in (
                ("aggressive_history", "Aggressive Analyst"),
                ("conservative_history", "Conservative Analyst"),
                ("neutral_history", "Neutral Analyst"),
            )
            if risk.get(key)
        ]
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        if risk.get("judge_decision"):
            sections.append(
                f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}"
            )

        header = (
            f"# Trading Analysis Report: {self.ticker}\n\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        path = self.run_dir / "complete_report.md"
        path.write_text(header + "\n\n".join(sections), encoding="utf-8")
        return path

    # --- Final-state JSON (parent TradingAgentsGraph._log_state, same keys) ---

    def write_states_log(self, state: dict) -> Path:
        """Atomically write full_states_log_{date}.json under TradingAgentsStrategy_logs."""
        debate = state.get("investment_debate_state") or {}
        risk = state.get("risk_debate_state") or {}
        payload = {
            "company_of_interest": state.get("company_of_interest", ""),
            "trade_date": state.get("trade_date", ""),
            "market_report": state.get("market_report", ""),
            "sentiment_report": state.get("sentiment_report", ""),
            "news_report": state.get("news_report", ""),
            "fundamentals_report": state.get("fundamentals_report", ""),
            "investment_debate_state": {
                "bull_history": debate.get("bull_history", ""),
                "bear_history": debate.get("bear_history", ""),
                "history": debate.get("history", ""),
                "current_response": debate.get("current_response", ""),
                "judge_decision": debate.get("judge_decision", ""),
            },
            "trader_investment_decision": state.get("trader_investment_plan", ""),
            "risk_debate_state": {
                "aggressive_history": risk.get("aggressive_history", ""),
                "conservative_history": risk.get("conservative_history", ""),
                "neutral_history": risk.get("neutral_history", ""),
                "history": risk.get("history", ""),
                "judge_decision": risk.get("judge_decision", ""),
            },
            "investment_plan": state.get("investment_plan", ""),
            "final_trade_decision": state.get("final_trade_decision", ""),
        }

        directory = self.results_dir / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"full_states_log_{self.trade_date}.json"
        tmp_path = log_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=4, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(log_path)
        return log_path


def append_decision(results_dir, row: dict) -> Path:
    """Append one schema_version-1 record to {results_dir}/decisions.jsonl.

    The row is normalized to the fixed schema (missing fields become nulls /
    zeroed stats; ``error`` is coerced to str and truncated to 4000 chars with
    a ``... [truncated]`` marker). The record is appended as ONE OS-level
    write: encoded to bytes and pushed through a single ``os.write`` on an
    ``O_APPEND`` descriptor (``O_BINARY`` on Windows so the trailing newline
    is not CRLF-translated) — a buffered ``f.write`` would split lines larger
    than the stdio buffer into multiple OS writes — so concurrent (manual CLI
    run racing the scheduled routine) or crashed runs never interleave
    partial lines.
    """
    models = row.get("models") or {}
    stats = row.get("stats") or {}
    record = {
        "schema_version": 1,
        "run_id": row.get("run_id"),
        "ts": row.get("ts")
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": row.get("ticker"),
        "trade_date": row.get("trade_date"),
        "decision": row.get("decision"),
        "rating": row.get("rating"),
        "trader_action": row.get("trader_action"),
        "status": row.get("status"),
        "error": _truncate_error(row.get("error")),
        "models": {"quick": models.get("quick"), "deep": models.get("deep")},
        "stats": {
            "llm_calls": int(stats.get("llm_calls") or 0),
            "tool_calls": int(stats.get("tool_calls") or 0),
            "tokens_in": int(stats.get("tokens_in") or 0),
            "tokens_out": int(stats.get("tokens_out") or 0),
        },
        "stage_timings": row.get("stage_timings") or {},
        "fallbacks_used": row.get("fallbacks_used") or [],
        "report_dir": str(row["report_dir"]) if row.get("report_dir") else None,
        "duration_s": row.get("duration_s"),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "decisions.jsonl"
    data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_BINARY"):  # Windows: suppress CRT LF->CRLF text mode
        flags |= os.O_BINARY
    fd = os.open(path, flags)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return path
