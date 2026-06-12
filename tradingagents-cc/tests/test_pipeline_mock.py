"""End-to-end mock pipeline tests.

Drives ``TradingAgentsPipeline.propagate`` with the ``MockAgentClient`` (zero
network, zero subscription credit) and asserts the frozen contract: state
shape and debate semantics, the per-run artifact tree, decisions.jsonl rows,
the memory-log Phase-A entry, structured-output fallback, checkpoint resume,
and past_context injection into the Portfolio Manager prompt.

All paths are routed through ``tmp_path``; every test builds its own config so
the file runs standalone (the suite conftest adds belt-and-braces isolation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents_cc.checkpointer import STAGES, thread_id
from tradingagents_cc.client import StageError
from tradingagents_cc.default_config import load_config
from tradingagents_cc.memory import TradingMemoryLog
from tradingagents_cc.mock import MockAgentClient
from tradingagents_cc.pipeline import TradingAgentsPipeline
from tradingagents_cc.rating import RATINGS_5_TIER
from tradingagents_cc.reporting import REPORT_SECTIONS

TICKER = "SPY"
TRADE_DATE = "2024-07-01"  # a past Monday; mock mode never fetches data for it

ANALYST_ROLES = (
    "market_analyst", "social_analyst", "news_analyst", "fundamentals_analyst",
)
ANALYST_STAGES = (
    "Market Analyst", "Social Analyst", "News Analyst", "Fundamentals Analyst",
)
REPORT_FIELDS = (
    "market_report", "sentiment_report", "news_report", "fundamentals_report",
)

# Fixed decisions.jsonl schema (reporting.append_decision, schema_version 1).
DECISION_ROW_KEYS = {
    "schema_version", "run_id", "ts", "ticker", "trade_date", "decision",
    "rating", "trader_action", "status", "error", "models", "stats",
    "stage_timings", "fallbacks_used", "report_dir", "duration_s",
}
STATS_KEYS = {"llm_calls", "tool_calls", "tokens_in", "tokens_out"}

LESSONS_MARKER = "Lessons from prior decisions and outcomes:"


def make_config(tmp_path: Path, depth: int = 1) -> dict:
    return load_config({
        "llm_backend": "mock",
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": str(tmp_path / "memory" / "trading_memory.md"),
        "max_debate_rounds": depth,
        "max_risk_discuss_rounds": depth,
    })


def checkpoint_path(cfg: dict) -> Path:
    return (
        Path(cfg["data_cache_dir"]) / "cc_checkpoints" / TICKER
        / f"{thread_id(TICKER, TRADE_DATE)}.json"
    )


def decision_rows(cfg: dict) -> list[dict]:
    path = Path(cfg["results_dir"]) / "decisions.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class PromptRecordingClient:
    """Delegating AgentClient that records (role, prompt) for every call.

    MockAgentClient records roles (``calls``/``call_counts``) but not prompt
    text; this wrapper adds it for the past_context injection assertions.
    """

    def __init__(self, inner: MockAgentClient) -> None:
        self._inner = inner
        self.prompts: list[tuple[str, str]] = []

    async def run(self, role: str, prompt: str, **kwargs):
        self.prompts.append((role, prompt))
        return await self._inner.run(role, prompt, **kwargs)


# ---------------------------------------------------------------------------
# Full run: state, debate semantics, artifacts, decisions row, memory, cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 3, 5])
async def test_full_run_mock(tmp_path: Path, depth: int) -> None:
    cfg = make_config(tmp_path, depth=depth)
    mock = MockAgentClient(cfg)
    pipeline = TradingAgentsPipeline(cfg, mock)

    state, signal = await pipeline.propagate(TICKER, TRADE_DATE)

    # --- All 4 analyst report fields non-empty ---
    for field in REPORT_FIELDS:
        assert state[field].strip(), f"{field} is empty"

    # --- Debate turn counts: exactly 2*N investment, 3*N risk ---
    debate = state["investment_debate_state"]
    risk = state["risk_debate_state"]
    assert debate["count"] == 2 * depth
    assert risk["count"] == 3 * depth
    for role in ("bull", "bear"):
        assert mock.call_counts[role] == depth
    for role in ("aggressive", "conservative", "neutral"):
        assert mock.call_counts[role] == depth

    # --- Bull speaks first (history is built as `history + "\n" + argument`,
    # verbatim parent semantics, so the first turn sits after one newline) ---
    history = debate["history"]
    assert history.lstrip("\n").startswith("Bull Analyst:")
    assert 0 <= history.find("Bull Analyst:") < history.find("Bear Analyst:")
    assert debate["bull_history"].lstrip("\n").startswith("Bull Analyst:")

    # --- Risk debate: Aggressive first, then Conservative, then Neutral ---
    risk_history = risk["history"]
    assert risk_history.lstrip("\n").startswith("Aggressive Analyst:")
    firsts = [
        risk_history.find(f"{label} Analyst:")
        for label in ("Aggressive", "Conservative", "Neutral")
    ]
    assert all(idx >= 0 for idx in firsts)
    assert firsts == sorted(firsts)

    # --- Final signal in the 5-tier vocabulary ---
    assert signal in RATINGS_5_TIER

    # --- Report tree: 7 section files + complete_report.md + states log ---
    run_dir = Path(cfg["results_dir"]) / TICKER / TRADE_DATE
    for section in REPORT_SECTIONS:
        section_path = run_dir / "reports" / f"{section}.md"
        assert section_path.is_file(), f"missing section file {section}.md"
        assert section_path.read_text(encoding="utf-8").strip()
    # Composite sections end HEADERLESS: after the Portfolio Manager stage a
    # final report-section pass overwrites every section with the raw state
    # value (parent end-of-run overwrite, cli/main.py:1167-1170), replacing
    # the headered live-tee debate snapshots.
    investment_plan_md = (run_dir / "reports" / "investment_plan.md").read_text(
        encoding="utf-8"
    )
    final_decision_md = (run_dir / "reports" / "final_trade_decision.md").read_text(
        encoding="utf-8"
    )
    assert investment_plan_md == state["investment_plan"]
    assert final_decision_md == state["final_trade_decision"]
    assert not investment_plan_md.startswith("###")
    assert not final_decision_md.startswith("###")

    complete = run_dir / "complete_report.md"
    assert complete.is_file()
    assert "## V. Portfolio Manager Decision" in complete.read_text(encoding="utf-8")

    states_log = (
        Path(cfg["results_dir"]) / TICKER / "TradingAgentsStrategy_logs"
        / f"full_states_log_{TRADE_DATE}.json"
    )
    assert states_log.is_file()
    payload = json.loads(states_log.read_text(encoding="utf-8"))
    assert payload["company_of_interest"] == TICKER
    assert payload["trade_date"] == TRADE_DATE
    assert payload["final_trade_decision"] == state["final_trade_decision"]

    # --- decisions.jsonl: one schema_version-1 row, status "ok" ---
    rows = decision_rows(cfg)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == DECISION_ROW_KEYS
    assert row["schema_version"] == 1
    assert row["status"] == "ok"
    assert row["error"] is None
    assert row["run_id"] == thread_id(TICKER, TRADE_DATE)
    assert row["ticker"] == TICKER
    assert row["trade_date"] == TRADE_DATE
    assert row["decision"] == signal
    assert row["rating"] == signal
    assert row["trader_action"] in ("Buy", "Hold", "Sell")
    assert row["models"] == {
        "quick": cfg["quick_think_llm"], "deep": cfg["deep_think_llm"],
    }
    assert set(row["stats"]) == STATS_KEYS
    assert set(row["stage_timings"]) == set(STAGES)
    assert row["fallbacks_used"] == []
    assert Path(row["report_dir"]) == run_dir

    # --- Memory log gained exactly one pending entry for this run ---
    pending = TradingMemoryLog(cfg).get_pending_entries()
    assert len(pending) == 1
    assert pending[0]["ticker"] == TICKER
    assert pending[0]["date"] == TRADE_DATE
    assert pending[0]["rating"] in RATINGS_5_TIER

    # --- Checkpoint cleared on success ---
    assert not checkpoint_path(cfg).exists()
    assert pipeline.checkpointer.load(TICKER, TRADE_DATE) is None


# ---------------------------------------------------------------------------
# Structured-output fallback (Hard invariant 4)
# ---------------------------------------------------------------------------


async def test_structured_fallback_portfolio_manager(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    mock = MockAgentClient(cfg, fail_structured={"portfolio_manager"})
    pipeline = TradingAgentsPipeline(cfg, mock)

    state, signal = await pipeline.propagate(TICKER, TRADE_DATE)

    # Exactly ONE free-text retry: structured attempt + fallback call.
    assert mock.call_counts["portfolio_manager"] == 2
    # The other structured stages succeeded on the first attempt.
    assert mock.call_counts["research_manager"] == 1
    assert mock.call_counts["trader"] == 1

    rows = decision_rows(cfg)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["fallbacks_used"] == ["Portfolio Manager"]

    # The free-text output still carries the **Rating** marker line, so the
    # vendored two-pass parse_rating recovers a real signal.
    assert signal in RATINGS_5_TIER
    assert rows[0]["decision"] == signal
    assert "**Rating**:" in state["final_trade_decision"]


# ---------------------------------------------------------------------------
# Checkpoint resume after a mid-run stage failure
# ---------------------------------------------------------------------------


async def test_resume_after_trader_failure(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    mock = MockAgentClient(cfg, raise_at_role="trader")
    pipeline = TradingAgentsPipeline(cfg, mock)

    # First run: the trader's first call raises; progress is checkpointed and
    # a failed decisions row is written before StageError propagates.
    with pytest.raises(StageError):
        await pipeline.propagate(TICKER, TRADE_DATE)

    cp_path = checkpoint_path(cfg)
    assert cp_path.is_file()
    checkpoint = json.loads(cp_path.read_text(encoding="utf-8"))
    for stage in (*ANALYST_STAGES, "Investment Debate", "Research Manager"):
        assert stage in checkpoint["completed"]
    assert "Trader" not in checkpoint["completed"]

    rows = decision_rows(cfg)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["decision"] is None
    assert rows[0]["error"]

    # Each analyst ran exactly once before the failure.
    for role in ANALYST_ROLES:
        assert mock.call_counts[role] == 1
    assert mock.call_counts["bull"] == 1
    assert mock.call_counts["bear"] == 1
    assert mock.call_counts["trader"] == 1  # the call that raised

    # Second run: resumes past the completed stages and finishes.
    state, signal = await pipeline.propagate(TICKER, TRADE_DATE)
    assert signal in RATINGS_5_TIER
    assert state["trader_investment_plan"].strip()

    # Analysts (and the finished debate/manager stages) were NOT re-run.
    for role in ANALYST_ROLES:
        assert mock.call_counts[role] == 1
    assert mock.call_counts["bull"] == 1
    assert mock.call_counts["bear"] == 1
    assert mock.call_counts["research_manager"] == 1
    assert mock.call_counts["trader"] == 2  # one raise + one success

    rows = decision_rows(cfg)
    assert [r["status"] for r in rows] == ["failed", "ok"]

    # Checkpoint cleared after the successful resume.
    assert not cp_path.exists()
    assert pipeline.checkpointer.load(TICKER, TRADE_DATE) is None


# ---------------------------------------------------------------------------
# past_context injection (memory log -> Portfolio Manager lessons line)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded-log", "empty-log"])
async def test_past_context_lessons_injection(tmp_path: Path, seeded: bool) -> None:
    cfg = make_config(tmp_path)
    reflection = "Trust confirmed momentum over headline noise."
    if seeded:
        # Pre-seed one RESOLVED entry (pending entries never feed past_context).
        log = TradingMemoryLog(cfg)
        log.store_decision(
            ticker=TICKER, trade_date="2024-06-03",
            final_trade_decision="**Rating**: Buy\n\nMomentum thesis held.",
        )
        log.update_with_outcome(
            ticker=TICKER, trade_date="2024-06-03",
            raw_return=0.05, alpha_return=0.02, holding_days=10,
            reflection=reflection,
        )
        assert log.get_pending_entries() == []

    recorder = PromptRecordingClient(MockAgentClient(cfg))
    pipeline = TradingAgentsPipeline(cfg, recorder)
    state, signal = await pipeline.propagate(TICKER, TRADE_DATE)
    assert signal in RATINGS_5_TIER

    pm_prompts = [p for role, p in recorder.prompts if role == "portfolio_manager"]
    assert len(pm_prompts) == 1
    if seeded:
        assert state["past_context"]
        assert f"Past analyses of {TICKER}" in state["past_context"]
        assert LESSONS_MARKER in pm_prompts[0]
        assert reflection in pm_prompts[0]
    else:
        assert state["past_context"] == ""
        assert LESSONS_MARKER not in pm_prompts[0]
    # The lessons line is PM-only (issue #572 parity): no other prompt has it.
    assert not any(
        LESSONS_MARKER in p for role, p in recorder.prompts
        if role != "portfolio_manager"
    )

    # The run's own decision joins the log as the (only) pending entry.
    entries = TradingMemoryLog(cfg).load_entries()
    pending = [e for e in entries if e["pending"]]
    assert len(pending) == 1
    assert pending[0]["date"] == TRADE_DATE
    assert len(entries) == (2 if seeded else 1)
