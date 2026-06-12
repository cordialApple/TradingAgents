"""Unit tests for vendored/ported pieces + parent-seam canary.

Covers the LLM-free building blocks: parse_rating, the schema renderers,
TradingMemoryLog lifecycle, conditional_logic loop formulas, the data-tool
layer (with route_to_vendor monkeypatched — no network), Phase-B return
fetching (with yfinance monkeypatched), the parent dataflows import seam,
and the SDK client's options builder. Nothing here spends subscription
credit or opens a socket.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest

from tradingagents_cc.checkpointer import RunCheckpointer
from tradingagents_cc.client import DISALLOWED_BUILTIN_TOOLS
from tradingagents_cc.reporting import append_decision
from tradingagents_cc.conditional_logic import (
    DEBATE_DONE,
    RISK_DONE,
    should_continue_debate,
    should_continue_risk_analysis,
)
from tradingagents_cc.memory import TradingMemoryLog
from tradingagents_cc.rating import RATINGS_5_TIER, parse_rating
from tradingagents_cc.schemas import (
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_trader_proposal,
)

_SEP = TradingMemoryLog._SEPARATOR


def _tools_data():
    """Import tools_data lazily — it is one of the two modules allowed to import the SDK."""
    pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc import tools_data
    return tools_data


# ---------------------------------------------------------------------------
# parse_rating (vendored rating.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", RATINGS_5_TIER)
def test_parse_rating_labeled_line_each_tier(word):
    text = f"Some preamble.\n**Rating**: {word}\n\nMore prose."
    assert parse_rating(text) == word


@pytest.mark.parametrize("word", RATINGS_5_TIER)
def test_parse_rating_lowercase_and_hyphen_separator(word):
    assert parse_rating(f"rating - {word.lower()}") == word


def test_parse_rating_bare_word_in_prose():
    assert parse_rating("After weighing the debate, we should sell the position.") == "Sell"
    assert parse_rating("Lean overweight, momentum is intact.") == "Overweight"


def test_parse_rating_garbage_defaults_to_hold():
    assert parse_rating("lorem ipsum dolor sit amet") == "Hold"
    assert parse_rating("") == "Hold"
    assert parse_rating("xyzzy", default="Sell") == "Sell"


def test_parse_rating_final_transaction_proposal_line():
    assert parse_rating("FINAL TRANSACTION PROPOSAL: **BUY**") == "Buy"
    assert parse_rating("FINAL TRANSACTION PROPOSAL: **SELL**") == "Sell"


def test_parse_rating_label_pass_beats_earlier_bare_word():
    # Pass 1 (explicit label) scans the whole text before the bare-word fallback.
    text = "There is heavy sell pressure near resistance.\n**Rating**: Buy"
    assert parse_rating(text) == "Buy"


def test_parse_rating_invalid_label_falls_through_to_bare_word():
    text = "Rating: Strong\nWe recommend underweight here."
    assert parse_rating(text) == "Underweight"


# ---------------------------------------------------------------------------
# Renderers (vendored schemas.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", list(TraderAction))
def test_render_trader_proposal_always_ends_with_proposal_line(action):
    rendered = render_trader_proposal(TraderProposal(action=action, reasoning="Because."))
    assert rendered.endswith(f"FINAL TRANSACTION PROPOSAL: **{action.value.upper()}**")
    assert f"**Action**: {action.value}" in rendered


def test_render_trader_proposal_optionals_keep_trailing_line():
    rendered = render_trader_proposal(
        TraderProposal(
            action=TraderAction.BUY,
            reasoning="Strong momentum.",
            entry_price=189.5,
            stop_loss=175.0,
            position_sizing="5% of portfolio",
        )
    )
    assert "**Entry Price**: 189.5" in rendered
    assert "**Stop Loss**: 175.0" in rendered
    assert "**Position Sizing**: 5% of portfolio" in rendered
    assert rendered.endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


@pytest.mark.parametrize("rating", list(PortfolioRating))
def test_render_pm_decision_contains_rating_line(rating):
    rendered = render_pm_decision(
        PortfolioDecision(
            rating=rating,
            executive_summary="Scale in over two weeks.",
            investment_thesis="Fundamentals support the move.",
        )
    )
    assert f"**Rating**: {rating.value}" in rendered
    # Round-trip: the vendored parser must recover the exact tier (invariant 5).
    assert parse_rating(rendered) == rating.value


def test_render_pm_decision_optionals():
    rendered = render_pm_decision(
        PortfolioDecision(
            rating=PortfolioRating.OVERWEIGHT,
            executive_summary="ES.",
            investment_thesis="IT.",
            price_target=210.0,
            time_horizon="3-6 months",
        )
    )
    assert "**Price Target**: 210.0" in rendered
    assert "**Time Horizon**: 3-6 months" in rendered
    assert "**Rating**: Overweight" in rendered


# ---------------------------------------------------------------------------
# TradingMemoryLog lifecycle (vendored memory.py)
# ---------------------------------------------------------------------------


def _make_log(tmp_path, **extra) -> TradingMemoryLog:
    return TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.md"), **extra})


def test_store_decision_exact_pending_tag_and_separator(tmp_path):
    log = _make_log(tmp_path)
    log.store_decision("NVDA", "2026-06-01", "**Rating**: Buy\n\nEnter on the dip.")

    raw = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert raw.splitlines()[0] == "[2026-06-01 | NVDA | Buy | pending]"
    assert "DECISION:\n**Rating**: Buy" in raw
    assert raw.endswith(_SEP)
    assert raw.count("<!-- ENTRY_END -->") == 1


def test_store_decision_idempotent_same_date_ticker(tmp_path):
    log = _make_log(tmp_path)
    log.store_decision("NVDA", "2026-06-01", "**Rating**: Buy\n\nFirst.")
    first = (tmp_path / "memory.md").read_text(encoding="utf-8")
    # Re-store same (date, ticker) — even with different text — is a no-op.
    log.store_decision("NVDA", "2026-06-01", "**Rating**: Sell\n\nSecond.")
    assert (tmp_path / "memory.md").read_text(encoding="utf-8") == first
    assert len(log.load_entries()) == 1

    # Different date or different ticker appends normally.
    log.store_decision("NVDA", "2026-06-02", "**Rating**: Hold\n\nNext day.")
    log.store_decision("MSFT", "2026-06-01", "**Rating**: Hold\n\nOther ticker.")
    entries = log.load_entries()
    assert len(entries) == 3
    assert all(e["pending"] for e in entries)
    raw = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert raw.count("<!-- ENTRY_END -->") == 3


def test_batch_update_rewrites_pending_tag_to_resolved(tmp_path):
    log = _make_log(tmp_path)
    log.store_decision("NVDA", "2026-06-01", "**Rating**: Buy\n\nEnter on the dip.")
    log.store_decision("NVDA", "2026-06-02", "**Rating**: Hold\n\nWait.")

    log.batch_update_with_outcomes([
        {
            "ticker": "NVDA",
            "trade_date": "2026-06-01",
            "raw_return": 0.032,
            "alpha_return": -0.011,
            "holding_days": 5,
            "reflection": "Entry timing was right; alpha lagged SPY.",
        },
    ])

    raw = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "[2026-06-01 | NVDA | Buy | +3.2% | -1.1% | 5d]" in raw
    assert "[2026-06-01 | NVDA | Buy | pending]" not in raw
    assert "REFLECTION:\nEntry timing was right; alpha lagged SPY." in raw
    assert "DECISION:\n**Rating**: Buy" in raw  # decision body preserved
    assert raw.count("<!-- ENTRY_END -->") == 2  # separators intact

    entries = log.load_entries()
    resolved = [e for e in entries if not e["pending"]]
    pending = [e for e in entries if e["pending"]]
    assert len(resolved) == 1 and len(pending) == 1
    assert resolved[0]["raw"] == "+3.2%"
    assert resolved[0]["alpha"] == "-1.1%"
    assert resolved[0]["holding"] == "5d"
    assert pending[0]["date"] == "2026-06-02"


def test_get_past_context_n_same_n_cross_slicing(tmp_path):
    log = _make_log(tmp_path)
    # Insertion order = chronological; pending NVDA 02-05 must never surface.
    log.store_decision("NVDA", "2026-02-02", "**Rating**: Buy\n\nNVDA day 1.")
    log.store_decision("MSFT", "2026-02-02", "**Rating**: Sell\n\nMSFT day 1.")
    log.store_decision("NVDA", "2026-02-03", "**Rating**: Buy\n\nNVDA day 2.")
    log.store_decision("MSFT", "2026-02-03", "**Rating**: Sell\n\nMSFT day 2.")
    log.store_decision("NVDA", "2026-02-04", "**Rating**: Buy\n\nNVDA day 3.")
    log.store_decision("NVDA", "2026-02-05", "**Rating**: Buy\n\nStill pending.")

    log.batch_update_with_outcomes([
        {"ticker": t, "trade_date": d, "raw_return": 0.01, "alpha_return": 0.005,
         "holding_days": 5, "reflection": f"Lesson {t} {d}."}
        for t, d in [
            ("NVDA", "2026-02-02"), ("MSFT", "2026-02-02"),
            ("NVDA", "2026-02-03"), ("MSFT", "2026-02-03"),
            ("NVDA", "2026-02-04"),
        ]
    ])

    ctx = log.get_past_context("NVDA", n_same=2, n_cross=1)

    assert "Past analyses of NVDA (most recent first):" in ctx
    assert "Recent cross-ticker lessons:" in ctx
    # Same-ticker: exactly the 2 most recent resolved NVDA entries, newest first.
    assert "[2026-02-04 | NVDA" in ctx
    assert "[2026-02-03 | NVDA" in ctx
    assert "[2026-02-02 | NVDA" not in ctx
    assert ctx.index("[2026-02-04 | NVDA") < ctx.index("[2026-02-03 | NVDA")
    # Cross-ticker: only the single most recent non-NVDA entry, reflection-only.
    assert "[2026-02-03 | MSFT" in ctx
    assert "[2026-02-02 | MSFT" not in ctx
    assert "Lesson MSFT 2026-02-03." in ctx
    # Pending entries are excluded entirely.
    assert "2026-02-05" not in ctx
    # Same-ticker entries are full-format (DECISION body included verbatim,
    # rating line and all — parent TradingMemoryLog parity); section order fixed.
    assert "DECISION:\n**Rating**: Buy\n\nNVDA day 3." in ctx
    assert ctx.index("Past analyses of NVDA") < ctx.index("Recent cross-ticker lessons:")


def test_rotation_prunes_oldest_resolved_never_pending(tmp_path):
    log = _make_log(tmp_path, memory_log_max_entries=2)
    # Oldest entry stays pending — rotation must keep it even when over cap.
    log.store_decision("NVDA", "2026-01-04", "**Rating**: Hold\n\nStays pending.")
    log.store_decision("NVDA", "2026-01-05", "**Rating**: Buy\n\nDay 1.")
    log.store_decision("NVDA", "2026-01-06", "**Rating**: Buy\n\nDay 2.")
    log.store_decision("NVDA", "2026-01-07", "**Rating**: Buy\n\nDay 3.")

    log.batch_update_with_outcomes([
        {"ticker": "NVDA", "trade_date": d, "raw_return": 0.02, "alpha_return": 0.01,
         "holding_days": 5, "reflection": f"Reflection {d}."}
        for d in ("2026-01-05", "2026-01-06", "2026-01-07")
    ])

    entries = log.load_entries()
    resolved_dates = [e["date"] for e in entries if not e["pending"]]
    pending_dates = [e["date"] for e in entries if e["pending"]]
    assert resolved_dates == ["2026-01-06", "2026-01-07"]  # oldest resolved dropped
    assert pending_dates == ["2026-01-04"]  # pending survives rotation


# ---------------------------------------------------------------------------
# conditional_logic loop formulas (ported, invariant 3)
# ---------------------------------------------------------------------------


def _simulate_debate(max_rounds: int) -> list[str]:
    state = {"investment_debate_state": {"count": 0, "current_response": ""}}
    order = []
    while True:
        nxt = should_continue_debate(state, max_rounds)
        order.append(nxt)
        if nxt == DEBATE_DONE:
            return order
        debate = state["investment_debate_state"]
        debate["count"] += 1
        prefix = "Bull" if nxt == "Bull Researcher" else "Bear"
        debate["current_response"] = f"{prefix} Analyst: turn {debate['count']}"


def _simulate_risk(max_rounds: int) -> list[str]:
    state = {"risk_debate_state": {"count": 0, "latest_speaker": ""}}
    order = []
    while True:
        nxt = should_continue_risk_analysis(state, max_rounds)
        order.append(nxt)
        if nxt == RISK_DONE:
            return order
        risk = state["risk_debate_state"]
        risk["count"] += 1
        risk["latest_speaker"] = nxt  # speaker name verbatim, prefix drives rotation


@pytest.mark.parametrize("n", [1, 3, 5])
def test_debate_speaker_order_exact(n):
    expected = ["Bull Researcher", "Bear Researcher"] * n + [DEBATE_DONE]
    assert _simulate_debate(n) == expected  # 2*n turns, Bull first, strict alternation


@pytest.mark.parametrize("n", [1, 3, 5])
def test_risk_speaker_order_exact(n):
    expected = (
        ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"] * n + [RISK_DONE]
    )
    assert _simulate_risk(n) == expected  # 3*n turns, fixed rotation


def test_done_sentinels_match_checkpoint_stage_names():
    assert DEBATE_DONE == "Research Manager"
    assert RISK_DONE == "Portfolio Manager"


# ---------------------------------------------------------------------------
# tools_data: toolsets + handlers (route_to_vendor monkeypatched, no network)
# ---------------------------------------------------------------------------


def test_analyst_toolsets_only_mcp_data_names():
    tools_data = _tools_data()
    toolsets = tools_data.analyst_toolsets({"bind_insider_to_news": False})

    assert set(toolsets) == {"market", "social", "news", "fundamentals"}
    assert toolsets["market"] == ["mcp__data__get_stock_data", "mcp__data__get_indicators"]
    assert toolsets["social"] == ["mcp__data__get_news"]
    assert toolsets["news"] == ["mcp__data__get_news", "mcp__data__get_global_news"]
    assert toolsets["fundamentals"] == [
        "mcp__data__get_fundamentals",
        "mcp__data__get_balance_sheet",
        "mcp__data__get_cashflow",
        "mcp__data__get_income_statement",
    ]
    for names in toolsets.values():
        assert names, "every analyst must have at least one tool"
        assert all(n.startswith("mcp__data__") for n in names)
        assert not set(names) & set(DISALLOWED_BUILTIN_TOOLS)


def test_analyst_toolsets_insider_only_when_bound():
    tools_data = _tools_data()
    insider = "mcp__data__get_insider_transactions"

    without = tools_data.analyst_toolsets({"bind_insider_to_news": False})
    assert all(insider not in names for names in without.values())

    with_flag = tools_data.analyst_toolsets({"bind_insider_to_news": True})
    assert with_flag["news"] == [
        "mcp__data__get_news", "mcp__data__get_global_news", insider,
    ]
    assert all(
        insider not in names for key, names in with_flag.items() if key != "news"
    )


async def test_handler_error_path_returns_is_error(monkeypatch):
    tools_data = _tools_data()

    def boom(method, *args, **kwargs):
        raise RuntimeError("vendor exploded")

    monkeypatch.setattr(tools_data, "route_to_vendor", boom)
    result = await tools_data.get_stock_data.handler(
        {"symbol": "NVDA", "start_date": "2026-06-01", "end_date": "2026-06-08"}
    )
    assert result["is_error"] is True
    assert "vendor exploded" in result["content"][0]["text"]
    assert result["content"][0]["type"] == "text"


async def test_handler_passthrough_routes_to_vendor(monkeypatch):
    tools_data = _tools_data()
    calls = []

    def fake(method, *args, **kwargs):
        calls.append((method,) + args)
        return "PRICE TABLE"

    monkeypatch.setattr(tools_data, "route_to_vendor", fake)
    result = await tools_data.get_stock_data.handler(
        {"symbol": "NVDA", "start_date": "2026-06-01", "end_date": "2026-06-08"}
    )
    assert result == {"content": [{"type": "text", "text": "PRICE TABLE"}]}
    assert calls == [("get_stock_data", "NVDA", "2026-06-01", "2026-06-08")]


async def test_get_indicators_comma_split(monkeypatch):
    tools_data = _tools_data()
    calls = []

    def fake(method, *args, **kwargs):
        calls.append((method,) + args)
        return f"{args[1].upper()} TABLE"

    monkeypatch.setattr(tools_data, "route_to_vendor", fake)
    result = await tools_data.get_indicators.handler(
        {"symbol": "NVDA", "indicator": "rsi, macd", "curr_date": "2026-06-01"}
    )
    assert calls == [
        ("get_indicators", "NVDA", "rsi", "2026-06-01", 30),
        ("get_indicators", "NVDA", "macd", "2026-06-01", 30),
    ]
    assert result["content"][0]["text"] == "RSI TABLE\n\nMACD TABLE"
    assert "is_error" not in result


# ---------------------------------------------------------------------------
# reflection.fetch_returns (yfinance monkeypatched)
# ---------------------------------------------------------------------------


class _FakeHistory:
    def __init__(self, rows: int):
        self._rows = rows

    def __len__(self) -> int:
        return self._rows


class _FakeYF:
    def __init__(self, rows: int):
        self._rows = rows

    def Ticker(self, symbol):
        return self

    def history(self, start=None, end=None):
        return _FakeHistory(self._rows)


def test_fetch_returns_insufficient_rows_is_none(monkeypatch):
    from tradingagents_cc import reflection

    monkeypatch.setattr(reflection, "yf", _FakeYF(rows=1))
    assert reflection.fetch_returns("NVDA", "2026-06-01") == (None, None, None)

    monkeypatch.setattr(reflection, "yf", _FakeYF(rows=0))
    assert reflection.fetch_returns("NVDA", "2026-06-01") == (None, None, None)


def test_fetch_returns_swallows_exceptions(monkeypatch):
    from tradingagents_cc import reflection

    class _RaisingYF:
        def Ticker(self, symbol):
            raise RuntimeError("network down")

    monkeypatch.setattr(reflection, "yf", _RaisingYF())
    assert reflection.fetch_returns("NVDA", "2026-06-01") == (None, None, None)


# ---------------------------------------------------------------------------
# Parent-seam canary (tradingagents.dataflows must stay LangChain-free)
# ---------------------------------------------------------------------------


_CANARY_SCRIPT = textwrap.dedent(
    """
    import sys
    import tradingagents.dataflows.interface
    import tradingagents.dataflows.config
    import tradingagents.dataflows.utils
    leaked = sorted(
        m for m in sys.modules if m.startswith(("langchain", "langgraph"))
    )
    assert not leaked, f"LangChain leaked into the parent data seam: {leaked}"
    print("SEAM_OK")
    """
)


def test_parent_seam_imports_without_langchain():
    # Fresh interpreter so this process's own imports can't mask a leak.
    proc = subprocess.run(
        [sys.executable, "-c", _CANARY_SCRIPT],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"canary failed:\n{proc.stderr}"
    assert "SEAM_OK" in proc.stdout


def test_parent_seam_signatures():
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.interface import route_to_vendor
    from tradingagents.dataflows.utils import safe_ticker_component

    params = list(inspect.signature(route_to_vendor).parameters.values())
    assert params[0].name == "method"
    assert params[1].kind is inspect.Parameter.VAR_POSITIONAL
    assert params[2].kind is inspect.Parameter.VAR_KEYWORD

    assert list(inspect.signature(set_config).parameters) == ["config"]

    stp = inspect.signature(safe_ticker_component).parameters
    assert list(stp) == ["value", "max_len"]
    assert stp["max_len"].kind is inspect.Parameter.KEYWORD_ONLY
    assert stp["max_len"].default == 32
    assert safe_ticker_component("NVDA") == "NVDA"
    with pytest.raises(ValueError):
        safe_ticker_component("../evil")


# ---------------------------------------------------------------------------
# RunCheckpointer loop-count validation (corrupted checkpoint -> fresh run)
# ---------------------------------------------------------------------------


_CP_TICKER, _CP_DATE = "SPY", "2024-07-01"


def _checkpoint_state() -> dict:
    return {
        "investment_debate_state": {"count": 2, "history": ""},
        "risk_debate_state": {"count": 3, "history": ""},
    }


def test_checkpoint_valid_loop_counts_round_trip(tmp_path):
    cp = RunCheckpointer(tmp_path / "cache")
    cp.save(_CP_TICKER, _CP_DATE, ["Market Analyst"], _checkpoint_state())
    loaded = cp.load(_CP_TICKER, _CP_DATE)
    assert loaded is not None
    assert loaded["completed"] == ["Market Analyst"]
    assert loaded["state"]["investment_debate_state"]["count"] == 2
    assert loaded["state"]["risk_debate_state"]["count"] == 3


@pytest.mark.parametrize(
    "bad_count",
    [-1, 1001, "3", 2.5, None, True],
    ids=["negative", "over-ceiling", "str", "float", "none", "bool"],
)
@pytest.mark.parametrize("key", ["investment_debate_state", "risk_debate_state"])
def test_checkpoint_corrupted_count_invalidates_whole_checkpoint(
    tmp_path, key, bad_count
):
    """A corrupted persisted count must never feed the debate loops: load()
    returns None (fresh run) for negative/huge/non-int counts in either
    debate state. bool is rejected despite being an int subclass."""
    cp = RunCheckpointer(tmp_path / "cache")
    state = _checkpoint_state()
    state[key]["count"] = bad_count
    cp.save(_CP_TICKER, _CP_DATE, [], state)
    assert cp.load(_CP_TICKER, _CP_DATE) is None


@pytest.mark.parametrize("key", ["investment_debate_state", "risk_debate_state"])
def test_checkpoint_missing_or_nondict_debate_state_invalidates(tmp_path, key):
    cp = RunCheckpointer(tmp_path / "cache")

    state = _checkpoint_state()
    del state[key]
    cp.save(_CP_TICKER, _CP_DATE, [], state)
    assert cp.load(_CP_TICKER, _CP_DATE) is None

    state = _checkpoint_state()
    state[key] = "not a dict"
    cp.save(_CP_TICKER, _CP_DATE, [], state)
    assert cp.load(_CP_TICKER, _CP_DATE) is None


# ---------------------------------------------------------------------------
# decisions.jsonl: single OS-level append + error truncation
# ---------------------------------------------------------------------------


def test_append_decision_lf_only_and_clean_lines(tmp_path):
    """Each record is one LF-terminated line (O_BINARY append suppresses the
    Windows CRT's LF->CRLF translation); successive appends stay parseable."""
    results = tmp_path / "results"
    path = append_decision(results, {"run_id": "r1", "status": "ok"})
    assert append_decision(results, {
        "run_id": "r2", "status": "failed", "error": "boom",
    }) == path

    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    assert rows[0]["error"] is None  # None passes through untouched
    assert rows[1]["error"] == "boom"  # short errors untouched
    assert all(r["schema_version"] == 1 for r in rows)


def test_append_decision_truncates_oversized_error(tmp_path):
    long_error = "x" * 9000
    path = append_decision(
        tmp_path / "results", {"status": "failed", "error": long_error},
    )
    [line] = path.read_text(encoding="utf-8").splitlines()
    error = json.loads(line)["error"]
    assert error == "x" * 4000 + "... [truncated]"
    assert len(error) == 4000 + len("... [truncated]")


def test_append_decision_error_coerced_to_str(tmp_path):
    path = append_decision(
        tmp_path / "results",
        {"status": "failed", "error": ValueError("bad stage")},
    )
    [line] = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["error"] == "bad stage"


# ---------------------------------------------------------------------------
# SdkAgentClient options builder (no query(), no network)
# ---------------------------------------------------------------------------


def _sdk_client(monkeypatch, tmp_path, **cfg_extra):
    pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc.client import SdkAgentClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test")
    cfg = {
        "data_cache_dir": str(tmp_path / "cache"),
        "quick_think_llm": "claude-sonnet-4-6",
        "deep_think_llm": "claude-opus-4-8",
        "anthropic_effort": None,
        **cfg_extra,
    }
    return SdkAgentClient(cfg)


async def test_options_builder_lockdown(monkeypatch, tmp_path):
    sdk = pytest.importorskip("claude_agent_sdk")
    client = _sdk_client(monkeypatch, tmp_path)
    server = object()
    options = client._build_options(
        system_prompt="sys", model="claude-sonnet-4-6",
        tools_server=server, allowed_tools=["mcp__data__get_news"],
        output_schema=None, max_turns=3, effort=None, deep=False,
    )

    assert options.setting_sources == []
    assert options.permission_mode == "acceptEdits"
    # Primary built-in kill switch: tools=[] disables every CLI built-in;
    # MCP servers (mounted separately via mcp_servers) are unaffected.
    assert options.tools == []
    for name in ("Bash", "WebSearch", "Write", "Task"):
        assert name in options.disallowed_tools
    assert set(DISALLOWED_BUILTIN_TOOLS) <= set(options.disallowed_tools)
    assert options.mcp_servers == {"data": server}
    assert options.allowed_tools == ["mcp__data__get_news"]
    assert options.max_turns == 3
    assert options.output_format is None
    assert options.effort is None  # quick tier never inherits anthropic_effort

    # Deny-by-default second layer: only this call's allowlist passes.
    allow = await options.can_use_tool("mcp__data__get_news", {}, None)
    deny = await options.can_use_tool("Bash", {}, None)
    assert isinstance(allow, sdk.PermissionResultAllow)
    assert isinstance(deny, sdk.PermissionResultDeny)


def test_options_builder_schema_and_deep_effort(monkeypatch, tmp_path):
    client = _sdk_client(monkeypatch, tmp_path, anthropic_effort="high")
    schema = {"type": "object", "properties": {}}
    options = client._build_options(
        system_prompt=None, model="claude-opus-4-8",
        tools_server=None, allowed_tools=None,
        output_schema=schema, max_turns=1, effort=None, deep=True,
    )
    assert options.output_format == {"type": "json_schema", "schema": schema}
    assert options.effort == "high"  # deep tier inherits config anthropic_effort
    assert options.setting_sources == []
    assert options.tools == []
    assert set(DISALLOWED_BUILTIN_TOOLS) <= set(options.disallowed_tools)


def test_options_builder_effort_gated_by_deep_flag_not_model(monkeypatch, tmp_path):
    """anthropic_effort applies only when the caller declares deep=True —
    never inferred from model-string equality, so a shared quick/deep model
    can no longer leak effort into quick-tier calls."""
    client = _sdk_client(
        monkeypatch, tmp_path,
        anthropic_effort="high",
        quick_think_llm="claude-opus-4-8",  # same model on both tiers
    )

    def build(*, effort, deep):
        return client._build_options(
            system_prompt=None, model="claude-opus-4-8",
            tools_server=None, allowed_tools=None,
            output_schema=None, max_turns=1, effort=effort, deep=deep,
        )

    assert build(effort=None, deep=False).effort is None  # no quick-tier leak
    assert build(effort=None, deep=True).effort == "high"
    assert build(effort="low", deep=True).effort == "low"  # explicit kwarg wins


def test_sdk_client_auth_guard(monkeypatch, tmp_path):
    pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc.client import AuthError, SdkAgentClient

    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AuthError, match="claude setup-token"):
        SdkAgentClient({"data_cache_dir": str(tmp_path / "cache")})

    # Metered key is evicted from the process so the SDK can never bill it.
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-metered")
    SdkAgentClient({"data_cache_dir": str(tmp_path / "cache")})
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_sdk_client_evicts_every_metered_routing_env_var(monkeypatch, tmp_path):
    """All five billing/routing variables are popped, not just the API key —
    any one of them could route the spawned CLI off subscription auth."""
    pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc.client import _METERED_AUTH_ENV_VARS, SdkAgentClient

    expected = {
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_BASE_URL",
    }
    assert set(_METERED_AUTH_ENV_VARS) == expected

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test")
    for var in _METERED_AUTH_ENV_VARS:
        monkeypatch.setenv(var, "metered-or-routed")
    SdkAgentClient({"data_cache_dir": str(tmp_path / "cache")})
    for var in _METERED_AUTH_ENV_VARS:
        assert var not in os.environ, f"{var} survived eviction"
    assert os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "test"  # subscription auth kept


# ---------------------------------------------------------------------------
# SdkAgentClient.run() against a faked sdk.query (no subprocess, no network)
# ---------------------------------------------------------------------------


def _result_message(sdk, **overrides):
    base = dict(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
        num_turns=1, session_id="test", result="ok",
        usage={"input_tokens": 3, "output_tokens": 5},
    )
    base.update(overrides)
    return sdk.ResultMessage(**base)


def _install_fake_query(monkeypatch, sdk, results):
    """Patch sdk.query with a fake that replays canned ResultMessages.

    Replicates the SDK's pre-spawn validation
    (claude_agent_sdk._internal.client): a ``str`` prompt combined with a
    ``can_use_tool`` callback is hard-rejected — the exact combination that
    used to kill every real call. Returns a dict recording call count plus
    the prompts/options seen.
    """
    calls = {"count": 0, "prompts": [], "options": []}

    async def fake_query(*, prompt, options=None, transport=None):
        if options is not None and options.can_use_tool and isinstance(prompt, str):
            raise ValueError(
                "can_use_tool callback requires streaming mode. "
                "Please provide prompt as an AsyncIterable instead of a string."
            )
        messages = prompt if isinstance(prompt, str) else [m async for m in prompt]
        calls["count"] += 1
        calls["prompts"].append(messages)
        calls["options"].append(options)
        yield results[min(calls["count"] - 1, len(results) - 1)]

    monkeypatch.setattr(sdk, "query", fake_query)
    return calls


async def test_run_streams_prompt_for_can_use_tool(monkeypatch, tmp_path):
    """Every options object carries can_use_tool, so the prompt must be sent
    in the SDK's streaming-mode shape (a str prompt raises pre-spawn)."""
    sdk = pytest.importorskip("claude_agent_sdk")
    client = _sdk_client(monkeypatch, tmp_path)
    calls = _install_fake_query(monkeypatch, sdk, [_result_message(sdk)])

    result = await client.run("trader", "hello there", model="claude-sonnet-4-6")

    assert result.text == "ok"
    assert result.usage == {
        "llm_calls": 0, "tool_calls": 0, "tokens_in": 3, "tokens_out": 5,
    }
    assert calls["count"] == 1
    assert calls["options"][0].can_use_tool is not None  # the rejecting combo
    [messages] = calls["prompts"]
    assert messages == [
        {"type": "user", "message": {"role": "user", "content": "hello there"}}
    ]


async def test_run_retries_rate_limit_shaped_error_results(monkeypatch, tmp_path):
    """DESIGN.md retry contract: 429-shaped error results get the backoff
    retries instead of failing the stage on attempt 1."""
    sdk = pytest.importorskip("claude_agent_sdk")
    client = _sdk_client(
        monkeypatch, tmp_path, retry_attempts=3, retry_base_delay=0.0,
    )
    rate_limited = _result_message(
        sdk, is_error=True,
        result="API error: 429 rate_limit_error", api_error_status=429,
    )
    calls = _install_fake_query(
        monkeypatch, sdk, [rate_limited, rate_limited, _result_message(sdk)]
    )

    result = await client.run("trader", "hi", model="claude-sonnet-4-6")

    assert result.text == "ok"
    assert calls["count"] == 3  # two 429s retried, third attempt succeeded


async def test_run_rate_limit_exhaustion_becomes_stage_error(monkeypatch, tmp_path):
    """A persistent rate limit still fails the stage — but only after all
    attempts, and as StageError (the internal retryable never escapes)."""
    sdk = pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc.client import StageError

    client = _sdk_client(
        monkeypatch, tmp_path, retry_attempts=3, retry_base_delay=0.0,
    )
    overloaded = _result_message(
        sdk, is_error=True, result=None, errors=["Overloaded"], api_error_status=529,
    )
    calls = _install_fake_query(monkeypatch, sdk, [overloaded])

    with pytest.raises(StageError, match="after 3 attempts"):
        await client.run("trader", "hi", model="claude-sonnet-4-6")
    assert calls["count"] == 3


async def test_run_non_rate_limit_error_fails_immediately(monkeypatch, tmp_path):
    """Non-429-shaped error results stay a first-attempt StageError."""
    sdk = pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc.client import StageError

    client = _sdk_client(
        monkeypatch, tmp_path, retry_attempts=3, retry_base_delay=0.0,
    )
    bad = _result_message(
        sdk, is_error=True, subtype="error_during_execution",
        result="invalid request",
    )
    calls = _install_fake_query(monkeypatch, sdk, [bad])

    with pytest.raises(StageError, match="error_during_execution"):
        await client.run("trader", "hi", model="claude-sonnet-4-6")
    assert calls["count"] == 1


@pytest.mark.live
async def test_live_sdk_round_trip(tmp_path):
    """One cheap real query() round-trip so the SDK seam is exercised
    end-to-end. Auto-skipped unless TRADINGAGENTS_CC_LIVE=1 (and a real
    CLAUDE_CODE_OAUTH_TOKEN); consumes subscription credit."""
    pytest.importorskip("claude_agent_sdk")
    from tradingagents_cc.client import SdkAgentClient

    client = SdkAgentClient({"data_cache_dir": str(tmp_path / "cache")})
    result = await client.run(
        "trader", "Reply with exactly one word: pong",
        model="claude-sonnet-4-6", max_turns=1,
    )
    assert result.text.strip()
