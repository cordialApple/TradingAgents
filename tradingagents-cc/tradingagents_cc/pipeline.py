# Ported from tradingagents/graph/{trading_graph,setup,propagation}.py — the LangGraph
# StateGraph becomes this plain asyncio driver: one client.run() per pipeline node.
"""The orchestrator: TradingAgentsGraph + GraphSetup + Propagator as one asyncio driver.

``TradingAgentsPipeline.propagate(ticker, trade_date)`` runs the parent graph's
exact node order with Python owning 100% of the control flow:

    Phase-B memory resolution (skipped in mock mode)
    -> initial state with injected past_context
    -> analysts in ``selected_analysts`` order (tool-enabled, sequential)
    -> Bull-first investment debate (``2 * max_debate_rounds`` turns)
    -> Research Manager (deep) -> Trader
    -> Aggressive-first risk debate (``3 * max_risk_discuss_rounds`` turns)
    -> Portfolio Manager (deep) -> final report-section pass -> ``parse_rating`` (no LLM)
    -> artifacts (complete report, states log, decisions.jsonl)
    -> Phase-A pending memory entry -> checkpoint cleared.

Every completed stage is checkpointed under the parent-identical thread id, and
the debate loops checkpoint after every turn — re-running the same ticker+date
skips completed stages, and a mid-debate crash re-enters the loop with the
verbatim conditional logic picking the next speaker from the persisted counts.
A node that fails permanently raises ``StageError``; it — and any other
exception escaping mid-run — propagates only after the checkpoint and a
status="failed" decisions.jsonl row (carrying the run's LLM/tool stats) are
written, so spent credit stays visible and the next scheduled run resumes
where this one stopped.

Mock mode (``llm_backend == "mock"``) drives the identical control flow with no
MCP data server (analysts answer tool-less) and never imports
``claude_agent_sdk`` anywhere on the path.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from tradingagents.dataflows.config import set_config

from .checkpointer import RunCheckpointer, thread_id
from .client import AgentClient, AgentResult, StageError
from .conditional_logic import (
    DEBATE_DONE,
    RISK_DONE,
    should_continue_debate,
    should_continue_risk_analysis,
)
from .default_config import to_parent_config
from .memory import TradingMemoryLog
from .nodes import (
    run_analyst,
    run_bear,
    run_bull,
    run_portfolio_manager,
    run_research_manager,
    run_risk_debator,
    run_trader,
)
from .rating import parse_rating
from .reflection import resolve_pending_entries
from .reporting import REPORT_SECTIONS, SECTION_HEADERS, RunReporter, append_decision
from .state import create_initial_state, normalize_ticker, safe_ticker_component

logger = logging.getLogger(__name__)

__all__ = ["TradingAgentsPipeline"]

# Analyst kind -> checkpoint stage name (exact strings from checkpointer.STAGES).
_ANALYST_STAGES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

# conditional_logic speaker string -> node role (doubles as SECTION_HEADERS key).
_RISK_ROLES = {
    "Aggressive Analyst": "aggressive",
    "Conservative Analyst": "conservative",
    "Neutral Analyst": "neutral",
}

# The trader renderer's guaranteed trailing marker; the free-text fallback may
# vary case, so the match is tolerant and the LAST occurrence wins.
_TRADER_ACTION_RE = re.compile(
    r"FINAL TRANSACTION PROPOSAL:\s*\*\*([A-Za-z]+)\*\*", re.IGNORECASE
)


def _trader_action(trader_plan: str) -> str | None:
    """Extract Buy/Hold/Sell from the trader's marker line, or None."""
    matches = _TRADER_ACTION_RE.findall(trader_plan or "")
    return matches[-1].capitalize() if matches else None


@contextmanager
def _timed(timings: dict[str, float], stage: str):
    """Record a stage's wall-clock seconds (kept even when the stage raises)."""
    start = time.monotonic()
    try:
        yield
    finally:
        timings[stage] = round(time.monotonic() - start, 3)


class _UsageRecordingClient:
    """Delegating AgentClient proxy that sums ``AgentResult.usage`` for one run.

    The pipeline wraps the injected client once per propagate() call so the
    decisions.jsonl stats row aggregates every LLM call the run makes
    (including Phase-B reflector calls) without changing any node signature.
    Passthrough is deliberately ``**kwargs``: nodes feature-detect the optional
    SDK-only ``effort`` parameter via ``inspect.signature``, and hiding it here
    is behavior-preserving because ``SdkAgentClient`` self-applies the config
    ``anthropic_effort`` knob to deep-tier calls.
    """

    def __init__(self, inner: AgentClient) -> None:
        self._inner = inner
        self.stats = {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0}

    async def run(self, role: str, prompt: str, **kwargs: Any) -> AgentResult:
        result = await self._inner.run(role, prompt, **kwargs)
        usage = result.usage if isinstance(result.usage, dict) else {}
        for key in self.stats:
            value = usage.get(key)
            if isinstance(value, int):
                self.stats[key] += value
        return result


class TradingAgentsPipeline:
    """Drives one full analysis per propagate() call; reusable across runs."""

    def __init__(self, config: dict[str, Any], client: AgentClient):
        if not config.get("selected_analysts"):
            # Parent GraphSetup parity: at least one analyst is required.
            raise ValueError("config['selected_analysts'] must name at least one analyst")
        self.config = config
        self.client = client
        self.memory_log = TradingMemoryLog(config)
        self.checkpointer = RunCheckpointer(
            config["data_cache_dir"],
            enabled=bool(config.get("checkpoint_enabled", True)),
        )
        Path(config["data_cache_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)

        self._mock = config.get("llm_backend", "sdk") == "mock"
        if self._mock:
            # Mock mode never imports claude_agent_sdk: no MCP server is built
            # and the analysts run tool-less on canned outputs.
            self.tools_server: object | None = None
        else:
            from .tools_data import build_data_server  # lazy: imports the SDK

            self.tools_server = build_data_server()

    async def propagate(self, ticker: str, trade_date: str) -> tuple[dict, str]:
        """Run the pipeline for one (ticker, trade_date).

        Returns ``(final_state, signal)`` — signal is one of Buy / Overweight /
        Hold / Underweight / Sell. Raises ``StageError`` when a node fails
        permanently and ``ValueError`` on a malformed ticker or date; any
        exception escaping mid-run (StageError or otherwise) propagates only
        after progress is checkpointed and a failed decisions.jsonl row is
        written.
        """
        cfg = self.config
        ticker = normalize_ticker(ticker)
        safe_ticker_component(ticker)  # reject path-escaping tickers up front
        trade_date = str(trade_date)

        # The parent dataflows layer routes vendors off its module-global
        # config. set_config is free, so it is applied in every mode and every
        # run (tools_data.apply_data_config does exactly this call, but
        # importing tools_data would pull claude_agent_sdk into mock runs).
        set_config(to_parent_config(cfg))

        reporter = RunReporter(cfg["results_dir"], ticker, trade_date)  # validates the date
        client = _UsageRecordingClient(self.client)

        # Phase B: resolve pending memory entries for this ticker. Needs live
        # yfinance prices and a real reflector call, so mock mode skips it.
        if not self._mock:
            await resolve_pending_entries(self.memory_log, ticker, client, cfg)

        checkpoint = self.checkpointer.load(ticker, trade_date)
        if checkpoint is not None:
            state: dict[str, Any] = checkpoint["state"]
            completed: list[str] = list(checkpoint["completed"])
            logger.info(
                "Resuming %s on %s; completed stages: %s",
                ticker, trade_date, ", ".join(completed) or "(none)",
            )
        else:
            past_context = self.memory_log.get_past_context(ticker)
            state = dict(create_initial_state(ticker, trade_date, past_context))
            completed = []
            logger.info("Starting fresh for %s on %s", ticker, trade_date)

        stage_timings: dict[str, float] = {}
        fallbacks_used: list[str] = []
        started = time.monotonic()

        def save() -> None:
            self.checkpointer.save(ticker, trade_date, completed, state)

        async def run_decision_stage(stage: str, node: Callable[..., Any]) -> None:
            """One structured stage; pops the node's fallback_used flag."""
            update = await node(state, client, cfg)
            if update.pop("fallback_used", False):
                fallbacks_used.append(stage)
            state.update(update)

        try:
            # --- Analyst team: one tool-enabled agentic query each ---
            for kind in cfg["selected_analysts"]:
                if kind not in _ANALYST_STAGES:
                    raise ValueError(
                        f"unknown analyst {kind!r} in selected_analysts; "
                        f"expected one of {tuple(_ANALYST_STAGES)}"
                    )
                stage = _ANALYST_STAGES[kind]
                if stage in completed:
                    continue
                with _timed(stage_timings, stage):
                    update = await run_analyst(
                        state, kind, client, cfg,
                        tools_server=self.tools_server, reporter=reporter,
                    )
                    state.update(update)
                    for section, text in update.items():
                        reporter.write_section(section, text)
                completed.append(stage)
                save()

            # --- Investment debate: Bull first, 2*max_debate_rounds turns ---
            stage = "Investment Debate"
            if stage not in completed:
                with _timed(stage_timings, stage):
                    # Hard bound: 2*N speaking turns plus one closing DONE
                    # check. A healthy run always breaks via DEBATE_DONE
                    # inside the bound; exhausting it means the debate count
                    # is corrupted, and failing fast caps the paid LLM turns.
                    debate_turns = 2 * max(cfg["max_debate_rounds"], 0) + 1
                    for _ in range(debate_turns):
                        speaker = should_continue_debate(state, cfg["max_debate_rounds"])
                        if speaker == DEBATE_DONE:
                            break
                        if speaker == "Bull Researcher":
                            state.update(await run_bull(state, client, cfg))
                            side = "bull"
                        else:
                            state.update(await run_bear(state, client, cfg))
                            side = "bear"
                        # Tee body is the stripped history, exactly as the
                        # parent CLI strips it — string-identical artifacts.
                        history = state["investment_debate_state"][f"{side}_history"]
                        reporter.write_section(
                            "investment_plan",
                            f"{SECTION_HEADERS[side]}\n{history.strip()}",
                        )
                        # Turn-level checkpoint: counts persist, but the stage
                        # joins `completed` only when the loop exits.
                        save()
                    else:
                        raise StageError(
                            f"investment debate exceeded {debate_turns} turns "
                            "without finishing; debate state is corrupted"
                        )
                completed.append(stage)
                save()

            # --- Research Manager (deep tier, structured) ---
            stage = "Research Manager"
            if stage not in completed:
                with _timed(stage_timings, stage):
                    await run_decision_stage(stage, run_research_manager)
                    reporter.write_section(
                        "investment_plan",
                        f"{SECTION_HEADERS['research_manager']}\n"
                        f"{state['investment_plan']}",
                    )
                completed.append(stage)
                save()

            # --- Trader (quick tier, structured) ---
            stage = "Trader"
            if stage not in completed:
                with _timed(stage_timings, stage):
                    await run_decision_stage(stage, run_trader)
                    reporter.write_section(
                        "trader_investment_plan", state["trader_investment_plan"]
                    )
                completed.append(stage)
                save()

            # --- Risk debate: Aggressive first, 3*max_risk_discuss_rounds turns ---
            stage = "Risk Debate"
            if stage not in completed:
                with _timed(stage_timings, stage):
                    # Same hard bound as the investment debate: 3*N turns plus
                    # one closing DONE check; exhaustion means corruption.
                    risk_turns = 3 * max(cfg["max_risk_discuss_rounds"], 0) + 1
                    for _ in range(risk_turns):
                        speaker = should_continue_risk_analysis(
                            state, cfg["max_risk_discuss_rounds"]
                        )
                        if speaker == RISK_DONE:
                            break
                        role = _RISK_ROLES[speaker]
                        state.update(await run_risk_debator(state, role, client, cfg))
                        # Stripped history: parent CLI tee parity.
                        history = state["risk_debate_state"][f"{role}_history"]
                        reporter.write_section(
                            "final_trade_decision",
                            f"{SECTION_HEADERS[role]}\n{history.strip()}",
                        )
                        save()
                    else:
                        raise StageError(
                            f"risk debate exceeded {risk_turns} turns "
                            "without finishing; risk state is corrupted"
                        )
                completed.append(stage)
                save()

            # --- Portfolio Manager (deep tier, structured) ---
            stage = "Portfolio Manager"
            if stage not in completed:
                with _timed(stage_timings, stage):
                    await run_decision_stage(stage, run_portfolio_manager)
                    reporter.write_section(
                        "final_trade_decision",
                        f"{SECTION_HEADERS['portfolio_manager']}\n"
                        f"{state['final_trade_decision']}",
                    )
                completed.append(stage)
                save()

            # --- Final report-section pass (parent CLI end-of-run refresh) ---
            # The parent overwrites every report section with the raw
            # final-state value (cli/main.py run end), so the composite
            # sections (investment_plan / final_trade_decision) end headerless.
            # Like the parent's tee decorator, empty sections are not written.
            for section in REPORT_SECTIONS:
                if state.get(section):
                    reporter.write_section(section, state[section])
        except Exception as exc:
            # StageError (a node failed permanently) and any other exception
            # escaping mid-run get identical bookkeeping: progress survives
            # via the checkpoint and the spent credit lands in a
            # status="failed" decisions.jsonl row before the error propagates.
            logger.error("Stage failed for %s on %s: %s", ticker, trade_date, exc)
            save()  # progress survives; the next run resumes from here
            append_decision(
                cfg["results_dir"],
                self._decision_row(
                    ticker=ticker, trade_date=trade_date, state=state,
                    reporter=reporter, stats=client.stats,
                    stage_timings=stage_timings, fallbacks_used=fallbacks_used,
                    started=started, status="failed", error=str(exc),
                ),
            )
            raise

        # --- Signal + per-run artifacts (parent _log_state and CLI save_report) ---
        signal = parse_rating(state["final_trade_decision"])
        reporter.write_complete_report(state)
        reporter.write_states_log(state)
        append_decision(
            cfg["results_dir"],
            self._decision_row(
                ticker=ticker, trade_date=trade_date, state=state,
                reporter=reporter, stats=client.stats,
                stage_timings=stage_timings, fallbacks_used=fallbacks_used,
                started=started, status="ok", signal=signal,
            ),
        )

        # Phase A: append the pending memory entry (idempotent, no LLM call),
        # then drop the checkpoint so a re-run starts fresh.
        self.memory_log.store_decision(
            ticker=ticker, trade_date=trade_date,
            final_trade_decision=state["final_trade_decision"],
        )
        self.checkpointer.clear(ticker, trade_date)
        logger.info("Run complete for %s on %s: %s", ticker, trade_date, signal)
        return state, signal

    def _decision_row(
        self, *,
        ticker: str,
        trade_date: str,
        state: dict[str, Any],
        reporter: RunReporter,
        stats: dict[str, int],
        stage_timings: dict[str, float],
        fallbacks_used: list[str],
        started: float,
        status: str,
        signal: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """One decisions.jsonl row; append_decision normalizes/timestamps it."""
        return {
            "run_id": thread_id(ticker, trade_date),
            "ticker": ticker,
            "trade_date": trade_date,
            "decision": signal,
            "rating": signal,
            "trader_action": _trader_action(state.get("trader_investment_plan", "")),
            "status": status,
            "error": error,
            "models": {
                "quick": self.config["quick_think_llm"],
                "deep": self.config["deep_think_llm"],
            },
            "stats": dict(stats),
            "stage_timings": dict(stage_timings),
            "fallbacks_used": list(fallbacks_used),
            "report_dir": reporter.run_dir,
            "duration_s": round(time.monotonic() - started, 3),
        }
