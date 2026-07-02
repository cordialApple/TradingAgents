# Ported from tradingagents/agents/* node closures (analysts, researchers, managers,
# trader, risk_mgmt) — LangGraph nodes become plain async functions over the AgentClient seam.
"""One async function per pipeline node: the bridge between state dict and client.run().

Each function takes the frozen ``AgentState`` dict plus the ``AgentClient`` and
config, performs exactly the LLM calls its parent LangGraph node performed, and
returns the same partial state-update dict the parent node returned (the
pipeline merges it).  Parent semantics kept verbatim:

- Analysts: ONE tool-enabled ``client.run()`` whose agentic loop replaces the
  parent's should_continue/ToolNode cycle; the free-text result is the report.
- Bull/Bear and the three risk debaters: one no-tool quick call on the verbatim
  prompt; the ``"<Role> Analyst: "`` prefix, history appends, and ``count += 1``
  are applied HERE, exactly as the parent closures did.
- Research Manager / Trader / Portfolio Manager: structured attempt
  (``output_schema``), then on a missing/invalid structured payload exactly ONE
  free-text retry with the same prompt (parent ``invoke_structured_or_freetext``
  parity).  Their returns carry a ``fallback_used`` bool — pipeline metadata,
  popped before the rest is merged into state.

Model tiers: deep (``deep_think_llm`` + optional ``anthropic_effort``) for the
Research Manager and Portfolio Manager only; quick for everything else.

This module never imports ``claude_agent_sdk``: the pipeline passes the mounted
``tools_server`` in, and ``tools_data`` (which does import the SDK) is imported
lazily only when a server is actually provided — mock-mode runs stay SDK-free.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from .prompts import (
    TRADER_SYSTEM_PROMPT,
    build_aggressive_prompt,
    build_analyst_system_prompt,
    build_bear_prompt,
    build_bull_prompt,
    build_conservative_prompt,
    build_instrument_context,
    build_neutral_prompt,
    build_portfolio_manager_prompt,
    build_research_manager_prompt,
    build_trader_user_prompt,
)
from .schemas import (
    PortfolioDecision,
    ResearchPlan,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_trader_proposal,
)

if TYPE_CHECKING:  # type-only: keeps runtime imports minimal
    from .client import AgentClient
    from .reporting import RunReporter

logger = logging.getLogger(__name__)

# Turn budget for the no-tool, single-response calls (debaters, risk debaters,
# structured decision stages). Logically these need one turn, but the bundled
# CLI's turn accounting can consume extra turns on a single response (the
# structured-output flow, thinking blocks) — with max_turns=1 the CLI dies with
# "Reached maximum number of turns (1)" (deterministic on the deep tier,
# intermittent on quick; observed 2026-07-01). Small headroom, no cost risk:
# these calls mount no tools, so the model cannot loop — it stops when the
# response (or structured payload) is complete.
SINGLE_SHOT_MAX_TURNS = 4

__all__ = [
    "run_analyst",
    "run_bull",
    "run_bear",
    "run_research_manager",
    "run_trader",
    "run_portfolio_manager",
    "run_risk_debator",
]


# ---------------------------------------------------------------------------
# Analysts
# ---------------------------------------------------------------------------

# kind -> AgentState report field (parent report keys, verbatim).
_REPORT_KEYS = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

# Bare tool names per analyst for the collaboration wrapper's {tool_names} —
# the parent comma-joins tool.name, so the prompt shows unqualified names.
# Mock-mode mirror of tools_data.analyst_toolsets (which cannot be imported
# without claude_agent_sdk); in SDK mode the names are derived from the real
# allowlist instead, so the two can never drift where it matters.
_ANALYST_TOOL_NAMES = {
    "market": ("get_stock_data", "get_indicators"),
    "social": ("get_news",),
    "news": ("get_news", "get_global_news"),
    "fundamentals": (
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    ),
}


async def run_analyst(
    state: dict[str, Any],
    kind: str,
    client: "AgentClient",
    cfg: dict[str, Any],
    tools_server: object | None = None,
    reporter: "RunReporter | None" = None,
) -> dict[str, Any]:
    """One tool-enabled query for one analyst; returns ``{<x>_report: text}``.

    The SDK agentic loop (budgeted by ``max_analyst_turns``) replaces the
    parent's tool loop; a fresh query per analyst replaces ``create_msg_delete``.
    ``tools_server`` is the pipeline-built MCP server config (``None`` in mock
    mode). When a reporter is given, the stage's tool-call lines are teed to
    ``message_tool.log``.
    """
    if kind not in _REPORT_KEYS:
        raise ValueError(f"unknown analyst kind {kind!r}; expected one of {tuple(_REPORT_KEYS)}")

    ticker = state["company_of_interest"]

    allowed_tools: list[str] | None = None
    if tools_server is not None:
        # Lazy: tools_data imports claude_agent_sdk; only reachable in SDK mode.
        from .tools_data import analyst_toolsets

        allowed_tools = analyst_toolsets(cfg)[kind]
        tool_names = [name.removeprefix("mcp__data__") for name in allowed_tools]
    else:
        tool_names = list(_ANALYST_TOOL_NAMES[kind])
        if kind == "news" and cfg.get("bind_insider_to_news"):
            tool_names.append("get_insider_transactions")

    system_prompt = build_analyst_system_prompt(
        kind,
        tool_names,
        state["trade_date"],
        ticker,
        cfg.get("output_language", "English"),
    )

    result = await client.run(
        f"{kind}_analyst",
        # Kickoff user message: the parent seeds messages with the company
        # name; the vendored instrument-context sentence carries the same
        # information plus the exact-ticker instruction.
        build_instrument_context(ticker),
        system_prompt=system_prompt,
        model=cfg["quick_think_llm"],
        tools_server=tools_server,
        allowed_tools=allowed_tools,
        max_turns=cfg["max_analyst_turns"],
    )
    if reporter is not None:
        reporter.write_tool_log(result.tool_call_log)
    return {_REPORT_KEYS[kind]: result.text}


# ---------------------------------------------------------------------------
# Bull / Bear researcher debate
# ---------------------------------------------------------------------------


async def _run_debater(
    state: dict[str, Any],
    client: "AgentClient",
    cfg: dict[str, Any],
    *,
    role: str,
    label: str,
    own_history_key: str,
    other_history_key: str,
    prompt_builder: Callable[..., str],
) -> dict[str, Any]:
    """Shared bull/bear turn: verbatim prompt, prefix applied here, count += 1."""
    debate = state["investment_debate_state"]
    history = debate.get("history", "")
    prompt = prompt_builder(
        state.get("market_report", ""),
        state.get("sentiment_report", ""),
        state.get("news_report", ""),
        state.get("fundamentals_report", ""),
        history,
        debate.get("current_response", ""),
    )

    result = await client.run(
        role, prompt, model=cfg["quick_think_llm"], max_turns=SINGLE_SHOT_MAX_TURNS
    )
    argument = f"{label}: {result.text}"

    new_debate = {
        "history": history + "\n" + argument,
        own_history_key: debate.get(own_history_key, "") + "\n" + argument,
        other_history_key: debate.get(other_history_key, ""),
        "current_response": argument,
        "judge_decision": debate.get("judge_decision", ""),
        "count": debate["count"] + 1,
    }
    return {"investment_debate_state": new_debate}


async def run_bull(
    state: dict[str, Any], client: "AgentClient", cfg: dict[str, Any]
) -> dict[str, Any]:
    """Bull turn: ``current_response`` going in is the last bear argument."""
    return await _run_debater(
        state, client, cfg,
        role="bull", label="Bull Analyst",
        own_history_key="bull_history", other_history_key="bear_history",
        prompt_builder=build_bull_prompt,
    )


async def run_bear(
    state: dict[str, Any], client: "AgentClient", cfg: dict[str, Any]
) -> dict[str, Any]:
    """Bear turn: ``current_response`` going in is the last bull argument."""
    return await _run_debater(
        state, client, cfg,
        role="bear", label="Bear Analyst",
        own_history_key="bear_history", other_history_key="bull_history",
        prompt_builder=build_bear_prompt,
    )


# ---------------------------------------------------------------------------
# Structured decision stages (invoke_structured_or_freetext parity)
# ---------------------------------------------------------------------------


def _run_accepts(client: "AgentClient", name: str) -> bool:
    """True when ``client.run()`` accepts the keyword argument ``name``.

    Feature detection for the optional SDK-only ``run()`` extensions (``deep``,
    ``effort``): the frozen AgentClient protocol carries neither, so they are
    forwarded only to clients that declare them — mock/dry-run clients without
    the parameter keep working. Delegating proxies whose ``run`` passes
    ``**kwargs`` through untouched (e.g. pipeline._UsageRecordingClient, which
    exposes the wrapped client as ``_inner``) are followed down the chain so
    the answer reflects the client that ultimately executes the call — a bare
    ``**kwargs`` never counts as acceptance on its own, since the proxy would
    just relay the kwarg to an inner client that may reject it.
    """
    target: Any = client
    seen: set[int] = set()
    while target is not None and id(target) not in seen:
        seen.add(id(target))
        run = getattr(target, "run", None)
        if run is None:
            return False
        try:
            params = inspect.signature(run).parameters
        except (TypeError, ValueError):  # builtins / odd callables: no intel
            return False
        if name in params:
            return True
        if not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            return False
        target = getattr(target, "_inner", None)
    return False


async def _structured_or_freetext(
    role: str,
    client: "AgentClient",
    cfg: dict[str, Any],
    *,
    prompt: str,
    model_cls: type[BaseModel],
    render: Callable[[Any], str],
    system_prompt: str | None = None,
    deep: bool = False,
) -> tuple[str, bool]:
    """Structured attempt, then exactly ONE free-text retry with the same prompt.

    Returns ``(markdown, fallback_used)``. The retry fires when the SDK
    exhausted its structured-output retries (``structured=None``) or when the
    payload fails Pydantic validation — the parent fell back on ANY exception,
    so rendering/validation failures are treated identically. Transport-level
    ``StageError`` propagates: that is a stage failure, not a fallback.
    """
    model = cfg["deep_think_llm"] if deep else cfg["quick_think_llm"]
    extra: dict[str, Any] = {}
    # SDK-only run() extensions — the frozen AgentClient protocol has neither
    # parameter, so each is forwarded via _run_accepts feature detection and
    # mock mode must never crash:
    # - deep: declares the tier explicitly so SdkAgentClient applies the config
    #   anthropic_effort knob to deep-tier calls only (the tier is never
    #   inferred from the model string, which both tiers may share).
    # - effort: explicit value, wins over the client's deep-gated config knob.
    if deep and _run_accepts(client, "deep"):
        extra["deep"] = True
    if deep and cfg.get("anthropic_effort") and _run_accepts(client, "effort"):
        extra["effort"] = cfg["anthropic_effort"]

    result = await client.run(
        role, prompt,
        system_prompt=system_prompt,
        model=model,
        output_schema=model_cls.model_json_schema(),
        max_turns=SINGLE_SHOT_MAX_TURNS,
        **extra,
    )
    if result.structured is not None:
        try:
            return render(model_cls.model_validate(result.structured)), False
        except Exception as exc:  # parity: any validation/render failure -> free text
            logger.warning(
                "[%s] structured payload failed validation (%s: %s); "
                "falling back to free text",
                role, type(exc).__name__, exc,
            )
    else:
        logger.warning("[%s] no structured output; falling back to free text", role)

    retry = await client.run(
        role, prompt,
        system_prompt=system_prompt,
        model=model,
        max_turns=SINGLE_SHOT_MAX_TURNS,
        **extra,
    )
    return retry.text, True


async def run_research_manager(
    state: dict[str, Any], client: "AgentClient", cfg: dict[str, Any]
) -> dict[str, Any]:
    """Judge the bull/bear debate into an investment plan (deep tier, structured)."""
    debate = state["investment_debate_state"]
    prompt = build_research_manager_prompt(
        state["company_of_interest"], debate.get("history", "")
    )
    plan, fallback_used = await _structured_or_freetext(
        "research_manager", client, cfg,
        prompt=prompt, model_cls=ResearchPlan, render=render_research_plan,
        deep=True,
    )

    new_debate = {
        "judge_decision": plan,
        "history": debate.get("history", ""),
        "bear_history": debate.get("bear_history", ""),
        "bull_history": debate.get("bull_history", ""),
        "current_response": plan,
        "count": debate.get("count", 0),
    }
    return {
        "investment_debate_state": new_debate,
        "investment_plan": plan,
        "fallback_used": fallback_used,
    }


async def run_trader(
    state: dict[str, Any], client: "AgentClient", cfg: dict[str, Any]
) -> dict[str, Any]:
    """Turn the investment plan into a transaction proposal (quick tier, structured).

    The rendered markdown always ends ``FINAL TRANSACTION PROPOSAL: **{ACTION}**``
    on the structured path (render_trader_proposal guarantee).
    """
    prompt = build_trader_user_prompt(
        state["company_of_interest"], state["investment_plan"]
    )
    plan, fallback_used = await _structured_or_freetext(
        "trader", client, cfg,
        prompt=prompt, model_cls=TraderProposal, render=render_trader_proposal,
        system_prompt=TRADER_SYSTEM_PROMPT,
    )
    return {
        "trader_investment_plan": plan,
        "sender": "Trader",
        "fallback_used": fallback_used,
    }


async def run_portfolio_manager(
    state: dict[str, Any], client: "AgentClient", cfg: dict[str, Any]
) -> dict[str, Any]:
    """Synthesize the risk debate into the final decision (deep tier, structured)."""
    risk = state["risk_debate_state"]
    prompt = build_portfolio_manager_prompt(
        state["company_of_interest"],
        state["investment_plan"],
        state["trader_investment_plan"],
        risk.get("history", ""),
        past_context=state.get("past_context", ""),
        language=cfg.get("output_language", "English"),
    )
    decision, fallback_used = await _structured_or_freetext(
        "portfolio_manager", client, cfg,
        prompt=prompt, model_cls=PortfolioDecision, render=render_pm_decision,
        deep=True,
    )

    new_risk = {
        "judge_decision": decision,
        "history": risk.get("history", ""),
        "aggressive_history": risk.get("aggressive_history", ""),
        "conservative_history": risk.get("conservative_history", ""),
        "neutral_history": risk.get("neutral_history", ""),
        "latest_speaker": "Judge",
        "current_aggressive_response": risk.get("current_aggressive_response", ""),
        "current_conservative_response": risk.get("current_conservative_response", ""),
        "current_neutral_response": risk.get("current_neutral_response", ""),
        "count": risk.get("count", 0),
    }
    return {
        "risk_debate_state": new_risk,
        "final_trade_decision": decision,
        "fallback_used": fallback_used,
    }


# ---------------------------------------------------------------------------
# Risk debate (Aggressive / Conservative / Neutral)
# ---------------------------------------------------------------------------

# role -> (prompt builder, the OTHER two analysts' current-response keys in the
# builder's argument order, exact latest_speaker string).
_RISK_DEBATORS: dict[str, tuple[Callable[..., str], tuple[str, str], str]] = {
    "aggressive": (
        build_aggressive_prompt,
        ("current_conservative_response", "current_neutral_response"),
        "Aggressive",
    ),
    "conservative": (
        build_conservative_prompt,
        ("current_aggressive_response", "current_neutral_response"),
        "Conservative",
    ),
    "neutral": (
        build_neutral_prompt,
        ("current_aggressive_response", "current_conservative_response"),
        "Neutral",
    ),
}


async def run_risk_debator(
    state: dict[str, Any], role: str, client: "AgentClient", cfg: dict[str, Any]
) -> dict[str, Any]:
    """One risk-debate turn for ``role`` in aggressive/conservative/neutral.

    Sets ``latest_speaker`` to exactly "Aggressive"/"Conservative"/"Neutral"
    (the rotation predicate keys on that prefix), ``current_<role>_response``
    to the prefixed argument, appends to history, count += 1.
    """
    if role not in _RISK_DEBATORS:
        raise ValueError(
            f"unknown risk debator role {role!r}; expected one of {tuple(_RISK_DEBATORS)}"
        )
    prompt_builder, other_response_keys, speaker = _RISK_DEBATORS[role]

    risk = state["risk_debate_state"]
    history = risk.get("history", "")
    prompt = prompt_builder(
        state["trader_investment_plan"],
        state.get("market_report", ""),
        state.get("sentiment_report", ""),
        state.get("news_report", ""),
        state.get("fundamentals_report", ""),
        history,
        risk.get(other_response_keys[0], ""),
        risk.get(other_response_keys[1], ""),
    )

    result = await client.run(
        role, prompt, model=cfg["quick_think_llm"], max_turns=SINGLE_SHOT_MAX_TURNS
    )
    argument = f"{speaker} Analyst: {result.text}"

    new_risk = {
        "history": history + "\n" + argument,
        "aggressive_history": risk.get("aggressive_history", ""),
        "conservative_history": risk.get("conservative_history", ""),
        "neutral_history": risk.get("neutral_history", ""),
        "latest_speaker": speaker,
        "current_aggressive_response": risk.get("current_aggressive_response", ""),
        "current_conservative_response": risk.get("current_conservative_response", ""),
        "current_neutral_response": risk.get("current_neutral_response", ""),
        "judge_decision": risk.get("judge_decision", ""),
        "count": risk["count"] + 1,
    }
    new_risk[f"{role}_history"] = new_risk[f"{role}_history"] + "\n" + argument
    new_risk[f"current_{role}_response"] = argument
    return {"risk_debate_state": new_risk}
