# Ported from tradingagents/graph/conditional_logic.py (+ analyst chaining from tradingagents/graph/setup.py).
"""Loop-routing predicates for the two debate stages, as standalone pure functions.

Verbatim parent semantics (Hard invariant 3) — trust the formulas, not the
parent's stale "3 rounds" comments: the investment debate runs ``2 * N`` turns
(Bull speaks first, strict alternation keyed on ``current_response``); the risk
debate runs ``3 * N`` turns rotating Aggressive -> Conservative -> Neutral keyed
on ``latest_speaker``. Return values are plain strings (next speaker, or a DONE
sentinel) or ``None`` so the asyncio driver dispatches without graph machinery.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "DEBATE_DONE",
    "RISK_DONE",
    "should_continue_debate",
    "should_continue_risk_analysis",
    "next_analyst",
]

# DONE sentinels: the parent's conditional-edge targets kept verbatim, so the
# routing string doubles as the name of the stage that follows the loop.
DEBATE_DONE = "Research Manager"
RISK_DONE = "Portfolio Manager"


def should_continue_debate(
    state: Mapping[str, Any], max_debate_rounds: int = 1
) -> str:
    """Next speaker in the bull/bear debate, or ``DEBATE_DONE`` when it ends.

    On a fresh debate state ``current_response`` is ``""``, so Bull speaks first.
    """
    debate = state["investment_debate_state"]
    if debate["count"] >= 2 * max_debate_rounds:
        return DEBATE_DONE
    if debate["current_response"].startswith("Bull"):
        return "Bear Researcher"
    return "Bull Researcher"


def should_continue_risk_analysis(
    state: Mapping[str, Any], max_risk_discuss_rounds: int = 1
) -> str:
    """Next speaker in the risk debate, or ``RISK_DONE`` when it ends.

    On a fresh risk state ``latest_speaker`` is ``""``, so Aggressive speaks
    first; thereafter the prefix rotation is Aggressive -> Conservative ->
    Neutral -> Aggressive.
    """
    risk = state["risk_debate_state"]
    if risk["count"] >= 3 * max_risk_discuss_rounds:
        return RISK_DONE
    if risk["latest_speaker"].startswith("Aggressive"):
        return "Conservative Analyst"
    if risk["latest_speaker"].startswith("Conservative"):
        return "Neutral Analyst"
    return "Aggressive Analyst"


def next_analyst(current: str, selected_analysts: list[str]) -> str | None:
    """Analyst key after ``current`` in the selected order, or ``None`` when
    ``current`` is last (the parent graph then hands off to Bull Researcher).

    Raises ``ValueError`` if ``current`` is not a selected analyst — that is a
    driver wiring bug, not a runtime condition to mask.
    """
    idx = selected_analysts.index(current)
    if idx + 1 < len(selected_analysts):
        return selected_analysts[idx + 1]
    return None
