"""Deterministic mock backend for the ``AgentClient`` seam.

Implements the frozen ``AgentClient`` protocol from client.py without ever
importing ``claude_agent_sdk`` (hard invariant: mock-mode tests consume zero
network and zero subscription credit).  Outputs are canned per role and seeded
by ``(role, ticker)`` — the ticker is recovered from the prompt text when
findable — so repeated runs over the same inputs produce byte-identical
results regardless of call order (required by the checkpoint-resume tests).

Test knobs:

- ``fail_structured``: roles whose structured attempt returns
  ``structured=None`` while ``.text`` still carries the parseable marker line
  (``**Rating**: X`` / ``**Recommendation**: X`` / the trailing
  ``FINAL TRANSACTION PROPOSAL: **X**``), exercising the pipeline's one-shot
  free-text fallback.
- ``raise_at_role``: the first call for that role raises ``StageError``;
  every later call succeeds, exercising checkpoint resume.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Iterable, Optional

from .schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_trader_proposal,
)

if TYPE_CHECKING:
    from .client import AgentResult


ROLES = (
    "market_analyst", "social_analyst", "news_analyst", "fundamentals_analyst",
    "bull", "bear", "research_manager", "trader",
    "aggressive", "conservative", "neutral",
    "portfolio_manager", "reflector",
)

_ANALYST_ROLES = frozenset(
    ("market_analyst", "social_analyst", "news_analyst", "fundamentals_analyst"),
)
_STRUCTURED_ROLES = frozenset(("research_manager", "trader", "portfolio_manager"))

# Pass 1: the instrument-context sentence every analyst/manager/trader/PM
# prompt embeds.  Pass 2: any backticked ticker-shaped token (analyst mock
# reports embed one, so debate prompts that quote those reports still seed by
# ticker).
_INSTRUMENT_RE = re.compile(r"instrument to analyze is `([^`\n]+)`")
_TICKER_RE = re.compile(r"`([A-Z0-9][A-Z0-9.\-=^]{0,11})`")


def _extract_ticker(prompt: str) -> Optional[str]:
    m = _INSTRUMENT_RE.search(prompt)
    if m:
        return m.group(1).strip()
    for cand in _TICKER_RE.findall(prompt):
        if any(c.isalpha() for c in cand):
            return cand
    return None


def _zero_usage() -> dict:
    return {"llm_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0}


# ---------------------------------------------------------------------------
# Canned analyst reports (markdown, each ending in a small key-points table)
# ---------------------------------------------------------------------------


def _market_report(symbol: str, rng: random.Random) -> str:
    close = round(rng.uniform(18.0, 480.0), 2)
    sma50 = round(close * rng.uniform(0.90, 1.10), 2)
    sma200 = round(close * rng.uniform(0.82, 1.15), 2)
    rsi = round(rng.uniform(28.0, 72.0), 1)
    macdh = round(rng.uniform(-2.5, 2.5), 2)
    atr = round(close * rng.uniform(0.015, 0.045), 2)
    trend = rng.choice(("constructive", "range-bound", "deteriorating"))
    above = "above" if close >= sma50 else "below"
    momentum = "improving" if macdh >= 0 else "fading"
    rsi_read = "stretched" if rsi > 65 else "washed out" if rsi < 35 else "mid-range"
    return (
        f"## Market Analysis — `{symbol}`\n\n"
        f"Price action over the trailing month is {trend}: the last close at {close} sits "
        f"{above} the 50-day SMA ({sma50}), while the 200-day SMA ({sma200}) anchors the "
        f"longer-horizon view. Momentum is {momentum}, with RSI at {rsi} ({rsi_read}) and "
        f"the MACD histogram printing {macdh}. ATR of {atr} frames the volatility budget "
        f"for any position taken this week.\n\n"
        f"Volume behaved in line with the prevailing trend, and the moving-average "
        f"structure offers clear reference levels for stops and adds.\n\n"
        f"| Indicator | Value | Read |\n"
        f"| --- | --- | --- |\n"
        f"| close | {close} | last session |\n"
        f"| close_50_sma | {sma50} | price {above} the average |\n"
        f"| close_200_sma | {sma200} | long-horizon anchor |\n"
        f"| rsi | {rsi} | {rsi_read} |\n"
        f"| macdh | {macdh} | momentum {momentum} |\n"
        f"| atr | {atr} | sizing guide |\n"
    )


def _social_report(symbol: str, rng: random.Random) -> str:
    mentions = [rng.randint(120, 4200) for _ in range(3)]
    net = [round(rng.uniform(-0.6, 0.7), 2) for _ in range(3)]
    mood = "constructive" if sum(net) > 0 else "skeptical"
    swing = "building" if net[-1] >= net[0] else "cooling"
    return (
        f"## Social Sentiment — `{symbol}`\n\n"
        f"Retail conversation around {symbol} over the past three sessions skews {mood}, "
        f"with mention volume peaking at {max(mentions)} and net sentiment {swing} into "
        f"the most recent session ({net[-1]:+.2f}). Thread quality is mixed: momentum "
        f"chatter dominates, but a vocal minority keeps pressing the cost and competition "
        f"questions.\n\n"
        f"No single influencer event drove the swing; the shift tracks the price tape, "
        f"which argues sentiment here is a follower rather than a leader.\n\n"
        f"| Day | Mentions | Net sentiment |\n"
        f"| --- | --- | --- |\n"
        f"| T-2 | {mentions[0]} | {net[0]:+.2f} |\n"
        f"| T-1 | {mentions[1]} | {net[1]:+.2f} |\n"
        f"| T | {mentions[2]} | {net[2]:+.2f} |\n"
    )


def _news_report(symbol: str, rng: random.Random) -> str:
    themes = rng.sample(
        (
            "Earnings preview chatter",
            "Supply-chain update",
            "Regulatory scrutiny",
            "Analyst coverage change",
            "Product launch coverage",
            "Macro rate commentary",
        ),
        3,
    )
    impacts = [rng.choice(("supportive", "adverse", "mixed")) for _ in range(3)]
    lean = "tailwind" if impacts.count("supportive") >= 2 else (
        "headwind" if impacts.count("adverse") >= 2 else "wash"
    )
    return (
        f"## News & Macro — `{symbol}`\n\n"
        f"The week's flow around {symbol} centers on three themes: "
        f"{themes[0].lower()}, {themes[1].lower()}, and {themes[2].lower()}. Taken "
        f"together the coverage nets out to a {lean} for the name, while the global "
        f"backdrop — rates, currency, and sector rotation — stays the dominant swing "
        f"factor for anything macro-sensitive.\n\n"
        f"Nothing in the flow changes the structural story; the items below are the "
        f"ones most likely to move positioning near term.\n\n"
        f"| Theme | Read | Implication |\n"
        f"| --- | --- | --- |\n"
        f"| {themes[0]} | {impacts[0]} | watch for follow-through |\n"
        f"| {themes[1]} | {impacts[1]} | priced within a session |\n"
        f"| {themes[2]} | {impacts[2]} | medium-term positioning |\n"
    )


def _fundamentals_report(symbol: str, rng: random.Random) -> str:
    pe = round(rng.uniform(8.0, 65.0), 1)
    rev_growth = round(rng.uniform(-4.0, 38.0), 1)
    gross_margin = round(rng.uniform(22.0, 78.0), 1)
    fcf_margin = round(rng.uniform(-2.0, 30.0), 1)
    leverage = rng.choice(("net cash", "modest net debt", "elevated leverage"))
    quality = "high" if gross_margin > 55 and fcf_margin > 12 else "middling"
    return (
        f"## Fundamentals — `{symbol}`\n\n"
        f"{symbol} screens as a {quality}-quality operator this quarter: revenue growth "
        f"of {rev_growth}% year over year against a {gross_margin}% gross margin, with "
        f"free-cash-flow conversion running at {fcf_margin}% of sales. The balance sheet "
        f"shows {leverage}, leaving management room to maneuver through a soft patch.\n\n"
        f"At {pe}x trailing earnings the market is paying for continuation of the "
        f"current trajectory; the statements themselves contain no red flags beyond the "
        f"usual working-capital seasonality.\n\n"
        f"| Metric | Value | Note |\n"
        f"| --- | --- | --- |\n"
        f"| P/E (trailing) | {pe}x | continuation priced in |\n"
        f"| Revenue growth (yoy) | {rev_growth}% | top-line trajectory |\n"
        f"| Gross margin | {gross_margin}% | pricing power proxy |\n"
        f"| FCF margin | {fcf_margin}% | cash conversion |\n"
        f"| Balance sheet | {leverage} | flexibility |\n"
    )


_ANALYST_BUILDERS = {
    "market_analyst": _market_report,
    "social_analyst": _social_report,
    "news_analyst": _news_report,
    "fundamentals_analyst": _fundamentals_report,
}


# ---------------------------------------------------------------------------
# Debate / risk prose (no role prefix — node code applies "Bull Analyst:" etc.)
# ---------------------------------------------------------------------------


_DEBATE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "bull": (
        "The growth runway for {symbol} remains underappreciated: revenue momentum is "
        "compounding while margins expand, and the moat keeps widening. The bear case "
        "leans on valuation alone, yet estimates keep revising higher — a setup that "
        "has historically rewarded staying invested. Even the technical picture shows "
        "accumulation rather than distribution.",
        "{symbol} is winning share in a market that is itself expanding, which "
        "compounds the upside in a way static multiples miss. The balance sheet gives "
        "management room to keep investing through any soft patch the bears point to. "
        "Sentiment is only beginning to catch up to the fundamentals, leaving room for "
        "further re-rating.",
        "Every leading indicator we track for {symbol} — bookings, pricing power, "
        "retention — is moving in the right direction at once. The bear argument "
        "recycles last quarter's worries that the latest numbers already refuted. When "
        "fundamentals and momentum agree like this, the asymmetric move is higher.",
    ),
    "bear": (
        "The market is pricing {symbol} for flawless execution, leaving no margin of "
        "safety if growth decelerates even modestly. Competitive pressure is "
        "intensifying just as the macro backdrop turns hostile to richly valued names. "
        "The recent pattern of lower highs suggests distribution by informed money is "
        "already underway.",
        "Beneath the headline numbers, {symbol}'s growth quality is eroding: "
        "incremental margins are thinning and customer concentration is rising. The "
        "bull thesis depends on a re-rating that the rate environment no longer "
        "supports. Risk is skewed to the downside until the next two quarters prove "
        "otherwise.",
        "{symbol} faces a classic expectations trap — guidance must keep beating a bar "
        "that resets higher every quarter. Insider behavior and waning retail "
        "enthusiasm both point to fading conviction. The prudent move is to reduce "
        "exposure before the crowd reaches the same conclusion.",
    ),
    "aggressive": (
        "Playing it safe on {symbol} is the riskiest move of all — the setup offers "
        "convexity that conservative sizing would squander. The data supports pressing "
        "the advantage while the crowd hesitates; outsized returns come from acting "
        "before consensus forms. Capping exposure now means paying for certainty "
        "exactly when it is most expensive.",
        "The trader's plan for {symbol} is directionally right but timid; this is the "
        "moment to lean in, not feather the position. Volatility is opportunity in "
        "this regime, and the downside scenarios the conservative desk worries about "
        "are already priced. Half-measures will leave most of the available return on "
        "the table.",
    ),
    "conservative": (
        "Preserving capital must come first: the plan for {symbol} assumes benign "
        "conditions the data does not guarantee. A staged entry with firm stops "
        "protects the desk if the thesis is wrong, at small cost if it is right. We "
        "should size for the drawdown we can survive, not the rally we hope for.",
        "The aggressive case ignores how quickly liquidity evaporates once this "
        "{symbol} trade gets crowded. Tightening risk limits and keeping dry powder "
        "costs little and keeps every option open. Discipline on position size is what "
        "lets us stay in the game long enough for the thesis to play out.",
    ),
    "neutral": (
        "Both desks overstate their case on {symbol}: the upside is real, but so is "
        "the fragility the conservative side flags. A middle path — moderate sizing "
        "with explicit triggers to scale either way — captures most of the opportunity "
        "while bounding the regret. Let the next data point, not conviction, dictate "
        "the adjustment.",
        "The truth on {symbol} sits between the two extremes argued here. Scaling in "
        "at moderate size with predefined add and trim levels respects both the "
        "momentum case and the drawdown risk. Flexibility, not bravado or caution, is "
        "the edge in this setup.",
    ),
}


def _debate_prose(role: str, symbol: str, rng: random.Random) -> str:
    return rng.choice(_DEBATE_TEMPLATES[role]).format(symbol=symbol)


def _reflector_blurb(symbol: str, rng: random.Random) -> str:
    factor = rng.choice(
        (
            "the agreement between momentum signals and the fundamentals narrative",
            "the debate surfacing a concrete invalidation level early",
            "discounting sentiment noise in favor of the cash-flow evidence",
        ),
    )
    return (
        f"The decisive factor this round was {factor} for {symbol}; weighting "
        f"confirmed evidence over conviction improved the quality of the final call. "
        f"Next time, fix an explicit invalidation level during the debate itself and "
        f"revisit sizing whenever the volatility regime shifts."
    )


# ---------------------------------------------------------------------------
# Structured decision-makers (valid vendored-schema instances, seeded rating)
# ---------------------------------------------------------------------------


def _research_plan(symbol: str, rng: random.Random) -> ResearchPlan:
    rec = rng.choice(list(PortfolioRating))
    if rec in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
        verdict = "the bull side carried the round with evidence the bears never countered"
        actions = (
            f"Build exposure in two tranches over the next week, sizing toward "
            f"{rng.randint(3, 8)}% of the book, with a stop below the recent swing low "
            f"consistent with the rating."
        )
    elif rec in (PortfolioRating.UNDERWEIGHT, PortfolioRating.SELL):
        verdict = "the bear side exposed assumptions the bulls could not defend"
        actions = (
            f"Trim exposure by roughly a {rng.choice(('third', 'quarter', 'half'))} "
            f"into strength, tighten stops on the remainder, and exit fully if the "
            f"thesis-breaking levels give way."
        )
    else:
        verdict = "neither side landed a decisive blow, so the evidence stays genuinely balanced"
        actions = (
            "Maintain the current position unchanged; set alerts at the levels both "
            "sides flagged and revisit after the next earnings update."
        )
    return ResearchPlan(
        recommendation=rec,
        rationale=(
            f"The bull case for {symbol} rested on compounding revenue momentum and a "
            f"widening moat, while the bear case centered on valuation and macro "
            f"fragility; on balance, {verdict}."
        ),
        strategic_actions=actions,
    )


def _trader_proposal(symbol: str, rng: random.Random) -> TraderProposal:
    action = rng.choice(list(TraderAction))
    entry = round(rng.uniform(18.0, 480.0), 2)
    sizing_pct = rng.randint(2, 8)
    if action is TraderAction.BUY:
        stop = round(entry * 0.93, 2)
        reasoning = (
            f"The research plan's conviction lines up with supportive technicals for "
            f"{symbol}, and the risk-reward at current levels clears the desk's "
            f"threshold. Entry near {entry} with a stop at {stop} bounds the downside "
            f"while momentum does the work."
        )
        return TraderProposal(
            action=action, reasoning=reasoning, entry_price=entry,
            stop_loss=stop, position_sizing=f"{sizing_pct}% of portfolio",
        )
    if action is TraderAction.SELL:
        stop = round(entry * 1.07, 2)
        reasoning = (
            f"The plan's bearish lean is confirmed by deteriorating breadth and fading "
            f"momentum in {symbol}; cutting now preserves gains before the crowd "
            f"repositions. A stop at {stop} caps the risk of a squeeze on the exit."
        )
        return TraderProposal(
            action=action, reasoning=reasoning, entry_price=entry,
            stop_loss=stop, position_sizing=f"{sizing_pct}% of portfolio",
        )
    reasoning = (
        f"The evidence on {symbol} is too balanced to justify new risk in either "
        f"direction this round. Standing pat with alerts around {entry} keeps the "
        f"desk ready to act once the range resolves."
    )
    return TraderProposal(action=action, reasoning=reasoning)


def _pm_decision(symbol: str, rng: random.Random) -> PortfolioDecision:
    rating = rng.choice(list(PortfolioRating))
    anchor = round(rng.uniform(18.0, 480.0), 2)
    horizon = rng.choice(("1-3 months", "3-6 months", "6-12 months"))
    if rating in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
        summary = (
            f"Initiate at market and add on weakness, building toward "
            f"{rng.randint(3, 8)}% of the portfolio over {horizon}. Respect a hard "
            f"stop {rng.randint(5, 9)}% below entry, with the 200-day average as the "
            f"key risk level."
        )
        target: Optional[float] = round(anchor * rng.uniform(1.08, 1.30), 2)
        stance = (
            "the aggressive desk's momentum evidence outweighed the drawdown math, "
            "and the fundamentals gave the move a floor"
        )
    elif rating in (PortfolioRating.UNDERWEIGHT, PortfolioRating.SELL):
        summary = (
            f"Reduce exposure into strength over the coming sessions and carry only a "
            f"residual position with tight stops, reassessing within {horizon}. "
            f"Redeploy once the risks the debate surfaced have resolved."
        )
        target = round(anchor * rng.uniform(0.72, 0.94), 2)
        stance = (
            "the conservative desk's drawdown math dominated, with the momentum case "
            "resting on assumptions the data no longer supports"
        )
    else:
        summary = (
            f"Keep the existing position and tighten alerts around the levels both "
            f"debate camps flagged, re-underwriting within {horizon} or on a decisive "
            f"break of the range."
        )
        target = None
        stance = (
            "neither desk produced evidence strong enough to overturn the other, so "
            "the balanced stance carries the least regret"
        )
    return PortfolioDecision(
        rating=rating,
        executive_summary=summary,
        investment_thesis=(
            f"The risk debate sharpened the trader's plan for {symbol}: {stance}. "
            f"Prior lessons in the context, where present, were weighed against the "
            f"current evidence rather than followed mechanically."
        ),
        price_target=target,
        time_horizon=horizon,
    )


_STRUCTURED_BUILDERS = {
    "research_manager": (_research_plan, render_research_plan),
    "trader": (_trader_proposal, render_trader_proposal),
    "portfolio_manager": (_pm_decision, render_pm_decision),
}


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class MockAgentClient:
    """Zero-cost ``AgentClient`` returning seeded canned outputs per role.

    ``config`` is accepted (and ignored) for symmetry with the
    ``get_client(config)`` factory; the test knobs are keyword-only.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        fail_structured: Optional[Iterable[str]] = None,
        raise_at_role: Optional[str] = None,
    ) -> None:
        self.config = config or {}
        self.fail_structured = frozenset(fail_structured or ())
        self.raise_at_role = raise_at_role
        self.call_counts: dict[str, int] = {}
        self.calls: list[str] = []  # ordered role names, for resume assertions

    async def run(
        self,
        role: str,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        model: str,
        tools_server: Optional[object] = None,
        allowed_tools: Optional[list[str]] = None,
        output_schema: Optional[dict] = None,
        max_turns: int = 1,
    ) -> "AgentResult":
        # Local import: client.get_client() imports this module, so a
        # module-level import would be circular.
        from .client import AgentResult, StageError

        if role not in ROLES:
            raise ValueError(f"unknown mock role {role!r}; expected one of {ROLES}")

        self.calls.append(role)
        self.call_counts[role] = self.call_counts.get(role, 0) + 1
        if role == self.raise_at_role and self.call_counts[role] == 1:
            raise StageError(f"injected mock failure at role {role!r} (first call)")

        ticker = _extract_ticker(prompt)
        symbol = ticker or "STOCK"
        rng = random.Random(f"{role}:{ticker}" if ticker else role)

        structured: Optional[dict] = None
        if role in _ANALYST_ROLES:
            text = _ANALYST_BUILDERS[role](symbol, rng)
        elif role in _STRUCTURED_ROLES:
            build, render = _STRUCTURED_BUILDERS[role]
            instance = build(symbol, rng)
            # Rendered markdown always carries the parseable marker line
            # (**Recommendation**/**Rating**/FINAL TRANSACTION PROPOSAL), so
            # the free-text fallback path stays parseable.
            text = render(instance)
            if output_schema is not None and role not in self.fail_structured:
                structured = instance.model_dump(mode="json")
        elif role == "reflector":
            text = _reflector_blurb(symbol, rng)
        else:
            text = _debate_prose(role, symbol, rng)

        return AgentResult(
            text=text,
            structured=structured,
            usage=_zero_usage(),
            num_turns=1,
            tool_call_log=[],
        )
