# Ported from tradingagents/graph/trading_graph.py (_fetch_returns, _resolve_pending_entries)
# and tradingagents/graph/reflection.py (Reflector) — LangChain invoke replaced by the
# AgentClient seam.
"""Phase-B outcome resolution for the memory log.

At the start of a run the driver resolves any pending memory-log entries for
the current ticker: fetch realised ticker and SPY returns over the holding
window via yfinance, generate a 2-4 sentence reflection with one quick-tier
LLM call per entry, then rewrite the log atomically in a single batch. Every
failure (price data not yet available, network error, LLM error) is logged
and non-fatal — the entry simply stays pending and is retried on the next run
for that ticker.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import yfinance as yf

from .memory import TradingMemoryLog
from .prompts import REFLECTOR_SYSTEM, build_reflector_user_prompt

if TYPE_CHECKING:  # avoid a runtime import of the client seam
    from .client import AgentClient

logger = logging.getLogger(__name__)

__all__ = ["fetch_returns", "resolve_pending_entries"]


def fetch_returns(
    ticker: str, trade_date: str, holding_days: int = 5
) -> tuple[float | None, float | None, int | None]:
    """Fetch raw and alpha return for ticker over holding_days from trade_date.

    Returns ``(raw_return, alpha_return, actual_holding_days)`` or
    ``(None, None, None)`` if price data is unavailable (too recent, delisted,
    or network error). The download window extends ``holding_days + 7``
    calendar days past the trade date as a buffer for weekends/holidays, and
    the realised holding period is clamped to the trading days actually
    returned: ``actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)``.
    """
    try:
        start = datetime.strptime(trade_date, "%Y-%m-%d")
        end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
        end_str = end.strftime("%Y-%m-%d")

        stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
        spy = yf.Ticker("SPY").history(start=trade_date, end=end_str)

        if len(stock) < 2 or len(spy) < 2:
            return None, None, None

        actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)
        raw = float(
            (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
            / stock["Close"].iloc[0]
        )
        spy_ret = float(
            (spy["Close"].iloc[actual_days] - spy["Close"].iloc[0])
            / spy["Close"].iloc[0]
        )
        alpha = raw - spy_ret
        return raw, alpha, actual_days
    except Exception as e:
        logger.warning(
            "Could not resolve outcome for %s on %s (will retry next run): %s",
            ticker, trade_date, e,
        )
        return None, None, None


async def resolve_pending_entries(
    memory_log: TradingMemoryLog,
    ticker: str,
    client: "AgentClient",
    cfg: dict[str, Any],
) -> None:
    """Resolve pending memory-log entries for ticker at the start of a new run.

    Fetches returns for each same-ticker pending entry, generates a reflection
    per resolvable entry with ONE quick-tier (``cfg["quick_think_llm"]``),
    no-tool, ``max_turns=1`` reflector call, then writes all updates in a
    single atomic batch via ``memory_log.batch_update_with_outcomes``. Any
    failure — unavailable price data, an LLM error, an empty reflection — is
    logged and non-fatal: the affected entry stays pending and is retried the
    next time this ticker runs.

    Contract: the CALLER must skip this function entirely in mock mode
    (``llm_backend == "mock"``) — Phase B needs live yfinance prices and a
    real reflector call, neither of which the mock backend can supply, and
    skipping it keeps mock runs network-free.

    Trade-off (parent parity): only same-ticker entries are resolved per run.
    Entries for other tickers accumulate until that ticker is run again.
    """
    pending = [e for e in memory_log.get_pending_entries() if e["ticker"] == ticker]
    if not pending:
        return

    updates: list[dict[str, Any]] = []
    for entry in pending:
        # yfinance is blocking network I/O — keep it off the event loop.
        raw, alpha, days = await asyncio.to_thread(fetch_returns, ticker, entry["date"])
        if raw is None:
            continue  # price not available yet — try again next run

        try:
            result = await client.run(
                role="reflector",
                prompt=build_reflector_user_prompt(
                    final_decision=entry.get("decision", ""),
                    raw_return=raw,
                    alpha_return=alpha,
                ),
                system_prompt=REFLECTOR_SYSTEM,
                model=cfg["quick_think_llm"],
                max_turns=1,
            )
        except Exception as e:
            logger.warning(
                "Reflection call failed for %s on %s (entry stays pending): %s",
                ticker, entry["date"], e,
            )
            continue

        reflection = result.text.strip()
        if not reflection:
            logger.warning(
                "Reflector returned empty text for %s on %s (entry stays pending)",
                ticker, entry["date"],
            )
            continue

        updates.append({
            "ticker": ticker,
            "trade_date": entry["date"],
            "raw_return": raw,
            "alpha_return": alpha,
            "holding_days": days,
            "reflection": reflection,
        })

    if not updates:
        return

    try:
        memory_log.batch_update_with_outcomes(updates)
        logger.info(
            "Resolved %d of %d pending memory entries for %s",
            len(updates), len(pending), ticker,
        )
    except Exception as e:
        logger.error(
            "Failed to write %d resolved outcomes for %s (entries stay pending): %s",
            len(updates), ticker, e,
        )
