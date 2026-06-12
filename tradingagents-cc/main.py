"""Minimal programmatic example — the port's equivalent of the parent main.py.

Runs one full pipeline for NVDA as of yesterday and prints the 5-tier signal
(Buy/Overweight/Hold/Underweight/Sell) plus the report path. Mock backend by
default so this is free and offline; pass --live to opt into the real Claude
Agent SDK backend (requires CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from tradingagents_cc.client import get_client
from tradingagents_cc.default_config import load_config
from tradingagents_cc.pipeline import TradingAgentsPipeline


async def main(live: bool) -> None:
    # Create a custom config (parent main.py parity). Data vendors default to
    # yfinance across the board — no extra API keys needed.
    config = load_config({
        "llm_backend": "sdk" if live else "mock",
        "max_debate_rounds": 1,
    })

    ticker = "NVDA"
    trade_date = (date.today() - timedelta(days=1)).isoformat()

    # Initialize the pipeline and forward propagate.
    pipeline = TradingAgentsPipeline(config, get_client(config))
    _final_state, signal = await pipeline.propagate(ticker, trade_date)

    report = Path(config["results_dir"]) / ticker / trade_date / "complete_report.md"
    print(f"{ticker} {trade_date}: {signal}")
    print(f"Report: {report}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(live="--live" in sys.argv[1:]))
