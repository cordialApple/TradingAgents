# Tool names, signatures, and docstrings vendored from tradingagents/agents/utils/
# {core_stock_tools,technical_indicators_tools,news_data_tools,fundamental_data_tools}.py (parent repo).
"""In-process MCP server exposing the 9 keyless data tools.

Each handler keeps the parent LangChain tool's exact name, parameters, and
docstring (the docstring becomes the MCP tool description — the analyst
prompts reference these tools by name and argument list, so they must not
drift). Bodies delegate to the parent's vendor router via
``asyncio.to_thread`` so blocking yfinance/pandas work never stalls the SDK
event loop. Handlers NEVER raise: any failure is returned as an
``is_error`` text result, because an exception escaping an SDK MCP handler
kills the whole ``query()``.

Call :func:`apply_data_config` once before any tool can run — the parent
dataflows layer routes vendors off its module-global config.
"""

from __future__ import annotations

import asyncio
from inspect import cleandoc
from typing import Annotated, Any

from claude_agent_sdk import McpSdkServerConfig, ToolAnnotations, create_sdk_mcp_server, tool

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor

from .default_config import to_parent_config

__all__ = [
    "get_stock_data",
    "get_indicators",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "build_data_server",
    "analyst_toolsets",
    "apply_data_config",
]

# All nine tools are pure reads of market data — no state is mutated anywhere.
_READ_ONLY = ToolAnnotations(readOnlyHint=True)


def _text(result: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(result)}]}


def _error(exc: BaseException) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {exc}"}], "is_error": True}


# ---------------------------------------------------------------------------
# Core stock data
# ---------------------------------------------------------------------------

@tool(
    "get_stock_data",
    cleandoc(
        """
        Retrieve stock price data (OHLCV) for a given ticker symbol.
        Uses the configured core_stock_apis vendor.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """
    ),
    {
        "symbol": Annotated[str, "ticker symbol of the company"],
        "start_date": Annotated[str, "Start date in yyyy-mm-dd format"],
        "end_date": Annotated[str, "End date in yyyy-mm-dd format"],
    },
    annotations=_READ_ONLY,
)
async def get_stock_data(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor,
                "get_stock_data",
                args["symbol"],
                args["start_date"],
                args["end_date"],
            )
        )
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

@tool(
    "get_indicators",
    cleandoc(
        """
        Retrieve a single technical indicator for a given ticker symbol.
        Uses the configured technical_indicators vendor.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
        """
    ),
    {
        "symbol": Annotated[str, "ticker symbol of the company"],
        "indicator": Annotated[str, "technical indicator to get the analysis and report of"],
        "curr_date": Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
        "look_back_days": Annotated[int, "how many days to look back"],
    },
    annotations=_READ_ONLY,
)
async def get_indicators(args: dict[str, Any]) -> dict[str, Any]:
    try:
        symbol = args["symbol"]
        curr_date = args["curr_date"]
        look_back_days = int(args.get("look_back_days", 30))
        # LLMs sometimes pass multiple indicators as a comma-separated string;
        # split and process each individually.
        indicators = [i.strip().lower() for i in args["indicator"].split(",") if i.strip()]
        results = []
        for ind in indicators:
            try:
                results.append(
                    await asyncio.to_thread(
                        route_to_vendor, "get_indicators", symbol, ind, curr_date, look_back_days
                    )
                )
            except ValueError as e:
                results.append(str(e))
        return _text("\n\n".join(results))
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

@tool(
    "get_news",
    cleandoc(
        """
        Retrieve news data for a given ticker symbol.
        Uses the configured news_data vendor.
        Args:
            ticker (str): Ticker symbol
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing news data
        """
    ),
    {
        "ticker": Annotated[str, "Ticker symbol"],
        "start_date": Annotated[str, "Start date in yyyy-mm-dd format"],
        "end_date": Annotated[str, "End date in yyyy-mm-dd format"],
    },
    annotations=_READ_ONLY,
)
async def get_news(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor,
                "get_news",
                args["ticker"],
                args["start_date"],
                args["end_date"],
            )
        )
    except Exception as e:
        return _error(e)


@tool(
    "get_global_news",
    cleandoc(
        """
        Retrieve global news data.
        Uses the configured news_data vendor.
        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): Number of days to look back (default 7)
            limit (int): Maximum number of articles to return (default 5)
        Returns:
            str: A formatted string containing global news data
        """
    ),
    {
        "curr_date": Annotated[str, "Current date in yyyy-mm-dd format"],
        "look_back_days": Annotated[int, "Number of days to look back"],
        "limit": Annotated[int, "Maximum number of articles to return"],
    },
    annotations=_READ_ONLY,
)
async def get_global_news(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor,
                "get_global_news",
                args["curr_date"],
                int(args.get("look_back_days", 7)),
                int(args.get("limit", 5)),
            )
        )
    except Exception as e:
        return _error(e)


@tool(
    "get_insider_transactions",
    cleandoc(
        """
        Retrieve insider transaction information about a company.
        Uses the configured news_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
        Returns:
            str: A report of insider transaction data
        """
    ),
    {
        "ticker": Annotated[str, "ticker symbol"],
    },
    annotations=_READ_ONLY,
)
async def get_insider_transactions(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(route_to_vendor, "get_insider_transactions", args["ticker"])
        )
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

@tool(
    "get_fundamentals",
    cleandoc(
        """
        Retrieve comprehensive fundamental data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing comprehensive fundamental data
        """
    ),
    {
        "ticker": Annotated[str, "ticker symbol"],
        "curr_date": Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    },
    annotations=_READ_ONLY,
)
async def get_fundamentals(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor, "get_fundamentals", args["ticker"], args["curr_date"]
            )
        )
    except Exception as e:
        return _error(e)


@tool(
    "get_balance_sheet",
    cleandoc(
        """
        Retrieve balance sheet data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            freq (str): Reporting frequency: annual/quarterly (default quarterly)
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing balance sheet data
        """
    ),
    {
        "ticker": Annotated[str, "ticker symbol"],
        "freq": Annotated[str, "reporting frequency: annual/quarterly"],
        "curr_date": Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    },
    annotations=_READ_ONLY,
)
async def get_balance_sheet(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor,
                "get_balance_sheet",
                args["ticker"],
                args.get("freq", "quarterly"),
                args.get("curr_date"),
            )
        )
    except Exception as e:
        return _error(e)


@tool(
    "get_cashflow",
    cleandoc(
        """
        Retrieve cash flow statement data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            freq (str): Reporting frequency: annual/quarterly (default quarterly)
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing cash flow statement data
        """
    ),
    {
        "ticker": Annotated[str, "ticker symbol"],
        "freq": Annotated[str, "reporting frequency: annual/quarterly"],
        "curr_date": Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    },
    annotations=_READ_ONLY,
)
async def get_cashflow(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor,
                "get_cashflow",
                args["ticker"],
                args.get("freq", "quarterly"),
                args.get("curr_date"),
            )
        )
    except Exception as e:
        return _error(e)


@tool(
    "get_income_statement",
    cleandoc(
        """
        Retrieve income statement data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            freq (str): Reporting frequency: annual/quarterly (default quarterly)
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing income statement data
        """
    ),
    {
        "ticker": Annotated[str, "ticker symbol"],
        "freq": Annotated[str, "reporting frequency: annual/quarterly"],
        "curr_date": Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    },
    annotations=_READ_ONLY,
)
async def get_income_statement(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _text(
            await asyncio.to_thread(
                route_to_vendor,
                "get_income_statement",
                args["ticker"],
                args.get("freq", "quarterly"),
                args.get("curr_date"),
            )
        )
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Server + wiring
# ---------------------------------------------------------------------------

def build_data_server() -> McpSdkServerConfig:
    """Build the in-process MCP server config (mounted as ``mcp_servers={"data": ...}``)."""
    return create_sdk_mcp_server(
        name="data",
        tools=[
            get_stock_data,
            get_indicators,
            get_news,
            get_global_news,
            get_insider_transactions,
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ],
    )


def analyst_toolsets(config: dict[str, Any]) -> dict[str, list[str]]:
    """Per-analyst allowed-tools lists, fully qualified as ``mcp__data__*``.

    Mirrors the parent's ``graph/setup.py`` bindings; ``bind_insider_to_news``
    additionally grants the news analyst the insider-transactions tool (the
    parent imports it but never binds it anywhere).
    """
    news_tools = ["mcp__data__get_news", "mcp__data__get_global_news"]
    if config.get("bind_insider_to_news"):
        news_tools.append("mcp__data__get_insider_transactions")
    return {
        "market": ["mcp__data__get_stock_data", "mcp__data__get_indicators"],
        "social": ["mcp__data__get_news"],
        "news": news_tools,
        "fundamentals": [
            "mcp__data__get_fundamentals",
            "mcp__data__get_balance_sheet",
            "mcp__data__get_cashflow",
            "mcp__data__get_income_statement",
        ],
    }


def apply_data_config(config: dict[str, Any]) -> None:
    """Push vendor routing + cache dir into the parent dataflows module-global config.

    Must run once per process before any tool handler fires; otherwise
    ``route_to_vendor`` resolves vendors off the parent's own defaults.
    """
    set_config(to_parent_config(config))
