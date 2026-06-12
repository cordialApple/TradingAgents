"""Default configuration for tradingagents-cc.

Mirrors the surviving keys of the parent ``tradingagents/default_config.py``,
minus the multi-provider LLM factory knobs — this port speaks only to the
Claude Agent SDK under subscription auth. Path keys honour the same
``TRADINGAGENTS_*`` environment overrides as the parent so both projects can
share one ``~/.tradingagents`` tree (cache, logs, memory log).

``TRADINGAGENTS_CC_MOCK=1`` forces ``llm_backend="mock"`` — the offline backend
that never imports ``claude_agent_sdk`` and never spends subscription credit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Valid analyst identifiers, in canonical pipeline order.
VALID_ANALYSTS: tuple[str, ...] = ("market", "social", "news", "fundamentals")

# Keys recognised in config/routine.toml; anything else is a typo and raises.
_ROUTINE_KEYS = frozenset({
    "tickers", "depth", "analysts", "quick_model", "deep_model",
    "output_language", "checkpoint_enabled",
})


def _mock_forced() -> bool:
    return os.environ.get("TRADINGAGENTS_CC_MOCK", "").strip() == "1"


def _path_default(env_var: str, default: Path) -> str:
    return str(Path(os.environ.get(env_var) or default))


def _default_config() -> dict[str, Any]:
    """Build a fresh default config, reading env overrides at call time."""
    home = Path.home() / ".tradingagents"
    return {
        # LLM settings. Quick tier: 4 analysts, bull/bear, trader, 3 risk
        # debaters, reflector. Deep tier: Research Manager + Portfolio Manager.
        "llm_backend": "mock" if _mock_forced() else "sdk",
        "quick_think_llm": "claude-sonnet-4-6",
        "deep_think_llm": "claude-opus-4-8",
        "anthropic_effort": None,  # None | "high" | "medium" | "low"
        # Pipeline shape
        "selected_analysts": list(VALID_ANALYSTS),
        "max_debate_rounds": 1,        # investment debate ends at count >= 2*rounds
        "max_risk_discuss_rounds": 1,  # risk debate ends at count >= 3*rounds
        "max_analyst_turns": 12,       # tool-loop budget per analyst query()
        "bind_insider_to_news": False,
        # Output language for analyst reports and the final decision.
        # Internal agent debate stays in English for reasoning quality.
        "output_language": "English",
        "checkpoint_enabled": True,
        # Paths — shared with the parent project, identical env overrides.
        "results_dir": _path_default("TRADINGAGENTS_RESULTS_DIR", home / "logs"),
        "data_cache_dir": _path_default("TRADINGAGENTS_CACHE_DIR", home / "cache"),
        "memory_log_path": _path_default(
            "TRADINGAGENTS_MEMORY_LOG_PATH", home / "memory" / "trading_memory.md"
        ),
        # Optional cap on resolved memory-log entries; None disables rotation.
        "memory_log_max_entries": None,
        # Data vendor configuration, forwarded to tradingagents.dataflows via
        # to_parent_config(). Defaults are 100% yfinance — keyless.
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
        # Tool-level overrides (take precedence over category-level).
        "tool_vendors": {},
        # SDK call retry knobs: attempts with backoff base*3^n (5s/15s/45s) + jitter.
        "retry_attempts": 3,
        "retry_base_delay": 5.0,
    }


DEFAULT_CONFIG: dict[str, Any] = _default_config()


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a fresh config: env-aware defaults merged with ``overrides``.

    Nested dict values (``data_vendors``, ``tool_vendors``) are merged
    key-by-key; scalar values are replaced. ``TRADINGAGENTS_CC_MOCK=1`` wins
    over any ``llm_backend`` override so tests can never reach the SDK.
    Never mutates ``DEFAULT_CONFIG`` or ``overrides``.
    """
    config = _default_config()
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    if _mock_forced():
        config["llm_backend"] = "mock"
    return config


def load_routine_config(path: str | Path) -> dict[str, Any]:
    """Parse a routine TOML file into ``{"tickers": [...], "overrides": {...}}``.

    ``overrides`` maps routine keys onto config keys (``depth`` sets both
    debate-round knobs, parity with the parent CLI depth presets) and is ready
    to feed to :func:`load_config`. Tickers are upper-cased so checkpoint
    thread ids stay stable across runs. Unknown or malformed keys raise
    ``ValueError`` so a typo in an unattended run fails loudly instead of
    being silently ignored.
    """
    try:
        import tomllib  # stdlib on Python 3.11+
    except ModuleNotFoundError:  # Python 3.10: declared dependency fallback
        import tomli as tomllib  # type: ignore[no-redef]

    with Path(path).open("rb") as f:
        raw = tomllib.load(f)

    unknown = sorted(set(raw) - _ROUTINE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown routine config keys {unknown}; valid keys: {sorted(_ROUTINE_KEYS)}"
        )

    tickers = raw.get("tickers", ["SPY"])
    if (
        not isinstance(tickers, list)
        or not tickers
        or not all(isinstance(t, str) and t.strip() for t in tickers)
    ):
        raise ValueError("'tickers' must be a non-empty list of ticker strings")
    tickers = [t.strip().upper() for t in tickers]

    overrides: dict[str, Any] = {}

    if "depth" in raw:
        depth = raw["depth"]
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ValueError("'depth' must be a positive integer")
        overrides["max_debate_rounds"] = depth
        overrides["max_risk_discuss_rounds"] = depth

    if "analysts" in raw:
        analysts = raw["analysts"]
        if (
            not isinstance(analysts, list)
            or not analysts
            or any(a not in VALID_ANALYSTS for a in analysts)
            or len(set(analysts)) != len(analysts)
        ):
            raise ValueError(
                f"'analysts' must be a non-empty, duplicate-free subset of {list(VALID_ANALYSTS)}"
            )
        overrides["selected_analysts"] = list(analysts)

    for toml_key, config_key in (
        ("quick_model", "quick_think_llm"),
        ("deep_model", "deep_think_llm"),
    ):
        if toml_key in raw:
            model = raw[toml_key]
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"'{toml_key}' must be a non-empty model name string")
            overrides[config_key] = model.strip()

    if "output_language" in raw:
        language = raw["output_language"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError("'output_language' must be a non-empty string")
        overrides["output_language"] = language.strip()

    if "checkpoint_enabled" in raw:
        enabled = raw["checkpoint_enabled"]
        if not isinstance(enabled, bool):
            raise ValueError("'checkpoint_enabled' must be a boolean")
        overrides["checkpoint_enabled"] = enabled

    return {"tickers": tickers, "overrides": overrides}


def to_parent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Project the port config onto the keys the parent dataflows layer reads.

    Fed to ``tradingagents.dataflows.config.set_config`` before any
    ``route_to_vendor`` call. Only the three keys the dataflows code consumes
    are forwarded (interface.py reads data_vendors/tool_vendors,
    stockstats_utils.py reads data_cache_dir) so port-specific keys never
    leak into the parent's module-global config.
    """
    return {
        "data_vendors": dict(config["data_vendors"]),
        "tool_vendors": dict(config["tool_vendors"]),
        "data_cache_dir": str(config["data_cache_dir"]),
    }
