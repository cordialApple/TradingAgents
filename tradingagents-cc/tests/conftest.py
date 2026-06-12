"""Test isolation guaranteeing zero subscription credit and zero network usage.

Hard invariant 1 is enforced here in layers: ``TRADINGAGENTS_CC_MOCK=1``
forces the mock backend, both auth env vars are deleted (an accidental
``SdkAgentClient`` construction raises ``AuthError`` instead of billing),
every artifact path is redirected into ``tmp_path``, and any socket connect
raises. Tests marked ``live`` opt out of the mock/auth/socket layers and are
auto-skipped unless ``TRADINGAGENTS_CC_LIVE=1`` (the marker itself is
declared in pyproject.toml).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Callable

import pytest

from tradingagents_cc.default_config import load_config

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_LIVE_ENABLED = os.environ.get("TRADINGAGENTS_CC_LIVE", "").strip() == "1"


def _is_live(request: pytest.FixtureRequest) -> bool:
    return request.node.get_closest_marker("live") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``live`` tests unless explicitly opted in via TRADINGAGENTS_CC_LIVE=1."""
    if _LIVE_ENABLED:
        return
    skip_live = pytest.mark.skip(
        reason=(
            "live test: set TRADINGAGENTS_CC_LIVE=1 (and a real "
            "CLAUDE_CODE_OAUTH_TOKEN) to run; consumes subscription credit"
        )
    )
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _force_mock_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config() can only ever yield llm_backend='mock' (live tests excepted)."""
    if _is_live(request):
        return
    monkeypatch.setenv("TRADINGAGENTS_CC_MOCK", "1")


@pytest.fixture(autouse=True)
def _delete_auth_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """An accidental real-SDK path fails loudly (AuthError) instead of billing.

    Live tests keep their environment: they require the real OAuth token.
    """
    if _is_live(request):
        return
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _redirect_data_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    """Artifact/cache/memory paths resolve under tmp_path, never ~/.tradingagents."""
    results_dir = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    memory_log = tmp_path / "memory" / "trading_memory.md"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    memory_log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("TRADINGAGENTS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("TRADINGAGENTS_MEMORY_LOG_PATH", str(memory_log))
    return {
        "results_dir": results_dir,
        "data_cache_dir": cache_dir,
        "memory_log_path": memory_log,
    }


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Socket guard: any non-loopback connect attempt fails fast.

    Loopback is allowed because Windows' Proactor event loop (required per
    DESIGN.md invariant 8) builds its self-pipe via ``socket.socketpair()``,
    which on Windows falls back to a 127.0.0.1 connect. Disabled for ``live``
    tests.
    """
    if _is_live(request):
        return

    real_connect = socket.socket.connect
    _LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

    def _guarded_connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else address
        if isinstance(host, str) and host in _LOOPBACK_HOSTS:
            return real_connect(self, address, *args, **kwargs)
        raise RuntimeError("network disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)


@pytest.fixture
def make_config(
    _redirect_data_dirs: dict[str, Path],
) -> Callable[[dict[str, Any] | None], dict[str, Any]]:
    """Factory for a full config dict: mock backend + tmp_path-scoped paths.

    ``overrides`` are merged via load_config(), so nested dicts
    (data_vendors/tool_vendors) merge key-by-key and TRADINGAGENTS_CC_MOCK=1
    (set by the autouse fixture) still wins over any llm_backend override.
    """

    def _make(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        base: dict[str, Any] = {
            "llm_backend": "mock",
            "results_dir": str(_redirect_data_dirs["results_dir"]),
            "data_cache_dir": str(_redirect_data_dirs["data_cache_dir"]),
            "memory_log_path": str(_redirect_data_dirs["memory_log_path"]),
        }
        base.update(overrides or {})
        return load_config(base)

    return _make


@pytest.fixture
def recorded_route_to_vendor(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Patch tools_data.route_to_vendor with a loader over tests/fixtures/ recordings.

    Each data method returns the recorded yfinance output string from
    ``tests/fixtures/{method}.txt``; an unrecorded method raises ValueError
    (mirroring the parent router's unsupported-method behavior). The returned
    fake exposes ``.calls`` — ordered ``(method, args, kwargs)`` tuples — so
    tests can assert pass-through argument order.
    """
    # Local import: tools_data pulls in claude_agent_sdk (permitted for that
    # module only); keep it out of collection-time imports.
    from tradingagents_cc import tools_data

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _fake_route_to_vendor(method: str, *args: Any, **kwargs: Any) -> str:
        calls.append((method, args, kwargs))
        recording = FIXTURES_DIR / f"{method}.txt"
        if not recording.is_file():
            raise ValueError(
                f"Method '{method}' has no recorded fixture under {FIXTURES_DIR}"
            )
        return recording.read_text(encoding="utf-8")

    _fake_route_to_vendor.calls = calls  # type: ignore[attr-defined]
    monkeypatch.setattr(tools_data, "route_to_vendor", _fake_route_to_vendor)
    return _fake_route_to_vendor
