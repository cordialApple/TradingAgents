"""The one LLM seam: Claude Agent SDK client behind the frozen AgentClient protocol.

Every pipeline node funnels through ``AgentClient.run()`` — one
``claude_agent_sdk.query()`` per call, subscription auth only
(``CLAUDE_CODE_OAUTH_TOKEN`` from ``claude setup-token``; metered/routing
variables — ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``,
``CLAUDE_CODE_USE_BEDROCK``, ``CLAUDE_CODE_USE_VERTEX``,
``ANTHROPIC_BASE_URL`` — are actively removed from the environment).

Windows note: the SDK's subprocess transport requires the default
**Proactor** event loop. Never install
``asyncio.WindowsSelectorEventLoopPolicy`` (or any Selector policy) in a
process that constructs ``SdkAgentClient`` — subprocess support is missing
from Selector loops on Windows and ``query()`` will fail to spawn the CLI.

``claude_agent_sdk`` is imported lazily inside the SDK code paths so that
mock-backend users (and the default test suite) can import this module
without the SDK installed and without ever touching the network.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Built-in tools blocked on every call: analyst agents may only ever touch the
# in-process mcp__data__* server. The PRIMARY kill switch is ``tools=[]`` in
# _build_options (disables every built-in at the CLI level; MCP tools mounted
# via mcp_servers are unaffected) — this denylist and the can_use_tool callback
# are kept as redundant defense-in-depth layers behind it.
DISALLOWED_BUILTIN_TOOLS: tuple[str, ...] = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch",
    "Task", "NotebookEdit", "TodoWrite", "WebView",
    # Bundled-CLI built-ins missing from the original list.
    "NotebookRead", "BashOutput", "KillShell", "ExitPlanMode",
    "ListMcpResourcesTool", "ReadMcpResourceTool", "AskUserQuestion", "Skill",
)


# Env vars evicted before any SDK call: each one can silently route the spawned
# CLI's billing onto metered auth (API key / bearer token) or a cloud provider
# (Bedrock / Vertex) or an alternate endpoint, defeating the subscription-only
# invariant. Popped with a warning in SdkAgentClient.__init__.
_METERED_AUTH_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_BASE_URL",
)


class AuthError(Exception):
    """Subscription auth is missing or unusable. Fail fast, never bill an API key."""


class StageError(Exception):
    """A pipeline stage's LLM call failed permanently (after retries / error result)."""


class _RateLimitError(Exception):
    """Internal: a 429-shaped error ResultMessage (rate limit / overload).

    DESIGN.md's retry contract includes "429-shaped errors" alongside the
    transport exceptions, so ``_run_once`` raises this instead of StageError
    when the error result looks like a rate limit; ``run()`` retries it with
    the same 5s/15s/45s + jitter backoff and converts it to StageError only
    after the attempts are exhausted. Never escapes ``run()``.
    """


@dataclass
class AgentResult:
    text: str                      # ResultMessage.result or "" — free-text answer
    structured: dict | None        # structured_output when schema requested & valid, else None
    usage: dict                    # {"llm_calls", "tool_calls", "tokens_in", "tokens_out"}
    num_turns: int
    tool_call_log: list[str] = field(default_factory=list)  # "HH:MM:SS [Tool Call] name(args)"


class AgentClient(Protocol):
    async def run(
        self, role: str, prompt: str, *,
        system_prompt: str | None = None,
        model: str,
        tools_server: object | None = None,
        allowed_tools: list[str] | None = None,
        output_schema: dict | None = None,
        max_turns: int = 1,
    ) -> AgentResult: ...


def _format_tool_call_line(name: str, args: Any) -> str:
    """Byte-format parity with the parent CLI's message_tool.log lines."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    if isinstance(args, dict):
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
    else:
        args_str = "" if args is None else str(args)
    return f"{timestamp} [Tool Call] {name}({args_str})"


# Marker text identifying 429-shaped error results when the CLI does not
# report an HTTP status (api_error_status requires CLI >= 2.1.110).
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "429", "rate limit", "rate_limit", "rate-limit", "too many requests",
    "overloaded", "overloaded_error",
)


def _is_rate_limit_shaped(result_message: Any) -> bool:
    """True when an error ResultMessage looks like a rate limit / overload.

    Prefers the definitive ``api_error_status`` HTTP code (429 rate limited,
    529 overloaded) and falls back to marker text in subtype/errors/result.
    """
    status = getattr(result_message, "api_error_status", None)
    if status in (429, 529):
        return True
    errors = result_message.errors or ()
    parts = [
        str(result_message.subtype or ""),
        str(result_message.result or ""),
        *(str(e) for e in errors),
    ]
    text = " ".join(parts).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


async def _stream_prompt(prompt: str):
    """Wrap a plain prompt in the SDK's streaming-mode message shape.

    ``claude_agent_sdk.query()`` rejects a ``str`` prompt whenever a
    ``can_use_tool`` callback is set ("can_use_tool callback requires
    streaming mode"), and every call here carries the deny-by-default
    callback — so the prompt is always sent as a one-message async iterable.
    """
    yield {"type": "user", "message": {"role": "user", "content": prompt}}


def _sum_tokens(usage: Any) -> tuple[int, int]:
    """Defensive (tokens_in, tokens_out) from a ResultMessage.usage dict."""
    if not isinstance(usage, dict):
        return 0, 0

    def _int(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) else 0

    tokens_in = (
        _int("input_tokens")
        + _int("cache_creation_input_tokens")
        + _int("cache_read_input_tokens")
    )
    return tokens_in, _int("output_tokens")


class SdkAgentClient:
    """Claude Agent SDK implementation of the AgentClient protocol.

    Construction enforces the auth invariant (OAuth token required, metered
    key evicted) and imports ``claude_agent_sdk`` — so the mere import of this
    module stays SDK-free for mock users.
    """

    STRUCTURED_RETRY_SUBTYPE = "error_max_structured_output_retries"

    def __init__(self, config: dict):
        # Auth guarantee: never let the SDK fall back to metered billing or an
        # alternate provider/endpoint — the spawned CLI inherits this process's
        # environment, so every routing/billing variable is evicted up front.
        for var in _METERED_AUTH_ENV_VARS:
            if os.environ.pop(var, None) is not None:
                logger.warning(
                    "%s was set; removed from this process so the Agent SDK can "
                    "only use subscription auth (CLAUDE_CODE_OAUTH_TOKEN).",
                    var,
                )
        if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip():
            raise AuthError(
                "CLAUDE_CODE_OAUTH_TOKEN is not set — this pipeline runs on Claude "
                "subscription auth only. Remediation: run: claude setup-token "
                "(interactive, one-time) and export the token in this environment."
            )

        import claude_agent_sdk  # lazy: only the sdk backend ever imports it
        self._sdk = claude_agent_sdk

        self._config = config
        self._retry_attempts = int(config.get("retry_attempts", 3))
        self._retry_base_delay = float(config.get("retry_base_delay", 5.0))

        # Neutral existing cwd: keeps the agent subprocess out of any repo with
        # .claude/ or CLAUDE.md context (belt-and-braces with setting_sources=[]).
        cache_dir = config.get("data_cache_dir")
        self._cwd = (
            Path(cache_dir) / "cc_workdir" if cache_dir else Path(tempfile.gettempdir())
        )
        self._cwd.mkdir(parents=True, exist_ok=True)

    async def run(
        self, role: str, prompt: str, *,
        system_prompt: str | None = None,
        model: str,
        tools_server: object | None = None,
        allowed_tools: list[str] | None = None,
        output_schema: dict | None = None,
        max_turns: int = 1,
        effort: str | None = None,
        deep: bool = False,
    ) -> AgentResult:
        """One query() with retry on transport failures and 429-shaped error
        results (5s/15s/45s + jitter), per the DESIGN.md retry contract.

        ``effort``/``deep`` extend the frozen AgentClient protocol: ``deep``
        marks deep-tier calls (Research Manager / Portfolio Manager) so the
        config ``anthropic_effort`` knob applies to them only; an explicit
        ``effort`` wins over that knob. The tier is always declared by the
        caller — never inferred from the model string, which both tiers may
        legitimately share.
        """
        retryable = (
            self._sdk.CLIConnectionError,
            self._sdk.ProcessError,
            self._sdk.CLIJSONDecodeError,
            _RateLimitError,
        )
        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                return await self._run_once(
                    role, prompt,
                    system_prompt=system_prompt, model=model,
                    tools_server=tools_server, allowed_tools=allowed_tools,
                    output_schema=output_schema, max_turns=max_turns,
                    effort=effort, deep=deep,
                )
            except retryable as exc:
                last_exc = exc
                if attempt + 1 >= self._retry_attempts:
                    break
                delay = self._retry_base_delay * (3 ** attempt)
                delay += random.uniform(0.0, delay * 0.25)  # jitter
                logger.warning(
                    "[%s] transient SDK failure (%s: %s); retry %d/%d in %.1fs",
                    role, type(exc).__name__, exc,
                    attempt + 1, self._retry_attempts - 1, delay,
                )
                await asyncio.sleep(delay)
        raise StageError(
            f"[{role}] SDK call failed after {self._retry_attempts} attempts: {last_exc}"
        ) from last_exc

    def _build_options(
        self, *,
        system_prompt: str | None,
        model: str,
        tools_server: object | None,
        allowed_tools: list[str] | None,
        output_schema: dict | None,
        max_turns: int,
        effort: str | None,
        deep: bool,
    ):
        sdk = self._sdk
        allowed = frozenset(allowed_tools or ())

        # Deny-by-default second enforcement layer behind allowed_tools.
        async def can_use_tool(tool_name: str, tool_input: dict, context: object):
            if tool_name in allowed:
                return sdk.PermissionResultAllow()
            return sdk.PermissionResultDeny(
                message=f"Tool '{tool_name}' is not in this stage's allowlist."
            )

        # Effort passthrough: explicit kwarg wins; otherwise config
        # anthropic_effort applies to deep-tier calls only. The tier comes from
        # the caller's deep flag — never from model-string equality, which
        # would leak effort to quick-tier calls whenever both tiers are
        # configured with the same model.
        if effort is None and deep:
            effort = self._config.get("anthropic_effort")

        kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "setting_sources": [],      # isolate from ~/.claude and project settings
            "permission_mode": "acceptEdits",
            # Primary built-in kill switch: [] disables ALL built-in tools at
            # the CLI level (MCP data-server tools mounted via mcp_servers /
            # allowed_tools are unaffected). disallowed_tools and can_use_tool
            # stay as redundant defense-in-depth layers behind it.
            "tools": [],
            "disallowed_tools": list(DISALLOWED_BUILTIN_TOOLS),
            "can_use_tool": can_use_tool,
            "cwd": str(self._cwd),
        }
        if tools_server is not None:
            kwargs["mcp_servers"] = {"data": tools_server}
            kwargs["allowed_tools"] = list(allowed_tools or ())
        if output_schema is not None:
            kwargs["output_format"] = {"type": "json_schema", "schema": output_schema}
        if effort is not None:
            kwargs["effort"] = effort
        return sdk.ClaudeAgentOptions(**kwargs)

    async def _run_once(
        self, role: str, prompt: str, *,
        system_prompt: str | None,
        model: str,
        tools_server: object | None,
        allowed_tools: list[str] | None,
        output_schema: dict | None,
        max_turns: int,
        effort: str | None,
        deep: bool,
    ) -> AgentResult:
        sdk = self._sdk
        options = self._build_options(
            system_prompt=system_prompt, model=model,
            tools_server=tools_server, allowed_tools=allowed_tools,
            output_schema=output_schema, max_turns=max_turns, effort=effort,
            deep=deep,
        )

        llm_calls = 0
        tool_calls = 0
        tool_call_log: list[str] = []
        result_message = None

        # Streaming-mode prompt: required by the SDK because options carry a
        # can_use_tool callback (a str prompt raises ValueError pre-spawn).
        #
        # The SDK's receive_messages() raises a plain Exception (not a typed
        # SDK error) when the CLI exits non-zero after emitting an error result.
        # The message text is "Claude Code returned an error result: <subtype>"
        # (e.g. "...result: success" when the CLI crashes with a contradictory
        # is_error=True/subtype=success payload — seen with the bundled CLI
        # when MCP data tools return empty responses).  This exception bypasses
        # the typed retryable tuple in run(), so we catch it here and re-raise
        # as CLIConnectionError so the retry loop in run() picks it up.
        try:
            async for message in sdk.query(prompt=_stream_prompt(prompt), options=options):
                if isinstance(message, sdk.AssistantMessage):
                    llm_calls += 1
                    for block in message.content:
                        if isinstance(block, sdk.ToolUseBlock):
                            tool_calls += 1
                            tool_call_log.append(
                                _format_tool_call_line(block.name, block.input)
                            )
                elif isinstance(message, sdk.ResultMessage):
                    result_message = message
        except (sdk.CLIConnectionError, sdk.ProcessError, sdk.CLIJSONDecodeError):
            raise  # already retryable — let run() handle them
        except Exception as exc:
            text = str(exc)
            if text.startswith("Claude Code returned an error result:"):
                # Re-raise as CLIConnectionError so run()'s retry loop treats
                # this as a transient transport failure.
                raise sdk.CLIConnectionError(text) from exc
            raise  # unexpected — propagate as-is

        if result_message is None:
            raise StageError(f"[{role}] query() ended without a ResultMessage")

        subtype = result_message.subtype
        if result_message.is_error and subtype != self.STRUCTURED_RETRY_SUBTYPE:
            detail = result_message.errors or result_message.result or ""
            if _is_rate_limit_shaped(result_message):
                status = getattr(result_message, "api_error_status", None)
                raise _RateLimitError(
                    f"[{role}] rate-limited (subtype={subtype}, "
                    f"status={status}): {detail}"
                )
            raise StageError(f"[{role}] query failed (subtype={subtype}): {detail}")

        structured = None
        if output_schema is not None and subtype != self.STRUCTURED_RETRY_SUBTYPE:
            so = result_message.structured_output
            structured = so if isinstance(so, dict) else None

        tokens_in, tokens_out = _sum_tokens(result_message.usage)
        usage = {
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
        num_turns = result_message.num_turns or 0
        logger.debug(
            "[%s] model=%s turns=%d llm_calls=%d tool_calls=%d tokens=%d/%d subtype=%s",
            role, model, num_turns, llm_calls, tool_calls, tokens_in, tokens_out, subtype,
        )
        return AgentResult(
            text=result_message.result or "",
            structured=structured,
            usage=usage,
            num_turns=num_turns,
            tool_call_log=tool_call_log,
        )


def get_client(config: dict) -> AgentClient:
    """Backend factory. The mock path never imports claude_agent_sdk."""
    backend = config.get("llm_backend", "sdk")
    if backend == "mock":
        from .mock import MockAgentClient  # lazy: keeps default pytest offline
        return MockAgentClient(config)
    if backend == "sdk":
        return SdkAgentClient(config)
    raise ValueError(f"Unknown llm_backend {backend!r}; expected 'sdk' or 'mock'")
