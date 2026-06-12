# tradingagents-cc — Design Contract (frozen)

Port of the TradingAgents pipeline to the **Claude Agent SDK** (Python), authenticated via the
user's **Claude subscription** (`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`), never a
metered `ANTHROPIC_API_KEY`. Plain asyncio driver — Python owns 100% of control flow; one
`claude_agent_sdk.query()` per pipeline node. Keyless data (yfinance) by default.

Full design: `.design/final-design.json` (architecture, file plan, risks).
Audit references: `.design/audit-graph.json`, `audit-agents.json`, `audit-dataflows.json`,
`audit-config-llm.json`, `audit-cli.json`, `audit-blockers.json`, `audit-agent-sdk-research.json`.

## Verified environment (Wave 0, do not re-verify)

- Workspace: uv 0.11.11, Python 3.13, `.venv` at repo root `C:\Users\randl\Documents\GitHub\TradingAgents\.venv`.
- `claude-agent-sdk==0.2.97` installed. Confirmed surface:
  - `ClaudeAgentOptions` HAS: `model`, `system_prompt`, `max_turns`, `allowed_tools`,
    `disallowed_tools`, `can_use_tool`, `permission_mode`, `mcp_servers`, `setting_sources`,
    `output_format`, `effort`, `fallback_model`, `max_budget_usd`, `env`, `cwd`.
  - `ResultMessage` fields: `subtype`, `is_error`, `num_turns`, `usage`, `result`,
    `structured_output`, `total_cost_usd`, `stop_reason`, `errors`, ...
  - `AssistantMessage` fields: `content` (blocks incl. `ToolUseBlock`), `model`, `usage`, ...
  - `tool(name: str, description: str, input_schema: type|dict, annotations: mcp.types.ToolAnnotations|None)`
    decorating `async def handler(args: dict) -> dict` returning `{"content":[{"type":"text","text":...}]}`
    (+ `"is_error": True` on failure — NEVER raise from a handler; it kills the whole query()).
  - `create_sdk_mcp_server(name, version='1.0.0', tools=[...])` → value for `options.mcp_servers={'data': server}`.
  - `ToolAnnotations` is the pydantic model from `mcp.types` (use `ToolAnnotations(readOnlyHint=True)`).
  - Structured output: `output_format={"type": "json_schema", "schema": <model_json_schema()>}`;
    success → `ResultMessage.structured_output` dict; exhaustion → `subtype == "error_max_structured_output_retries"`.
  - Exceptions: `CLIConnectionError`, `CLINotFoundError`, `ProcessError`, `CLIJSONDecodeError` (all importable).
- Parent import surface (verified LangChain-free): `tradingagents.dataflows.interface.route_to_vendor`,
  `tradingagents.dataflows.config.set_config`, `tradingagents.dataflows.utils.safe_ticker_component`.
  NEVER import anything from `tradingagents.agents.*` (its `__init__` eagerly imports LangChain).

## Frozen interfaces (every module codes against these exactly)

### client.py seam

```python
@dataclass
class AgentResult:
    text: str                      # ResultMessage.result or "" — free-text answer
    structured: dict | None       # structured_output when schema requested & valid, else None
    usage: dict                    # {"llm_calls": int, "tool_calls": int, "tokens_in": int, "tokens_out": int}
    num_turns: int
    tool_call_log: list[str]       # "HH:MM:SS [Tool Call] name(args)" lines

class AgentClient(Protocol):
    async def run(
        self, role: str, prompt: str, *,
        system_prompt: str | None = None,
        model: str,
        tools_server: object | None = None,     # McpSdkServerConfig — mounted as mcp_servers={"data": ...}
        allowed_tools: list[str] | None = None,  # fully-qualified mcp__data__* names only
        output_schema: dict | None = None,       # JSON schema → output_format json_schema
        max_turns: int = 1,
    ) -> AgentResult: ...

def get_client(config: dict) -> AgentClient   # "mock" backend never imports claude_agent_sdk
class AuthError(Exception); class StageError(Exception)
```

`role` values: `market_analyst`, `social_analyst`, `news_analyst`, `fundamentals_analyst`,
`bull`, `bear`, `research_manager`, `trader`, `aggressive`, `conservative`, `neutral`,
`portfolio_manager`, `reflector`. (Mock keys canned outputs off these.)

SdkAgentClient construction: `os.environ.pop("ANTHROPIC_API_KEY")` (log warning if present);
require `CLAUDE_CODE_OAUTH_TOKEN` else raise `AuthError` with remediation "run: claude setup-token".
Every options object: `setting_sources=[]`, `permission_mode="acceptEdits"`,
`disallowed_tools=["Bash","Read","Write","Edit","Glob","Grep","WebSearch","WebFetch","Task","NotebookEdit","TodoWrite","WebView"]`,
`can_use_tool` deny-by-default callback (allow only names in this call's allowed_tools).
Retry wrapper: 3 attempts, backoff 5s/15s/45s + jitter on CLIConnectionError/ProcessError/429-shaped errors.
Structured-output exhaustion → return `structured=None` (caller does one free-text fallback call).

### pipeline.py

```python
class TradingAgentsPipeline:
    def __init__(self, config: dict, client: AgentClient): ...
    async def propagate(self, ticker: str, trade_date: str) -> tuple[dict, str]:
        """Returns (final_state, signal) — signal ∈ Buy/Overweight/Hold/Underweight/Sell."""
```

### state.py — field names are the parent's, verbatim

`AgentState`: `company_of_interest`, `trade_date`, `sender`, `market_report`, `sentiment_report`,
`news_report`, `fundamentals_report`, `investment_debate_state`, `investment_plan`,
`trader_investment_plan`, `risk_debate_state`, `final_trade_decision`, `past_context`.
`InvestDebateState`: `bull_history`, `bear_history`, `history`, `current_response`,
`judge_decision`, `count`. `RiskDebateState`: `aggressive_history`, `conservative_history`,
`neutral_history`, `history`, `latest_speaker`, `current_aggressive_response`,
`current_conservative_response`, `current_neutral_response`, `judge_decision`, `count`.

### checkpointer.py — stage names (exact strings in `completed`)

`"Market Analyst"`, `"Social Analyst"`, `"News Analyst"`, `"Fundamentals Analyst"`,
`"Investment Debate"`, `"Research Manager"`, `"Trader"`, `"Risk Debate"`, `"Portfolio Manager"`.
Loops ("Investment Debate"/"Risk Debate") checkpoint state after EVERY turn (counts persist)
but join `completed` only when the loop exits; resume re-enters the loop and the verbatim
conditional logic picks the next speaker from `current_response`/`latest_speaker`.
`thread_id = sha256(f"{TICKER}:{date}").hexdigest()[:16]`; file
`{data_cache_dir}/cc_checkpoints/{safe_ticker}/{thread_id}.json`; atomic temp + `os.replace`.

### tools_data.py — ANALYST_TOOLSETS (fully-qualified)

- market: `mcp__data__get_stock_data`, `mcp__data__get_indicators`
- social: `mcp__data__get_news`
- news: `mcp__data__get_news`, `mcp__data__get_global_news` (+ `mcp__data__get_insider_transactions` when `bind_insider_to_news`)
- fundamentals: `mcp__data__get_fundamentals`, `mcp__data__get_balance_sheet`, `mcp__data__get_cashflow`, `mcp__data__get_income_statement`

Handlers: original names/signatures, body = `await asyncio.to_thread(route_to_vendor, ...)`,
try/except → `{"is_error": True}`, `ToolAnnotations(readOnlyHint=True)`.

### default_config.py — config keys

`quick_think_llm="claude-sonnet-4-6"`, `deep_think_llm="claude-opus-4-8"`, `llm_backend="sdk"|"mock"`
(env `TRADINGAGENTS_CC_MOCK=1` forces mock), `max_debate_rounds=1`, `max_risk_discuss_rounds=1`,
`max_analyst_turns=12`, `selected_analysts=["market","social","news","fundamentals"]`,
`output_language="English"`, `checkpoint_enabled=True`, `results_dir`/`data_cache_dir`/`memory_log_path`
(honor `TRADINGAGENTS_RESULTS_DIR`/`TRADINGAGENTS_CACHE_DIR`/`TRADINGAGENTS_MEMORY_LOG_PATH`,
defaults under `~/.tradingagents`), `memory_log_max_entries=None`, `data_vendors` (all `"yfinance"`),
`tool_vendors={}`, `bind_insider_to_news=False`, `anthropic_effort=None`, retry knobs,
`to_parent_config()` helper feeding `tradingagents.dataflows.config.set_config`.

## Model tiers (parity with parent quick/deep split)

quick (`quick_think_llm`): 4 analysts, bull, bear, trader, 3 risk debaters, reflector.
deep (`deep_think_llm`): Research Manager, Portfolio Manager only.

## Hard invariants

1. Default `pytest` consumes ZERO network and ZERO subscription credit (mock backend; conftest
   deletes both auth env vars; socket guard).
2. Mock mode never imports `claude_agent_sdk`; SDK client class only constructed when backend="sdk".
3. Debate semantics verbatim: Bull first; after each turn `count >= 2*max_debate_rounds` → done,
   else Bear if `current_response.startswith("Bull")` else Bull. Risk: Aggressive first;
   `count >= 3*max_risk_discuss_rounds` → done; rotation Aggressive→Conservative→Neutral keyed on
   `latest_speaker` prefix.
4. Structured stages (Research Manager / Trader / Portfolio Manager): structured attempt, then on
   `structured=None` exactly ONE free-text retry with the same prompt (parity with parent
   `invoke_structured_or_freetext`); flag `fallbacks_used` in decisions.jsonl.
5. Renderers: trader output always ends `FINAL TRANSACTION PROPOSAL: **{ACTION}**`; PM output always
   contains `**Rating**: {X}`; signal via vendored two-pass `parse_rating`, default "Hold", no LLM.
6. Memory log byte-format identical to parent `TradingMemoryLog` (interoperable file).
7. All file paths through `safe_ticker_component`; atomic writes via temp + `os.replace`.
8. Windows: never install a Selector event-loop policy (SDK needs Proactor); absolute paths in all
   ops scripts; file logging (Task Scheduler swallows stdout); `PYTHONUTF8=1` in wrappers.
