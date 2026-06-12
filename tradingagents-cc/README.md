# tradingagents-cc

TradingAgents multi-agent trading pipeline running on the **Claude Agent SDK** (Python),
authenticated with your **Claude subscription** — zero API keys. A plain-asyncio driver makes one
`claude_agent_sdk.query()` call per pipeline stage: 4 analysts (market / social / news /
fundamentals, each with read-only yfinance tools), a bull/bear investment debate, Research
Manager, Trader, a 3-way risk debate, and a Portfolio Manager that emits a 5-tier rating
(Buy / Overweight / Hold / Underweight / Sell). Designed for unattended daily runs via Windows
Task Scheduler.

## How it differs from the parent

- **No LangChain / LangGraph.** The graph is replaced by explicit Python control flow in
  `pipeline.py`; the model never orchestrates anything. Per-analyst tool loops use the SDK's
  agentic loop; debate routing is the parent's conditional logic ported verbatim.
- **Same prompts and semantics.** Prompts, debate-turn counts (`2 * max_debate_rounds`,
  `3 * max_risk_discuss_rounds`), speaker rotation, model tiering, structured-output-then-freetext
  fallback, report markers (`FINAL TRANSACTION PROPOSAL: **X**`, `**Rating**: X`), artifact
  layout, and the memory-log file format are all preserved. Small parent modules (memory, rating,
  schemas, prompts) are vendored with source citations because the parent gates them behind a
  LangChain-importing package `__init__`.
- **Subscription auth only.** `ANTHROPIC_API_KEY` is removed from the environment at client
  construction; metered billing is impossible by design.
- **Keyless data.** All market data comes from yfinance via the parent's `tradingagents.dataflows`
  package (the only parent import surface — verified LangChain-free). Alpha Vantage remains
  opt-in exactly as upstream (set a vendor to `"alpha_vantage"` and export
  `ALPHA_VANTAGE_API_KEY`); it is never required.
- **Always-on checkpointing.** Each stage writes a JSON checkpoint; a crashed or killed run
  resumes from the last completed stage on the next invocation.

## Prerequisites

- Windows 10/11 (the routine scripts are PowerShell; the pipeline itself is portable).
- [uv](https://docs.astral.sh/uv/) and Python 3.13 (package supports >= 3.10; the workspace pins 3.13).
- Claude Code installed and logged in with a **Pro or Max** subscription (needed once, to mint the
  OAuth token; the SDK also spawns the Claude Code CLI at run time).

Install from the **workspace root** (this repo's top level, not `tradingagents-cc/`):

```powershell
cd C:\Users\randl\Documents\GitHub\TradingAgents
uv sync --all-packages
```

This resolves both workspace members (`tradingagents` + `tradingagents-cc`) into the shared
`.venv` and installs the `tradingagents-cc` console script.

## Authentication

One-time setup:

```powershell
claude setup-token            # interactive; prints a CLAUDE_CODE_OAUTH_TOKEN
.\tradingagents-cc\scripts\register_task.ps1   # prompts for the token, stores it, registers the task
```

`register_task.ps1` stores the token in **Windows Credential Manager** (primary; avoids
plaintext) or, with `-PersistUserEnv`, as a User-scope environment variable. The SDK picks up
`CLAUDE_CODE_OAUTH_TOKEN` automatically — no login session or browser at run time. For manual
runs in a fresh shell, set it yourself:

```powershell
$env:CLAUDE_CODE_OAUTH_TOKEN = "<token from claude setup-token>"
```

Notes, deliberate behaviors:

- **`ANTHROPIC_API_KEY` is popped from the environment** when the SDK client is constructed (a
  warning is logged if it was set). Runs can only draw from your subscription, never metered
  billing.
- **Terms of service:** the OAuth token is licensed exclusively for Claude Code / Agent SDK use,
  is personal, and is non-poolable for team use. A personal daily research routine like this one
  qualifies.
- **Credit pool:** from **2026-06-15**, headless Agent SDK runs draw from a dedicated monthly
  credit pool — **Pro $20 / Max 5x $100 / Max 20x $200** — separate from interactive Claude Code
  usage. A depth-1 run is ~13–16 LLM calls (≈ $0.30–0.80-equivalent at the default
  sonnet-quick/opus-deep tiers); 1–3 tickers daily at depth 1 fits comfortably on Pro. Per-run
  usage lands in `decisions.jsonl` and the routine logs a monthly cumulative tally so headroom
  stays visible.

## Quick start

```powershell
# 1. Offline mock run — no network, no credit, validates install
uv run tradingagents-cc run NVDA --mock

# 2. Dry run — mock backend but writes the full artifact tree (plumbing check)
uv run tradingagents-cc run NVDA --dry-run

# 3. Cheap live smoke — real subscription call, smallest sensible models/depth
uv run tradingagents-cc run SPY --depth 1 --quick-model claude-haiku-4-5 --deep-model claude-sonnet-4-6
```

A minimal programmatic example lives in `main.py` (mock by default, `--live` opt-in).

## CLI reference

Console script `tradingagents-cc`, installed into the workspace `.venv`. Tickers are normalized
with `strip().upper()` and validated before any path use.

```text
tradingagents-cc run TICKER [options]
    --date YYYY-MM-DD     Trade date (default: today; future dates rejected).
    --depth {1,3,5}       Sets BOTH debate-round knobs (1=Shallow, 3=Medium, 5=Deep).
    --analysts LIST       Comma-separated subset of market,social,news,fundamentals.
    --language LANG       Output language for analyst reports + final decision.
    --quick-model NAME    Override quick_think_llm (analysts, debaters, trader, reflector).
    --deep-model NAME     Override deep_think_llm (Research Manager, Portfolio Manager).
    --mock                Offline mock backend; never imports claude_agent_sdk.
    --dry-run             Mock backend + full artifact writing (end-to-end plumbing check).
    --no-checkpoint       Disable stage checkpointing for this run.

tradingagents-cc resume TICKER --date YYYY-MM-DD    Resume from the saved checkpoint.
tradingagents-cc routine [--dry-run]                Run the unattended daily routine once.
tradingagents-cc clear-checkpoints                  Delete saved checkpoints.
tradingagents-cc show-decisions [-n 10]             Tail of decisions.jsonl.
```

## Daily routine and scheduling

`python -m tradingagents_cc.routine` (what the scheduled task runs) does, in order:

1. Preflight auth: assert `CLAUDE_CODE_OAUTH_TOKEN`, pop `ANTHROPIC_API_KEY`; failure → **exit 2**
   with a one-line remediation log (`re-run claude setup-token`).
2. Single-instance lock at `{results_dir}/.routine.lock` (PID + stale-lock detection — overlapping
   runs are impossible even if Task Scheduler misfires).
3. Weekday guard, then trading-day check (no SPY rows today via yfinance → log and **exit 0**).
4. Load `config/routine.toml`; run each ticker sequentially with per-ticker exception isolation.
5. Write the report tree + a `decisions.jsonl` row per ticker; log to a dated file under
   `{results_dir}/logs/` (Task Scheduler swallows stdout); log the monthly token tally.
6. Exit code: **0** ok/skip, **1** partial (some tickers failed), **2** auth/fatal.

### register_task.ps1

```powershell
.\tradingagents-cc\scripts\register_task.ps1 [-Time 18:30] [-TaskName TradingAgentsCC] [-DryRun] [-PersistUserEnv]
```

Registers a scheduled task that runs `scripts\run_daily.ps1` via
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File <absolute path>`, working directory =
repo root, trigger Mon–Fri at 18:30 local (after US close so same-day OHLCV exists), with
`-StartWhenAvailable` (catch-up after sleep), battery-friendly flags, `-MultipleInstances
IgnoreNew`, a 2-hour execution limit, and 2 restarts at 30-minute intervals. Choose "Run whether
user is logged on or not" if you want runs while logged out. `-DryRun` registers a one-shot task
that runs `routine --dry-run` — a free validation of the entire scheduler wiring.

`run_daily.ps1` loads the token from Credential Manager into the process environment, sets
`PYTHONUTF8=1`, invokes the workspace venv interpreter **directly by absolute path**
(`C:\...\TradingAgents\.venv\Scripts\python.exe -m tradingagents_cc.routine` — no venv
activation, no uv at run time), tees output to
`%LOCALAPPDATA%\tradingagents-cc\logs\routine_YYYYMMDD.log`, and propagates the exit code.

Verify after registering:

```powershell
Start-ScheduledTask -TaskName TradingAgentsCC
Get-ScheduledTaskInfo -TaskName TradingAgentsCC   # LastTaskResult: 0 ok/skip, 1 partial, 2 auth
Get-Content "$env:LOCALAPPDATA\tradingagents-cc\logs\routine_$(Get-Date -Format yyyyMMdd).log" -Tail 30
```

## Configuration reference

`tradingagents_cc/default_config.py` — every key, with defaults:

| Key | Default | Meaning |
|---|---|---|
| `llm_backend` | `"sdk"` | `"sdk"` or `"mock"`. `TRADINGAGENTS_CC_MOCK=1` forces mock and wins over any override. |
| `quick_think_llm` | `"claude-sonnet-4-6"` | Quick tier: 4 analysts, bull/bear, trader, 3 risk debaters, reflector. |
| `deep_think_llm` | `"claude-opus-4-8"` | Deep tier: Research Manager and Portfolio Manager only. |
| `anthropic_effort` | `None` | Optional effort passthrough (`"low"`/`"medium"`/`"high"`) for SDK calls. |
| `selected_analysts` | all four | Ordered subset of `market, social, news, fundamentals`. |
| `max_debate_rounds` | `1` | Investment debate ends at `count >= 2 * rounds`. |
| `max_risk_discuss_rounds` | `1` | Risk debate ends at `count >= 3 * rounds`. |
| `max_analyst_turns` | `12` | Tool-loop budget per analyst `query()`. |
| `bind_insider_to_news` | `False` | When true, the news analyst also gets `get_insider_transactions`. |
| `output_language` | `"English"` | Language of analyst reports + final decision (debate stays English). |
| `checkpoint_enabled` | `True` | Stage-level JSON checkpointing (resume on rerun). |
| `results_dir` | `~/.tradingagents/logs` | Env override `TRADINGAGENTS_RESULTS_DIR`. |
| `data_cache_dir` | `~/.tradingagents/cache` | Env override `TRADINGAGENTS_CACHE_DIR`. OHLCV CSV cache + checkpoints. |
| `memory_log_path` | `~/.tradingagents/memory/trading_memory.md` | Env override `TRADINGAGENTS_MEMORY_LOG_PATH`. |
| `memory_log_max_entries` | `None` | Optional rotation cap on resolved entries (pending never pruned). |
| `data_vendors` | all `"yfinance"` | Per-category vendor map fed to the parent dataflows layer. |
| `tool_vendors` | `{}` | Per-tool vendor overrides (take precedence over categories). |
| `retry_attempts` | `3` | SDK call retries on connection/process/429-shaped failures. |
| `retry_base_delay` | `5.0` | Backoff base: 5s/15s/45s + jitter. |

Path keys are shared with the parent project (same env overrides, same `~/.tradingagents` tree).

### config/routine.toml

Read only by the routine. All keys optional; unknown keys raise at load time so typos in an
unattended job fail loudly.

| Key | Default | Notes |
|---|---|---|
| `tickers` | `["SPY"]` | Processed sequentially; upper-cased on load. |
| `depth` | `1` | Sets both round knobs (1/3/5 = parent CLI Shallow/Medium/Deep). |
| `analysts` | all four | Duplicate-free subset, in run order. |
| `quick_model` / `deep_model` | unset | Override the two model tiers. |
| `output_language` | `"English"` | |
| `checkpoint_enabled` | `true` | |

## Testing

```powershell
uv run --project tradingagents-cc pytest
```

The default suite consumes **zero network and zero subscription credit** — guaranteed three ways:
`conftest.py` forces the mock backend, deletes both `CLAUDE_CODE_OAUTH_TOKEN` and
`ANTHROPIC_API_KEY` (an accidental real-SDK path fails loudly), and installs a socket guard. The
SDK client class is not even constructed unless `llm_backend="sdk"`.

One opt-in `live` marker performs a single tiny no-tool query (a few hundred tokens) to verify
subscription auth end to end:

```powershell
$env:TRADINGAGENTS_CC_LIVE = "1"; uv run --project tradingagents-cc pytest -m live
```

## Artifacts

Per run, under `{results_dir}` (default `~/.tradingagents/logs`):

```text
{results_dir}/
  decisions.jsonl                                  one JSON line per run
  {TICKER}/
    {date}/
      reports/{section}.md                         live tee, 7 canonical sections
      message_tool.log                             "HH:MM:SS [Tool Call] name(args)"
      complete_report.md                           consolidated "## I." .. "## V." report
    TradingAgentsStrategy_logs/
      full_states_log_{date}.json                  final state, parent-identical keys
```

The 7 sections (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`,
`investment_plan`, `trader_investment_plan`, `final_trade_decision`) are written live after each
stage, so a failed unattended run still leaves partial output for postmortem.

### decisions.jsonl (schema_version 1)

One record per run: `schema_version`, `run_id`, `ts` (UTC ISO), `ticker`, `trade_date`,
`decision` (5-tier signal), `rating`, `trader_action`, `status`, `error`,
`models {quick, deep}`, `stats {llm_calls, tool_calls, tokens_in, tokens_out}`,
`stage_timings`, `fallbacks_used` (stages where structured output fell back to free text —
watch this for drift), `report_dir`, `duration_s`. Tail it with
`tradingagents-cc show-decisions -n 10`.

### Memory log

`{memory_log_path}` is byte-format identical to the parent's `TradingMemoryLog`, so the two
projects can share one file: pending entries
`[{date} | {ticker} | {rating} | pending]` are appended after each run and resolved on the next
same-ticker run with 5-day raw/alpha returns (keyless yfinance vs SPY) plus a short reflection.
Single-process writes are atomic (temp + `os.replace`), but **concurrent writers are not safe** —
if you ever run the parent and this port at the same time, point the port at its own file:

```powershell
$env:TRADINGAGENTS_MEMORY_LOG_PATH = "$HOME\.tradingagents\memory\trading_memory_cc.md"
```

### Checkpoints

`{data_cache_dir}/cc_checkpoints/{TICKER}/{thread_id}.json`, `thread_id =
sha256("{TICKER}:{date}")[:16]` (parent-identical). Re-running the same ticker+date resumes after
the last completed stage (mid-debate resumes from persisted counts); cleared on success; removed
manually via `tradingagents-cc clear-checkpoints`.

## Troubleshooting

- **Exit code 2 / `AuthError`** — the OAuth token is missing, expired, or revoked (there is no
  auto-refresh). Re-run `claude setup-token`, then re-run `register_task.ps1` to store the new
  token. The failure is a distinct log line in the routine log and visible as result code 2 in
  Task Scheduler history.
- **Task ran, exit 0, but no reports** — that's the trading-day skip (weekend/holiday: no SPY
  rows for today). The routine log says so explicitly.
- **yfinance 429s or empty data** — Yahoo's unofficial endpoints throttle. The dataflows layer
  already retries 2s/4s/8s; persistent failures degrade to error-string report sections instead
  of aborting, and unresolved memory entries stay pending and self-heal on later runs. If it
  persists, lower the ticker count or move the scheduled time.
- **`NotImplementedError` from asyncio subprocess on Windows** — something installed a Selector
  event-loop policy. The SDK's subprocess transport requires the default **Proactor** loop; never
  set `WindowsSelectorEventLoopPolicy` in any process that runs the pipeline.
- **Moved the repo or recreated `.venv`** — the scheduled task and `run_daily.ps1` use absolute
  paths, so the task breaks *silently* until you check the wrapper log or Event Viewer. Re-run
  `register_task.ps1` (and `uv sync --all-packages`) after any move.
- **Where to look, in order:** `%LOCALAPPDATA%\tradingagents-cc\logs\routine_YYYYMMDD.log`
  (wrapper tee) → `{results_dir}/logs/` (Python routine log) → Task Scheduler → task → History
  tab (enable "Enable All Tasks History" if empty) → Event Viewer → Windows Logs → Application.
- **Stale `.routine.lock`** — the routine detects locks whose PID is gone and reclaims them; if a
  lock from a dead host genuinely wedges things, delete `{results_dir}/.routine.lock`.
- **Decisions drifting toward Hold** — check `fallbacks_used` in `decisions.jsonl`; persistent
  structured-output fallbacks mean the PM free-text output is being rescued by
  `parse_rating(default="Hold")`.
