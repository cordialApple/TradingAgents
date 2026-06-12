# Ported from cli/main.py — the Typer/Rich/questionary questionnaire becomes plain argparse
# flags (restoring the strip().upper() ticker normalization the parent CLI dropped).
"""argparse command-line interface for tradingagents-cc.

Subcommands:

    run TICKER [--date YYYY-MM-DD] [--depth 1|3|5] [--analysts LIST] ...
        One full analysis. --date defaults to today; future dates are rejected.
    resume TICKER --date YYYY-MM-DD
        Identical execution path to ``run`` — checkpointing makes it resume:
        completed stages are skipped and debate loops re-enter mid-flight.
    routine [--dry-run]
        The unattended daily driver (delegates to ``tradingagents_cc.routine``).
    clear-checkpoints
        Delete every saved checkpoint under ``{data_cache_dir}/cc_checkpoints``.
    show-decisions [-n 10]
        Pretty-print the tail of ``{results_dir}/decisions.jsonl``.

``--mock`` vs ``--dry-run``: both run the offline mock backend (canned outputs,
``claude_agent_sdk`` is never imported, no subscription credit) and both still
write the full artifact set — reports, decisions.jsonl, checkpoints.
``--dry-run`` is the plumbing-check name: reach for it when verifying wiring,
paths, or scheduling end to end; ``--mock`` says what it does.

Exit codes: 0 success, 1 failure (bad value, failed stage, I/O error), 2 auth
(``CLAUDE_CODE_OAUTH_TOKEN`` missing/unusable — run: claude setup-token).
argparse usage errors (unknown flag, missing subcommand) exit with the
conventional status 2 before any run starts.

Windows note: no event-loop policy is installed here — the SDK's subprocess
transport needs the default Proactor loop (design invariant 8).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from .checkpointer import RunCheckpointer
from .client import AuthError, StageError, get_client
from .default_config import VALID_ANALYSTS, load_config
from .pipeline import TradingAgentsPipeline
from .state import normalize_ticker, safe_ticker_component

__all__ = ["main"]


def _parse_trade_date(value: str) -> str:
    """Validate/normalize a --date value; reject malformed or future dates."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"--date must be YYYY-MM-DD, got {value!r}") from None
    if parsed > date.today():
        raise ValueError(
            f"--date {parsed.isoformat()} is in the future; analyses run on today or past dates"
        )
    return parsed.isoformat()  # zero-pads e.g. 2026-6-1 -> 2026-06-01


def _parse_analysts(value: str) -> list[str]:
    """Comma-separated analyst list -> canonical-order subset of VALID_ANALYSTS."""
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("--analysts must name at least one analyst")
    unknown = sorted(set(items) - set(VALID_ANALYSTS))
    if unknown:
        raise ValueError(
            f"unknown analysts {unknown}; valid choices: {list(VALID_ANALYSTS)}"
        )
    if len(set(items)) != len(items):
        raise ValueError(f"--analysts contains duplicates: {value!r}")
    # Parent CLI parity: selection is re-ordered to the canonical pipeline order.
    return [analyst for analyst in VALID_ANALYSTS if analyst in items]


def _build_overrides(args: argparse.Namespace) -> dict:
    """Map run/resume flags onto config keys for load_config()."""
    overrides: dict = {}
    if args.depth is not None:
        # Depth preset sets BOTH round knobs (parent CLI select_research_depth).
        overrides["max_debate_rounds"] = args.depth
        overrides["max_risk_discuss_rounds"] = args.depth
    if args.analysts is not None:
        overrides["selected_analysts"] = _parse_analysts(args.analysts)
    if args.language is not None:
        language = args.language.strip()
        if not language:
            raise ValueError("--language must be a non-empty string")
        overrides["output_language"] = language
    for flag, key, model in (
        ("--quick-model", "quick_think_llm", args.quick_model),
        ("--deep-model", "deep_think_llm", args.deep_model),
    ):
        if model is not None:
            if not model.strip():
                raise ValueError(f"{flag} must be a non-empty model name")
            overrides[key] = model.strip()
    if args.mock or args.dry_run:
        overrides["llm_backend"] = "mock"
    if args.no_checkpoint:
        overrides["checkpoint_enabled"] = False
    return overrides


def _cmd_run(args: argparse.Namespace) -> int:
    """`run` and `resume`: one propagate() per invocation, checkpoint-aware."""
    ticker = safe_ticker_component(normalize_ticker(args.ticker))
    trade_date = (
        _parse_trade_date(args.date) if args.date is not None else date.today().isoformat()
    )
    config = load_config(_build_overrides(args))
    client = get_client(config)  # AuthError here -> exit 2 in main()
    pipeline = TradingAgentsPipeline(config, client)
    _final_state, signal = asyncio.run(pipeline.propagate(ticker, trade_date))
    print(f"Signal for {ticker} on {trade_date}: {signal}")
    print(f"Report directory: {Path(config['results_dir']) / ticker / trade_date}")
    return 0


def _cmd_routine(args: argparse.Namespace) -> int:
    """Delegate to the unattended daily driver (lazy: it wires file logging)."""
    from . import routine

    code = routine.main(["--dry-run"] if args.dry_run else [])
    return 0 if code is None else int(code)


def _cmd_clear_checkpoints(args: argparse.Namespace) -> int:
    config = load_config()
    count = RunCheckpointer(config["data_cache_dir"]).clear_all()
    print(
        f"Removed {count} checkpoint(s) under "
        f"{Path(config['data_cache_dir']) / 'cc_checkpoints'}"
    )
    return 0


def _format_decision(record: dict) -> str:
    """One decisions.jsonl record -> a readable multi-line block."""
    lines = [
        f"{record.get('ts') or '?'}  {record.get('ticker') or '?'} "
        f"{record.get('trade_date') or '?'} -> "
        f"{record.get('decision') or '(no decision)'} [{record.get('status') or '?'}]"
    ]
    models = record.get("models") or {}
    if models.get("quick") or models.get("deep"):
        lines.append(f"  models: quick={models.get('quick')} deep={models.get('deep')}")
    stats = record.get("stats") or {}
    bits = " ".join(
        f"{key}={stats.get(key, 0)}"
        for key in ("llm_calls", "tool_calls", "tokens_in", "tokens_out")
    )
    if record.get("duration_s") is not None:
        bits += f" duration={record['duration_s']}s"
    lines.append(f"  stats: {bits}")
    if record.get("fallbacks_used"):
        lines.append(f"  fallbacks: {', '.join(record['fallbacks_used'])}")
    if record.get("error"):
        lines.append(f"  error: {record['error']}")
    if record.get("report_dir"):
        lines.append(f"  report: {record['report_dir']}")
    return "\n".join(lines)


def _cmd_show_decisions(args: argparse.Namespace) -> int:
    if args.n < 1:
        raise ValueError("-n must be a positive integer")
    config = load_config()
    path = Path(config["results_dir"]) / "decisions.jsonl"
    if not path.exists():
        print(f"No decisions recorded yet ({path} does not exist).")
        return 0

    records: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                skipped += 1

    if not records:
        print(f"No decisions recorded yet ({path} is empty).")
    else:
        print("\n\n".join(_format_decision(r) for r in records[-args.n:]))
    if skipped:
        print(f"(skipped {skipped} malformed line(s) in {path})", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tradingagents-cc",
        description=(
            "TradingAgents on the Claude Agent SDK — subscription auth, "
            "keyless yfinance data, resumable checkpointed runs."
        ),
        epilog=(
            "exit codes: 0 success, 1 failure, 2 auth "
            "(CLAUDE_CODE_OAUTH_TOKEN missing — run: claude setup-token)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # Flags shared by run and resume (resume is run; checkpointing makes it resume).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "ticker",
        help="ticker symbol, exchange suffixes allowed (e.g. SPY, CNC.TO, 7203.T); "
        "normalized to strip().upper()",
    )
    common.add_argument(
        "--depth", type=int, choices=(1, 3, 5), default=None,
        help="research depth preset (1=shallow, 3=medium, 5=deep); sets BOTH "
        "max_debate_rounds and max_risk_discuss_rounds",
    )
    common.add_argument(
        "--analysts", default=None, metavar="LIST",
        help=f"comma-separated subset of {','.join(VALID_ANALYSTS)} "
        "(re-ordered to canonical pipeline order; default: all four)",
    )
    common.add_argument(
        "--language", default=None, metavar="LANG",
        help="output language for reports and the final decision "
        "(internal agent debate stays in English)",
    )
    common.add_argument(
        "--quick-model", default=None, metavar="MODEL",
        help="override quick_think_llm (analysts, bull/bear, trader, risk debaters, reflector)",
    )
    common.add_argument(
        "--deep-model", default=None, metavar="MODEL",
        help="override deep_think_llm (Research Manager and Portfolio Manager)",
    )
    common.add_argument(
        "--mock", action="store_true",
        help="offline mock backend: canned outputs, never imports claude_agent_sdk, "
        "spends no subscription credit; still writes the full artifact set",
    )
    common.add_argument(
        "--dry-run", action="store_true",
        help="alias of --mock under the plumbing-check name: same mock backend and the "
        "same full artifacts (reports, decisions.jsonl, checkpoints) — use it to "
        "verify wiring, paths, and scheduling end to end",
    )
    common.add_argument(
        "--no-checkpoint", action="store_true",
        help="disable checkpointing (the run can neither resume nor be resumed)",
    )

    run_parser = subparsers.add_parser(
        "run", parents=[common],
        help="run one full analysis for TICKER",
        description="Run one full multi-agent analysis. Re-running the same "
        "ticker+date resumes from the saved checkpoint automatically.",
    )
    run_parser.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="trade date (default: today; future dates rejected)",
    )
    run_parser.set_defaults(handler=_cmd_run)

    resume_parser = subparsers.add_parser(
        "resume", parents=[common],
        help="resume an interrupted run for TICKER on --date",
        description="Identical to `run` — checkpointing makes it resume: completed "
        "stages are skipped and debate loops re-enter mid-flight. With no "
        "checkpoint on disk it simply starts fresh.",
    )
    resume_parser.add_argument(
        "--date", required=True, metavar="YYYY-MM-DD",
        help="trade date of the run to resume",
    )
    resume_parser.set_defaults(handler=_cmd_run)

    routine_parser = subparsers.add_parser(
        "routine",
        help="run the unattended daily routine (config/routine.toml)",
        description="Delegates to tradingagents_cc.routine — the Task Scheduler "
        "driver with file logging, single-instance lock, and trading-day guard.",
    )
    routine_parser.add_argument(
        "--dry-run", action="store_true",
        help="run the routine on the mock backend (full artifacts, no subscription credit)",
    )
    routine_parser.set_defaults(handler=_cmd_routine)

    clear_parser = subparsers.add_parser(
        "clear-checkpoints",
        help="delete every saved checkpoint",
        description="Delete every checkpoint under {data_cache_dir}/cc_checkpoints.",
    )
    clear_parser.set_defaults(handler=_cmd_clear_checkpoints)

    show_parser = subparsers.add_parser(
        "show-decisions",
        help="pretty-print the tail of decisions.jsonl",
        description="Pretty-print the most recent records from {results_dir}/decisions.jsonl.",
    )
    show_parser.add_argument(
        "-n", type=int, default=10, metavar="N",
        help="number of most recent decisions to show (default: 10)",
    )
    show_parser.set_defaults(handler=_cmd_show_decisions)

    return parser


def _configure_logging() -> None:
    """Console handler; package at INFO so stage progress shows, libraries stay quiet."""
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("tradingagents_cc").setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging()
    try:
        code = args.handler(args)
    except AuthError as exc:
        print(f"Auth error: {exc}", file=sys.stderr)
        code = 2
    except (StageError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
