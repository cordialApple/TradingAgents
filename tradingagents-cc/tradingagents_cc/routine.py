"""Unattended daily driver for Windows Task Scheduler (and any cron-alike).

``python -m tradingagents_cc.routine [--dry-run] [--config PATH]`` runs the
full pipeline once per configured ticker, sequentially, with everything an
unattended run needs and an interactive one takes for granted:

1.  Dated file log under ``{results_dir}/logs/routine_YYYYMMDD.log`` (Task
    Scheduler swallows stdout) plus a stdout stream for interactive use.
2.  Auth preflight (skipped in dry-run): constructs the client up front so a
    missing/expired ``CLAUDE_CODE_OAUTH_TOKEN`` exits 2 with a remediation
    line *before* taking the lock or touching the network.
3.  Single-instance lock ``{results_dir}/.routine.lock`` (PID inside) so a
    Task Scheduler misfire can never overlap runs — overlap would race the
    shared memory-log file. Stale-lock rule documented at ``_lock_is_stale``.
4.  Weekday guard, then a trading-day check (today's local date must appear
    in SPY's recent yfinance rows — the task fires at 18:30, after US close)
    so weekends/holidays exit 0 without spending a single token.
5.  Declarative config from ``config/routine.toml`` (tickers, depth,
    analysts, model overrides, output language, checkpointing).
6.  Per-ticker exception isolation: one ticker failing (its progress is
    checkpointed and a failed decisions.jsonl row written by the pipeline)
    never stops the remaining tickers.
7.  A monthly usage tally summed from this month's decisions.jsonl rows, so
    subscription-credit burn stays visible in the log.

``--dry-run`` forces the mock backend and skips the auth preflight and the
weekday/trading-day guards: a free end-to-end plumbing check (full artifact
tree, lock, logging) for validating the Task Scheduler wiring.

Exit codes: 0 = all tickers ok, or a clean skip (weekend / market closed /
another instance holds the lock); 1 = partial failure (at least one ticker
failed); 2 = fatal (auth, malformed routine config, unexpected error).

Windows note: never install a Selector event-loop policy here — the Agent
SDK's subprocess transport needs the default Proactor loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from .client import AuthError, get_client
from .default_config import load_config, load_routine_config
from .pipeline import TradingAgentsPipeline

logger = logging.getLogger(__name__)

__all__ = ["main"]

# <repo>/config/routine.toml — the package dir's parent (source checkout layout).
DEFAULT_ROUTINE_TOML = Path(__file__).resolve().parent.parent / "config" / "routine.toml"

# A routine run is hard-capped at 2h by the scheduled task; a lock older than
# 6h cannot belong to a live legitimate run.
LOCK_STALE_AGE_S = 6 * 60 * 60

_USAGE_KEYS = ("llm_calls", "tool_calls", "tokens_in", "tokens_out")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents_cc.routine",
        description="Unattended daily TradingAgents run (one pipeline pass per configured ticker).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="mock backend, no auth preflight, no market-day guards — free plumbing check",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_ROUTINE_TOML,
        metavar="PATH",
        help=f"routine TOML file (default: {DEFAULT_ROUTINE_TOML})",
    )
    return parser.parse_args(argv)


# --- Logging -----------------------------------------------------------------


def _setup_logging(results_dir: Path, now: datetime) -> list[logging.Handler]:
    """Attach a dated file handler + stdout handler to the root logger.

    Root-level so pipeline/client/reflection loggers all land in the same
    file. Handlers are returned for removal in main()'s finally, keeping
    repeated main() calls (tests) from stacking duplicates.
    """
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"routine_{now.strftime('%Y%m%d')}.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return handlers


def _teardown_logging(handlers: list[logging.Handler]) -> None:
    root = logging.getLogger()
    for handler in handlers:
        root.removeHandler(handler)
        handler.close()


# --- Single-instance lock ------------------------------------------------------


def _pid_running(pid: int) -> bool:
    """Best-effort "is this PID alive?" probe; returns True when uncertain.

    Windows: ``os.kill(pid, 0)`` is NOT a probe there — any non-CTRL signal
    routes through TerminateProcess and would kill a live lock holder — so a
    query-only OpenProcess handle is used instead. POSIX: signal 0 is the
    standard existence check.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # Access denied = the process exists but is protected; anything
            # else (invalid parameter) = no such process.
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return True  # cannot query — keep respecting the lock
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # e.g. EPERM: exists but owned by someone else
    return True


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        return int(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _lock_is_stale(lock_path: Path, st: os.stat_result | None = None) -> bool:
    """Stale = the holder is provably gone, or the lock is ancient.

    Documented psutil-free rule: a lock younger than 6h is treated as live
    unless its PID is provably not running; at 6h or older it is stale
    regardless of the PID — the scheduled task kills runs at 2h and PIDs
    recycle, so the age bound beats an unreliable PID match.

    ``st`` is an optional caller-captured stat snapshot of the lock file so
    the staleness verdict and the subsequent atomic break
    (:func:`_break_stale_lock`) are keyed to the same observation; omitted, a
    fresh stat is taken.
    """
    if st is None:
        try:
            st = lock_path.stat()
        except OSError:
            return False  # vanished mid-check: the holder is racing us — live
    age = time.time() - st.st_mtime
    if age >= LOCK_STALE_AGE_S:
        return True
    pid = _read_lock_pid(lock_path)
    if pid is None:
        return False  # young and unreadable: assume a holder mid-write
    return not _pid_running(pid)


def _same_lock_identity(a: os.stat_result, b: os.stat_result) -> bool:
    """True when two stat snapshots plausibly describe the same lock file.

    Prefers (st_dev, st_ino). On Windows a path-based ``os.stat`` can report
    ``st_ino == 0`` (the inode is only reliable when stat'd via an open
    handle), so when either inode is 0 the comparison falls back to
    st_mtime_ns + st_size + st_ctime_ns.
    """
    if a.st_ino and b.st_ino:
        return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    return (
        a.st_mtime_ns == b.st_mtime_ns
        and a.st_size == b.st_size
        and a.st_ctime_ns == b.st_ctime_ns
    )


def _break_stale_lock(lock_path: Path, snapshot: os.stat_result) -> bool:
    """Atomically retire a stale lock file; True only if WE retired it.

    Re-stats the lock and confirms (via :func:`_same_lock_identity`) it is
    still the exact file ``snapshot`` judged stale — if a new holder replaced
    it in the meantime, back off. The break itself is
    ``os.replace(lock, lock.with_suffix(".stale"))``: rename is atomic, so
    exactly one racer's replace can move the file; every loser gets
    FileNotFoundError (already moved), or PermissionError/FileExistsError
    (Windows, destination in use) and backs off. The .stale file is deleted
    best-effort after a successful break.
    """
    try:
        current = os.stat(lock_path)
    except OSError:
        return False  # vanished: someone else broke/released it first
    if not _same_lock_identity(snapshot, current):
        return False  # a new (live) holder re-created the lock — respect it
    stale_path = lock_path.with_suffix(".stale")
    try:
        os.replace(lock_path, stale_path)
    except (FileNotFoundError, FileExistsError, PermissionError):
        return False  # lost the break race — exactly one racer may win
    try:
        stale_path.unlink()
    except OSError:
        pass  # best-effort cleanup; a future break's os.replace overwrites it
    return True


def _acquire_lock(results_dir: Path) -> Path | None:
    """Atomically create {results_dir}/.routine.lock with our PID, or None.

    One stale-break retry: when the existing lock is judged stale (see
    :func:`_lock_is_stale`, against a captured stat snapshot), it is retired
    atomically by :func:`_break_stale_lock` — an identity-checked
    ``os.replace`` to ``.routine.stale`` that exactly one racer can win, so
    two processes can never both break the same stale lock and run
    concurrently. Only the winner re-attempts creation (exactly once); losing
    the break race, or the re-creation race, means another live instance —
    back off (None).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    lock_path = results_dir / ".routine.lock"
    for attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt == 0:
                try:
                    snapshot = os.stat(lock_path)
                except OSError:
                    return None  # vanished mid-check: the holder is racing us
                if _lock_is_stale(lock_path, snapshot):
                    logger.warning(
                        "Breaking stale lock %s (pid=%s)",
                        lock_path, _read_lock_pid(lock_path),
                    )
                    if _break_stale_lock(lock_path, snapshot):
                        continue
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return lock_path
    return None


def _release_lock(lock_path: Path) -> None:
    try:
        if _read_lock_pid(lock_path) == os.getpid():
            lock_path.unlink()
        else:
            # Only possible if another process (wrongly) broke our lock.
            logger.warning("Lock %s is no longer ours at release; leaving it.", lock_path)
    except OSError as exc:
        logger.warning("Could not release lock %s: %s", lock_path, exc)


# --- Market guards -------------------------------------------------------------


def _is_trading_day(trade_date: str) -> bool | None:
    """True/False = SPY traded on trade_date or not; None = check unavailable.

    One tiny yfinance history call against the same keyless source the data
    tools use. ``None`` (network/API failure) fails OPEN with a warning in the
    caller — a transient check outage must not silently skip a real trading
    day, and downstream data failures already degrade to error-string reports
    instead of aborting.
    """
    try:
        rows = yf.Ticker("SPY").history(period="5d")
        if rows is None or len(rows) == 0:
            return None
        dates = {idx.strftime("%Y-%m-%d") for idx in rows.index}
        return trade_date in dates
    except Exception as exc:
        logger.warning("SPY trading-day check failed: %s", exc)
        return None


# --- Monthly usage tally ---------------------------------------------------------


def _log_monthly_usage(results_dir: Path) -> None:
    """Sum this month's decisions.jsonl stats so credit burn is visible.

    Months keyed on the rows' UTC ``ts`` prefix (append_decision writes UTC
    ISO timestamps). Malformed lines are skipped, never fatal.
    """
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    totals = dict.fromkeys(_USAGE_KEYS, 0)
    runs = 0
    path = results_dir / "decisions.jsonl"
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not str(row.get("ts") or "").startswith(month):
                    continue
                runs += 1
                stats = row.get("stats") or {}
                for key in _USAGE_KEYS:
                    value = stats.get(key)
                    if isinstance(value, int):
                        totals[key] += value
    except FileNotFoundError:
        logger.info("Monthly usage %s: no decisions.jsonl yet", month)
        return
    except OSError as exc:
        logger.warning("Monthly usage tally unavailable (%s): %s", path, exc)
        return
    logger.info(
        "Monthly usage %s: %d runs, llm_calls=%d, tool_calls=%d, tokens_in=%d, tokens_out=%d",
        month, runs, totals["llm_calls"], totals["tool_calls"],
        totals["tokens_in"], totals["tokens_out"],
    )


# --- Driver ---------------------------------------------------------------------


async def _run_tickers(
    pipeline: TradingAgentsPipeline, tickers: list[str], trade_date: str
) -> list[str]:
    """Run propagate() per ticker sequentially; return the tickers that failed.

    Sequential on purpose: protects the single memory-log file's atomic
    rewrites and throttles yfinance. Any exception (StageError after the
    pipeline checkpointed and wrote its failed decisions row, bad ticker
    ValueError, anything unexpected) is isolated to its ticker.
    """
    failures: list[str] = []
    for ticker in tickers:
        logger.info("=== %s @ %s ===", ticker, trade_date)
        started = time.monotonic()
        try:
            _state, signal = await pipeline.propagate(ticker, trade_date)
        except Exception:
            logger.exception("Ticker %s failed; continuing with the rest.", ticker)
            failures.append(ticker)
        else:
            logger.info("%s -> %s (%.1fs)", ticker, signal, time.monotonic() - started)
    return failures


def _run_locked(args: argparse.Namespace, results_dir: Path, now: datetime) -> int:
    """Everything that happens while holding the single-instance lock."""
    trade_date = now.strftime("%Y-%m-%d")

    if args.dry_run:
        logger.info("Dry run: skipping market-day guards; forcing the mock backend.")
    else:
        if now.weekday() >= 5:  # Saturday/Sunday
            logger.info("Weekend (%s) — market closed, nothing to do.", now.strftime("%A"))
            return 0
        open_today = _is_trading_day(trade_date)
        if open_today is False:
            logger.info("Market closed on %s (no SPY row) — nothing to do.", trade_date)
            return 0
        if open_today is None:
            logger.warning("Trading-day check unavailable; proceeding anyway.")

    try:
        routine = load_routine_config(args.config)
    except FileNotFoundError:
        logger.error("Routine config not found: %s (pass --config PATH)", args.config)
        return 2
    except ValueError as exc:
        logger.error("Routine config invalid (%s): %s", args.config, exc)
        return 2

    overrides = dict(routine["overrides"])
    if args.dry_run:
        overrides["llm_backend"] = "mock"
    config = load_config(overrides)
    tickers: list[str] = routine["tickers"]
    logger.info(
        "Tickers: %s | analysts=%s | rounds=%d/%d | quick=%s deep=%s | backend=%s",
        ", ".join(tickers), ",".join(config["selected_analysts"]),
        config["max_debate_rounds"], config["max_risk_discuss_rounds"],
        config["quick_think_llm"], config["deep_think_llm"], config["llm_backend"],
    )

    # The pipeline client is built from the merged config (the preflight one
    # used pre-toml defaults and is discarded; construction is local and cheap).
    try:
        client = get_client(config)
    except AuthError as exc:
        logger.error("Auth failed: %s", exc)
        logger.error("Remediation: re-run `claude setup-token` and update the stored token.")
        return 2
    pipeline = TradingAgentsPipeline(config, client)

    # One asyncio.run() per invocation; default (Proactor) loop policy untouched.
    failures = asyncio.run(_run_tickers(pipeline, tickers, trade_date))

    _log_monthly_usage(results_dir)

    if failures:
        logger.error(
            "Partial failure: %d/%d tickers failed: %s",
            len(failures), len(tickers), ", ".join(failures),
        )
        return 1
    logger.info("All %d ticker(s) completed.", len(tickers))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    now = datetime.now()
    results_dir = Path(load_config()["results_dir"])  # toml cannot override paths

    handlers = _setup_logging(results_dir, now)
    try:
        logger.info(
            "Routine starting: date=%s dry_run=%s config=%s pid=%d",
            now.strftime("%Y-%m-%d"), args.dry_run, args.config, os.getpid(),
        )

        # Auth preflight: fail fast (exit 2, distinct remediation line for the
        # Task Scheduler history) before locking or any network access.
        if not args.dry_run:
            try:
                get_client(load_config())
            except AuthError as exc:
                logger.error("Auth preflight failed: %s", exc)
                logger.error(
                    "Remediation: re-run `claude setup-token` and update the stored token."
                )
                return 2

        lock_path = _acquire_lock(results_dir)
        if lock_path is None:
            logger.info(
                "Another routine instance holds %s — skipping this run.",
                results_dir / ".routine.lock",
            )
            return 0
        try:
            return _run_locked(args, results_dir, now)
        finally:
            _release_lock(lock_path)
    except Exception:
        logger.exception("Fatal error in routine")
        return 2
    finally:
        _teardown_logging(handlers)


if __name__ == "__main__":
    sys.exit(main())
