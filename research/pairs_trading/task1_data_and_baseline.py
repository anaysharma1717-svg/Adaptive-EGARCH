"""
task1_data_and_baseline.py
===========================
Pairs trading (GARCH-DCC hedge ratio project) -- Task 1: data + roll-discontinuity
report only. No cointegration testing, no hedge ratio construction here.

Three hand-picked pairs, scope fixed, no screening:
  GC=F / SI=F   (gold / silver)
  CL=F / NG=F   (crude oil / natural gas)
  ES=F / YM=F   (S&P 500 / Dow e-mini futures)

Conventions matched to the rest of this repo (see research/extended_model_zoo.py):
  - yfinance for data, UTC-naive tz-stripped index, flattened MultiIndex columns.
  - logging module for progress, matplotlib not needed here.
  - Caching (new to this repo, but matches the .gitignore's existing `data/` and
    `plots/` exclusions): raw OHLCV cached to research/pairs_trading/data/*.csv;
    if a cache file exists it's reused as-is (pass --refresh to force re-download).

Cleaning:
  - Reindexed onto a business-day calendar (this both drops any stray weekend rows
    AND makes missing trading days visible as NaN rows, which is required before a
    gap-limited forward-fill means anything).
  - OHLC forward-filled with limit=2 (at most a 2-day-old price carried forward);
    longer gaps are left as NaN, not filled -- downstream windows spanning them
    should be skipped, not silently patched over.
  - Volume is NOT forward-filled (a carried-forward volume would misrepresent
    trading activity on a day with none) -- left as NaN on filled days.

Roll-discontinuity flagging (report only -- nothing is adjusted or deleted here):
  A day t is flagged if |log_return_t| > 5 * trailing_std_t, where trailing_std_t
  is the rolling standard deviation of log returns over the prior 90 trading days
  (shifted by 1, i.e. computed from data strictly BEFORE t -- so a huge return on
  day t cannot inflate its own reference distribution and mask itself).
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")

# --- Pair scope decision (post-Task-1-DATA-review) -------------------------
# Originally three hand-picked pairs were in scope. CL=F/NG=F is DROPPED as of
# this decision: CL=F went negative on 2020-04-20/21 (WTI's COVID storage-glut
# settlement, see task1_invalid_log_returns.csv), which makes log(CL=F) and
# therefore any log-space cointegration test, hedge ratio, or spread for this
# pair undefined across that window. Patching it (e.g. clipping, shifting the
# series, or excluding just those two days) would be an unprincipled judgment
# call with no principled way to prove it doesn't also distort the hedge ratio
# estimated over the surrounding window. Decision: drop the pair, not patch it.
# CL=F and NG=F stay in TICKERS below so their leg-level diagnostics (this is
# the evidence for the decision) remain intact in every rerun; they are simply
# excluded from ACTIVE_PAIRS, which is what all downstream pair-construction
# work (cointegration/hedge ratio/spread, in task1_step2_pair_construction.py)
# is scoped to.
# Task 1B (extend the pre-specified pair list): 5 new tickers, same pipeline.
TASK1B_PAIRS = [
    ("ZB=F", "ZN=F"),   # 30Y / 10Y Treasury futures -- curve
    ("ES=F", "NQ=F"),   # S&P 500 / Nasdaq 100 e-mini -- large cap vs tech
    ("ES=F", "RTY=F"),  # S&P 500 / Russell 2000 e-mini -- large cap vs small cap
    ("NQ=F", "RTY=F"),  # Nasdaq 100 / Russell 2000 e-mini -- tech vs small cap
    ("GC=F", "HG=F"),   # gold / copper -- precious vs industrial metals
    ("GC=F", "SI=F"),   # gold / silver -- repeat of Task 1, for comparison
]

PAIRS = [
    ("GC=F", "SI=F"),   # gold / silver -- kept, but see NOT_VIABLE_PAIRS below
    ("CL=F", "NG=F"),   # crude oil / natural gas -- DROPPED, see DROPPED_PAIRS
    ("ES=F", "YM=F"),   # S&P 500 / Dow e-mini futures -- DROPPED, see DROPPED_PAIRS
]
DROPPED_PAIRS = {
    ("CL=F", "NG=F"): (
        "CL=F close is negative on 2020-04-20 (-37.63) and 2020-04-21's prior "
        "close is that same negative value (see task1_invalid_log_returns.csv). "
        "log-space cointegration/hedge-ratio/spread is undefined across this "
        "window; any patch (clip/shift/drop-two-days) would be an unprincipled "
        "judgment call. Pair dropped entirely, not patched."
    ),
    ("ES=F", "YM=F"): (
        "Fails the tradeability bar found in Task 1's pipeline: OU half-life "
        "125.1 days (outside the 5-60d tradeable band), full-sample ADF "
        "p=0.057 (not stationary at 5%), zero cointegrated refit windows since "
        "late 2019, and the hedge ratio drifted monotonically from beta=1.15 "
        "(2015) to beta=0.81 (2026) rather than mean-reverting. Not a data "
        "issue -- a statistical result. Dropped from all further pair work; "
        "not included in Task 1B."
    ),
}
# Not dropped, but flagged: also failed the tradeability bar (half-life 90.5d,
# outside 5-60d) even though it IS cointegrated in some windows and full-sample
# stationary. Kept per explicit instruction "for comparison" -- report its
# numbers, but do not present it as a viable trading pair as-is.
NOT_VIABLE_PAIRS = {
    ("GC=F", "SI=F"): "OU half-life 90.5 days, outside the 5-60d tradeable band."
}
ACTIVE_PAIRS = [p for p in PAIRS if p not in DROPPED_PAIRS]
TICKERS = sorted({t for pair in (PAIRS + TASK1B_PAIRS) for t in pair})

START = "2015-01-01"
FFILL_LIMIT = 2
ROLL_WINDOW = 90
ROLL_MIN_PERIODS = 30
ROLL_Z = 5.0


def _cache_path(ticker: str) -> str:
    safe = ticker.replace("=", "_").replace("^", "")
    return os.path.join(DATA_DIR, f"{safe}.csv")


def fetch_and_cache(ticker: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _cache_path(ticker)
    if os.path.exists(path) and not refresh:
        log.info(f"{ticker}: loading cached {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    log.info(f"{ticker}: downloading from yfinance ({start} -> {end})...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        log.error(f"{ticker}: no data returned.")
        return df
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(path)
    log.info(f"{ticker}: cached {len(df)} rows -> {path}")
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reindex onto a business-day calendar, ffill(limit=2) OHLC only.
    Returns (cleaned_df, report_dict)."""
    if raw.empty:
        return raw, {}

    raw_first, raw_last = raw.index.min(), raw.index.max()
    raw_rows = len(raw)

    bdays = pd.bdate_range(raw_first, raw_last)
    reindexed = raw.reindex(bdays)

    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in reindexed.columns]
    cleaned = reindexed.copy()
    cleaned[ohlc_cols] = cleaned[ohlc_cols].ffill(limit=FFILL_LIMIT)
    # Volume intentionally NOT forward-filled -- see module docstring.

    nan_rows_close = int(cleaned["Close"].isna().sum()) if "Close" in cleaned.columns else -1

    report = {
        "raw_first_date": raw_first.date().isoformat(),
        "raw_last_date": raw_last.date().isoformat(),
        "raw_rows": raw_rows,
        "bday_calendar_rows": len(bdays),
        "cleaned_rows": len(cleaned),
        "remaining_nan_close": nan_rows_close,
    }
    return cleaned, report


def detect_invalid_log_returns(cleaned: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Days where close<=0 or prior close<=0 make log(close/close.shift(1)) undefined
    (NaN). These are NOT roll discontinuities -- they are a distinct failure mode
    (log-return math itself breaks) and are invisible to detect_roll_discontinuities
    below, since `NaN > threshold` is False in pandas, so a flagged==False day here
    would otherwise vanish from the report silently. Reported separately, on purpose."""
    close = cleaned["Close"]
    prev_close = close.shift(1)
    invalid = (close <= 0) | (prev_close <= 0)
    invalid = invalid.fillna(False)

    out = pd.DataFrame({
        "ticker": ticker,
        "date": cleaned.index,
        "close": close.values,
        "prev_close": prev_close.values,
    })
    out = out.loc[invalid.values].reset_index(drop=True)
    return out


def detect_roll_discontinuities(cleaned: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = cleaned["Close"]
    log_ret = np.log(close / close.shift(1))
    trailing_std = log_ret.rolling(ROLL_WINDOW, min_periods=ROLL_MIN_PERIODS).std().shift(1)
    threshold = ROLL_Z * trailing_std
    # NaN log_ret (from a non-positive close or prior close) can never exceed
    # threshold, so those days are excluded here by construction -- they are
    # reported instead by detect_invalid_log_returns() above.
    flagged = log_ret.abs() > threshold

    out = pd.DataFrame({
        "ticker": ticker,
        "date": cleaned.index,
        "log_return": log_ret.values,
        "trailing_std_90d": trailing_std.values,
        "abs_z": (log_ret.abs() / trailing_std).values,
    })
    out = out.loc[flagged.values].reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser(description="Pairs trading Task 1: data + roll-discontinuity report")
    p.add_argument("--refresh", action="store_true", help="force re-download, ignore cache")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    reports = {}
    cleaned_data = {}
    all_flags = []
    all_invalid = []

    for ticker in TICKERS:
        raw = fetch_and_cache(ticker, START, end, refresh=args.refresh)
        if raw.empty:
            reports[ticker] = {"error": "no data returned"}
            continue
        cleaned, report = clean(raw)
        cleaned_data[ticker] = cleaned
        reports[ticker] = report

        invalid = detect_invalid_log_returns(cleaned, ticker)
        all_invalid.append(invalid)

        flags = detect_roll_discontinuities(cleaned, ticker)
        all_flags.append(flags)

    # ── Report: date range, row counts, NaNs remaining ──────────────────────
    print("\n" + "=" * 100)
    print("  DATA REPORT")
    print("=" * 100)
    rep_df = pd.DataFrame(reports).T
    print(rep_df.to_string())
    rep_df.to_csv(os.path.join(RESULTS_DIR, "task1_data_report.csv"))

    # ── Report: invalid log returns (non-positive close on day t or t-1) ────
    invalid_df = pd.concat(all_invalid, ignore_index=True) if all_invalid else pd.DataFrame()
    print("\n" + "=" * 100)
    print("  INVALID LOG RETURNS  (close<=0 or prior close<=0 -- log() undefined)")
    print("=" * 100)
    if len(invalid_df):
        print(f"  Total: {len(invalid_df)} day(s). These are EXCLUDED from the roll-discontinuity")
        print("  list below by construction (NaN log_return can never exceed the z threshold),")
        print("  so they are reported here separately to avoid a silent drop.\n")
        print(invalid_df.to_string(index=False))
        print()
    else:
        print("  None.")
    invalid_df.to_csv(os.path.join(RESULTS_DIR, "task1_invalid_log_returns.csv"), index=False)

    # ── Report: roll-discontinuity candidates ────────────────────────────────
    flags_df = pd.concat(all_flags, ignore_index=True) if all_flags else pd.DataFrame()
    print("\n" + "=" * 100)
    print(f"  ROLL-DISCONTINUITY CANDIDATES  (|log return| > {ROLL_Z:.0f} trailing std, {ROLL_WINDOW}d window)")
    print("=" * 100)
    if len(flags_df):
        print(f"  Total flagged days across all {len(TICKERS)} tickers: {len(flags_df)}\n")
        for ticker in TICKERS:
            sub = flags_df[flags_df["ticker"] == ticker].sort_values("date")
            print(f"  --- {ticker} ({len(sub)} flagged) ---")
            if len(sub):
                print(sub.to_string(index=False))
            print()
    else:
        print("  None flagged.")
    flags_df.to_csv(os.path.join(RESULTS_DIR, "task1_roll_discontinuities.csv"), index=False)

    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'task1_data_report.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1_invalid_log_returns.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1_roll_discontinuities.csv')}")
    print("\nNo adjustment or deletion applied to any flagged/invalid day -- report only, per instructions.")

    # ── Pair scope decision ──────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  PAIR SCOPE DECISION")
    print("=" * 100)
    scope_rows = []
    all_pairs_seen = list(dict.fromkeys(PAIRS + TASK1B_PAIRS))  # de-duped, order-preserving
    for pair in all_pairs_seen:
        if pair in DROPPED_PAIRS:
            status, reason = "DROPPED", DROPPED_PAIRS[pair]
        elif pair in NOT_VIABLE_PAIRS:
            status, reason = "ACTIVE (not viable as-is)", NOT_VIABLE_PAIRS[pair]
        else:
            status, reason = "ACTIVE", ""
        print(f"  {pair[0]}/{pair[1]:<6} -> {status}" + (f"  -- {reason}" if reason else ""))
        scope_rows.append({"pair": f"{pair[0]}/{pair[1]}", "status": status, "reason": reason})
    pd.DataFrame(scope_rows).to_csv(os.path.join(RESULTS_DIR, "task1_pair_scope_decision.csv"), index=False)
    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'task1_pair_scope_decision.csv')}")
    print("Active pairs proceed to task1_step2_pair_construction.py (cointegration/hedge ratio/spread).")


if __name__ == "__main__":
    main()
