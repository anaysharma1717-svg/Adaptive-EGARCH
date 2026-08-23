"""
task1b_extended_pairs.py
==========================
Task 1B: extend the pre-specified pair list. Same Task 1 pipeline (data
quality, invalid log returns, leg-level 5-sigma flags, rolling Engle-Granger
cointegration, log-space hedge ratio, spread, spread-level 5-sigma
discontinuity check with leg overlap, OU half-life, full-sample ADF, ex-ante
rolling z-score) run UNCHANGED on 6 pre-specified pairs. This is a fixed,
economically-motivated list, not a screen -- nothing here adds pairs,
generates combinations, or selects by correlation.

  ZB=F / ZN=F    30Y / 10Y Treasury futures -- Treasury curve
  ES=F / NQ=F    S&P 500 / Nasdaq 100 e-mini -- large cap vs tech
  ES=F / RTY=F   S&P 500 / Russell 2000 e-mini -- large cap vs small cap
  NQ=F / RTY=F   Nasdaq 100 / Russell 2000 e-mini -- tech vs small cap
  GC=F / HG=F    gold / copper -- precious vs industrial metals
  GC=F / SI=F    gold / silver -- repeat of Task 1, included for comparison

Scope note: ES=F/YM=F was DROPPED after Task 1 (half-life 125.1d, outside the
5-60d tradeable band; ADF p=0.057, not stationary; zero cointegrated windows
since late 2019; hedge ratio drifted monotonically beta=1.15->0.81) -- see
task1_data_and_baseline.py's DROPPED_PAIRS. It is not part of this list and
is not rerun here. GC=F/SI=F is kept per instruction "for comparison", but
Task 1 already found it not viable as-is (half-life 90.5d, also outside
5-60d) -- see NOT_VIABLE_PAIRS in the same file.

Data-quality note on the 5 new tickers (from the Task 1 data report, unchanged
here): ZB=F and ZN=F both have FULL 2015-01-02 -> 2026-08-21 history with 0
remaining NaN after cleaning -- no patchiness observed in this pull, contrary
to the a-priori expectation that Treasury futures data would be shorter/dirtier.
The actual short-history ticker is RTY=F: yfinance only returns data from
2017-07-10 (2296 rows vs ~3036 for the others). This shortens the usable
window -- and therefore MIN_TRAIN=252's first possible refit date -- for both
pairs that include RTY=F (ES=F/RTY=F, NQ=F/RTY=F).

CONSTRAINTS -- identical to Task 1, none retuned here:
  MIN_TRAIN=252 (expanding window), REFIT_FREQ=21 trading days,
  cointegration p-threshold=0.05, tradeability half-life band=[5,60] days,
  spread-discontinuity detector=5-sigma on 90d trailing std of spread diff.
  Every estimate for day t uses data through t-1 only (see
  task1_step2_pair_construction.py's no-look-ahead convention, reused
  unmodified here).

No DCC, no dynamic (time-varying) hedge ratio beyond the periodic-refit
expanding-window OLS already used in Task 1, no backtest, no trading rule --
none of that is built here, per instruction.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task1_data_and_baseline import RESULTS_DIR
from task1_step2_pair_construction import (
    load_pair_prices, rolling_cointegration_and_hedge_ratio,
    detect_spread_discontinuities, leg_flag_overlap, ou_half_life,
    adf_full_sample, rolling_zscore, plot_pair,
    REFIT_FREQ, MIN_TRAIN, ZSCORE_WINDOW,
)

PAIRS_TASK1B = [
    ("ZB=F", "ZN=F"),
    ("ES=F", "NQ=F"),
    ("ES=F", "RTY=F"),
    ("NQ=F", "RTY=F"),
    ("GC=F", "HG=F"),
    ("GC=F", "SI=F"),
]

HALF_LIFE_MIN, HALF_LIFE_MAX = 5, 60
COINT_PASS_RATE_MIN = 0.30
ADF_P_MAX = 0.05
LAST_N_YEARS = 2


def coint_pass_rate(fit_df: pd.DataFrame, since=None):
    d = fit_df if since is None else fit_df[fit_df["refit_date"] >= since]
    if len(d) == 0:
        return np.nan, 0
    valid = d["eg_cointegrated_5pct"].dropna()
    if len(valid) == 0:
        return np.nan, 0
    return float(valid.mean()), len(valid)


def main():
    end_ts = pd.Timestamp.today()
    end = end_ts.strftime("%Y-%m-%d")
    since_2y = end_ts - pd.DateOffset(years=LAST_N_YEARS)

    leg_flags_all = pd.read_csv(os.path.join(RESULTS_DIR, "task1_roll_discontinuities.csv"), parse_dates=["date"])

    summary_rows = []
    all_fit_rows, all_spread_rows, all_flag_rows = [], [], []

    for tick_x, tick_y in PAIRS_TASK1B:
        pair_label = f"{tick_x}/{tick_y}"
        print("\n" + "=" * 100)
        print(f"  PAIR: {pair_label}")
        print("=" * 100)

        df = load_pair_prices(tick_x, tick_y, end)
        print(f"  Aligned rows: {len(df)}   Range: {df.index.min().date()} -> {df.index.max().date()}")
        if len(df) < MIN_TRAIN + REFIT_FREQ:
            print(f"  WARNING: only {len(df)} aligned rows -- too little history for even one refit "
                  f"(need >= {MIN_TRAIN + REFIT_FREQ}). Skipping pair-level fit.")
            summary_rows.append({
                "pair": pair_label, "coint_pass_rate": np.nan, "coint_pass_rate_last_2y": np.nan,
                "half_life_days": np.nan, "adf_pvalue": np.nan,
                "hedge_ratio_min": np.nan, "hedge_ratio_max": np.nan,
                "spread_discontinuity_count": np.nan,
            })
            continue

        fit_df, spread_df = rolling_cointegration_and_hedge_ratio(df)
        fit_df.insert(0, "pair", pair_label)
        all_fit_rows.append(fit_df)

        pass_all, n_all = coint_pass_rate(fit_df)
        pass_2y, n_2y = coint_pass_rate(fit_df, since=since_2y)
        print(f"  Rolling Engle-Granger cointegration: {len(fit_df)} refits, "
              f"pass rate (full)={pass_all:.1%} (N={n_all}), "
              f"pass rate (last {LAST_N_YEARS}y)={pass_2y:.1%} (N={n_2y})" if not np.isnan(pass_all) else
              f"  Rolling Engle-Granger cointegration: no valid refits")

        beta_min, beta_max = float(fit_df["beta"].min()), float(fit_df["beta"].max())
        print(f"  Hedge ratio (beta) range across refits: {beta_min:.4f} -> {beta_max:.4f}")

        spread_flags = detect_spread_discontinuities(spread_df, pair_label)
        spread_flags.insert(0, "pair_group", "task1b")
        all_flag_rows.append(spread_flags)
        overlap = leg_flag_overlap(spread_flags, leg_flags_all, tick_x, tick_y)
        print(f"  Spread-level discontinuities (5-sigma): {len(spread_flags)}  "
              f"(exact-date overlap with leg flags: {overlap['n_exact_date_overlap']}/{overlap['n_spread_flags']})")

        ou = ou_half_life(spread_df["spread"])
        adf_res = adf_full_sample(spread_df["spread"])
        print(f"  OU half-life: {ou['ou_half_life_days']:.2f} days (mean_reverting={ou['mean_reverting']})   "
              f"ADF: stat={adf_res['adf_stat']:.4f} p={adf_res['adf_pvalue']:.4f} "
              f"stationary_5pct={adf_res['stationary_5pct']}")

        z = rolling_zscore(spread_df["spread"])
        spread_df = spread_df.copy()
        spread_df["zscore"] = z
        spread_df.insert(0, "pair", pair_label)
        all_spread_rows.append(spread_df.reset_index())

        plot_path = plot_pair(pair_label, spread_df, z, spread_flags)
        print(f"  Saved plot: {plot_path}")

        summary_rows.append({
            "pair": pair_label,
            "coint_pass_rate": pass_all,
            "coint_pass_rate_last_2y": pass_2y,
            "half_life_days": ou["ou_half_life_days"],
            "adf_pvalue": adf_res["adf_pvalue"],
            "hedge_ratio_min": beta_min,
            "hedge_ratio_max": beta_max,
            "spread_discontinuity_count": len(spread_flags),
        })

    # --- save combined CSVs ---
    if all_fit_rows:
        pd.concat(all_fit_rows, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "task1b_hedge_ratio_history.csv"), index=False)
    if all_spread_rows:
        pd.concat(all_spread_rows, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "task1b_spread_series.csv"), index=False)
    if all_flag_rows:
        pd.concat(all_flag_rows, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "task1b_spread_discontinuities.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)

    def meets_all_4(row):
        if row[["half_life_days", "coint_pass_rate", "coint_pass_rate_last_2y", "adf_pvalue"]].isna().any():
            return False
        return bool(
            HALF_LIFE_MIN <= row["half_life_days"] <= HALF_LIFE_MAX
            and row["coint_pass_rate"] > COINT_PASS_RATE_MIN
            and row["coint_pass_rate_last_2y"] > 0.0
            and row["adf_pvalue"] < ADF_P_MAX
        )

    summary_df["meets_all_4_criteria"] = summary_df.apply(meets_all_4, axis=1)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "task1b_pair_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("  TASK 1B SUMMARY -- all 6 pairs (unranked)")
    print("=" * 100)
    display_df = summary_df.copy()
    display_df["hedge_ratio_range"] = display_df.apply(
        lambda r: f"{r['hedge_ratio_min']:.3f} - {r['hedge_ratio_max']:.3f}"
        if pd.notna(r["hedge_ratio_min"]) else "n/a", axis=1)
    cols = ["pair", "coint_pass_rate", "coint_pass_rate_last_2y", "half_life_days",
            "adf_pvalue", "hedge_ratio_range", "spread_discontinuity_count"]
    print(display_df[cols].to_string(index=False))

    print(f"\nCriteria applied (fixed in advance, not retuned): "
          f"half-life in [{HALF_LIFE_MIN},{HALF_LIFE_MAX}]d, "
          f"cointegration pass rate > {COINT_PASS_RATE_MIN:.0%}, "
          f"cointegrated in >=1 window in the last {LAST_N_YEARS}y, "
          f"ADF p < {ADF_P_MAX}")

    qualifying = summary_df.loc[summary_df["meets_all_4_criteria"], "pair"].tolist()
    print("\n" + "=" * 100)
    if qualifying:
        print(f"  PAIRS MEETING ALL 4 CRITERIA: {qualifying}")
    else:
        print("  PAIRS MEETING ALL 4 CRITERIA: NONE.")
        print("  Zero of the 6 pre-specified pairs pass the fixed thresholds. Thresholds were")
        print("  not loosened to produce a survivor -- this is the reported finding.")
    print("=" * 100)

    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'task1b_pair_summary.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1b_hedge_ratio_history.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1b_spread_series.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1b_spread_discontinuities.csv')}")


if __name__ == "__main__":
    main()
