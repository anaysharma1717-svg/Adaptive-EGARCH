"""
task1_step2_pair_construction.py
==================================
Pairs trading Task 1 (continued): rolling cointegration, log-space hedge
ratio, spread construction, spread-level discontinuity check, OU half-life,
ADF stationarity test, rolling z-score, plots, CSVs.

Scope: ACTIVE_PAIRS only (GC=F/SI=F, ES=F/YM=F). CL=F/NG=F was dropped in the
prior data-review step -- see task1_data_and_baseline.py's DROPPED_PAIRS and
results/task1_pair_scope_decision.csv for the reasoning (WTI's 2020-04-20
negative settlement price makes log-space work undefined for that pair). No
screening happens here: both remaining pairs run through the full pipeline
regardless of what the cointegration/ADF/half-life numbers turn out to be.

No-look-ahead convention (matches research/extended_model_zoo.py's walk-forward):
  - Hedge ratio (alpha, beta) is fit by EXPANDING-window OLS on log prices,
    refit every REFIT_FREQ trading days. On a refit day t, the fit uses only
    rows strictly BEFORE t; the fitted (alpha, beta) is then applied to price
    t and every day up to (not including) the next refit -- so no day's
    spread value is ever built from a hedge ratio that saw that day's price.
  - The rolling z-score's mean/std are computed on a trailing window and then
    shifted by 1 day, so z_t only uses spread history through t-1.
  - Cointegration (Engle-Granger via ADF on the fit residuals) is tested on
    the SAME expanding training window used for the hedge ratio at each
    refit, and reported as a time series -- it is a diagnostic, not a filter
    (scope is fixed at 2 pairs; nothing here adds or removes a pair).

Spread-level discontinuity detector: leg-level detection (in
task1_data_and_baseline.py) uses log RETURNS because price is a level.
Spread is already a log-price differential (logy - a - b*logx), so "log of
the spread" is undefined whenever the spread is <= 0 -- the same problem that
got CL=F dropped. The correct analog is the day-over-day CHANGE in the spread
(already in log-price units), z-scored against its own trailing 90-day
std -- same window and same 5-sigma multiplier as the leg-level detector, so
the two are comparable.

Nothing here is tuned after the fact: REFIT_FREQ, MIN_TRAIN, ZSCORE_WINDOW,
and the discontinuity threshold are fixed before either pair's results were
inspected, and are the same for both pairs.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task1_data_and_baseline import fetch_and_cache, clean, START, ACTIVE_PAIRS, RESULTS_DIR, HERE

PLOTS_DIR = os.path.join(HERE, "plots")

REFIT_FREQ = 21            # trading days between hedge-ratio refits (matches extended_model_zoo.py)
MIN_TRAIN = 252            # 1yr minimum expanding-window history before the first fit
ZSCORE_WINDOW = 63         # trailing ~1 quarter for rolling z-score, shifted by 1 (ex-ante)
SPREAD_ROLL_WINDOW = 90    # same window as the leg-level roll-discontinuity detector
SPREAD_ROLL_MIN_PERIODS = 30
SPREAD_ROLL_Z = 5.0


def load_pair_prices(tick_x: str, tick_y: str, end: str) -> pd.DataFrame:
    raw_x = fetch_and_cache(tick_x, START, end)
    raw_y = fetch_and_cache(tick_y, START, end)
    clean_x, _ = clean(raw_x)
    clean_y, _ = clean(raw_y)
    df = pd.DataFrame({"close_x": clean_x["Close"], "close_y": clean_y["Close"]})
    # Any remaining NaN means a gap > the 2-day ffill limit on at least one leg --
    # skip those rows entirely, per the original Task 1 data instruction.
    df = df.dropna()
    df["logx"] = np.log(df["close_x"])
    df["logy"] = np.log(df["close_y"])
    return df


def rolling_cointegration_and_hedge_ratio(df: pd.DataFrame):
    """Expanding-window OLS hedge ratio (logy ~ a + b*logx), refit every
    REFIT_FREQ days on data strictly before the refit day. Engle-Granger
    cointegration test (ADF on the in-sample fit residuals) reported at each
    refit. Returns (fit_history_df, spread_df) where spread_df carries the
    active (a, b) alongside the resulting spread for full audit-ability."""
    n = len(df)
    logx = df["logx"].values
    logy = df["logy"].values
    dates = df.index

    spread = np.full(n, np.nan)
    active_a_arr = np.full(n, np.nan)
    active_b_arr = np.full(n, np.nan)
    active_a, active_b = None, None
    fit_rows = []

    for t in range(n):
        is_refit_point = (t >= MIN_TRAIN) and ((t - MIN_TRAIN) % REFIT_FREQ == 0)
        if is_refit_point:
            X = np.column_stack([np.ones(t), logx[:t]])   # rows [0, t) -- strictly before t
            y = logy[:t]
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            resid = y - X @ coef
            adf_stat, adf_p = np.nan, np.nan
            try:
                res = adfuller(resid, autolag="AIC")
                adf_stat, adf_p = float(res[0]), float(res[1])
            except Exception as e:
                pass
            fit_rows.append({
                "refit_date": dates[t], "n_train": t,
                "alpha": a, "beta": b,
                "eg_adf_stat": adf_stat, "eg_adf_pvalue": adf_p,
                "eg_cointegrated_5pct": bool(adf_p < 0.05) if not np.isnan(adf_p) else None,
            })
            active_a, active_b = a, b

        if active_a is not None:
            active_a_arr[t] = active_a
            active_b_arr[t] = active_b
            spread[t] = logy[t] - active_a - active_b * logx[t]

    fit_df = pd.DataFrame(fit_rows)
    spread_df = pd.DataFrame({
        "date": dates, "logx": logx, "logy": logy,
        "alpha_active": active_a_arr, "beta_active": active_b_arr, "spread": spread,
    }).set_index("date")
    return fit_df, spread_df


def detect_spread_discontinuities(spread_df: pd.DataFrame, pair_label: str) -> pd.DataFrame:
    spread = spread_df["spread"]
    d = spread.diff()
    trailing_std = d.rolling(SPREAD_ROLL_WINDOW, min_periods=SPREAD_ROLL_MIN_PERIODS).std().shift(1)
    threshold = SPREAD_ROLL_Z * trailing_std
    flagged = (d.abs() > threshold).fillna(False)

    out = pd.DataFrame({
        "pair": pair_label,
        "date": spread.index,
        "spread_diff": d.values,
        "trailing_std_90d": trailing_std.values,
        "abs_z": (d.abs() / trailing_std).values,
    })
    return out.loc[flagged.values].reset_index(drop=True)


def leg_flag_overlap(spread_flags: pd.DataFrame, leg_flags_all: pd.DataFrame, tick_x: str, tick_y: str) -> dict:
    leg_dates = set(leg_flags_all.loc[leg_flags_all["ticker"].isin([tick_x, tick_y]), "date"])
    spread_dates = list(pd.to_datetime(spread_flags["date"]))
    leg_dates = set(pd.to_datetime(list(leg_dates)))

    exact_overlap = sum(1 for d in spread_dates if d in leg_dates)
    # secondary, non-tuned check: within +/-1 trading day (index position, not calendar day)
    near_overlap = 0
    sorted_leg = sorted(leg_dates)
    for d in spread_dates:
        if d in leg_dates:
            near_overlap += 1
            continue
        if any(abs((d - ld).days) <= 3 for ld in sorted_leg):  # covers weekends around a +/-1 bday
            near_overlap += 1

    return {
        "n_spread_flags": len(spread_flags),
        "n_leg_flags_this_pair": len(leg_dates),
        "n_exact_date_overlap": exact_overlap,
        "n_near_overlap_within_3cal_days": near_overlap,
    }


def ou_half_life(spread: pd.Series) -> dict:
    """Full-sample descriptive fit of the constructed (already ex-ante) spread
    series -- a summary statistic about the pair's mean-reversion speed, not a
    per-day forecast, so using the whole series here does not reintroduce
    look-ahead into anything upstream."""
    s = spread.dropna()
    y = s.diff().dropna()
    x = s.shift(1).reindex(y.index)
    X = np.column_stack([np.ones(len(x)), x.values])
    coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    half_life = float(-np.log(2) / b) if b < 0 else float("nan")
    return {"ou_a": a, "ou_b": b, "ou_half_life_days": half_life, "mean_reverting": bool(b < 0)}


def adf_full_sample(spread: pd.Series) -> dict:
    s = spread.dropna().values
    stat, p, *_ = adfuller(s, autolag="AIC")[:2]
    return {"adf_stat": float(stat), "adf_pvalue": float(p), "stationary_5pct": bool(p < 0.05)}


def rolling_zscore(spread: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    roll_mean = spread.rolling(window, min_periods=window).mean().shift(1)
    roll_std = spread.rolling(window, min_periods=window).std().shift(1)
    return (spread - roll_mean) / roll_std


def plot_pair(pair_label: str, spread_df: pd.DataFrame, z: pd.Series, spread_flags: pd.DataFrame):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(spread_df.index, spread_df["spread"], color="#1f77b4", lw=0.8, label="spread")
    if len(spread_flags):
        flag_dates = pd.to_datetime(spread_flags["date"])
        flag_vals = spread_df.loc[spread_df.index.isin(flag_dates), "spread"]
        ax1.scatter(flag_vals.index, flag_vals.values, color="red", s=25, zorder=5,
                    label=f"spread discontinuity (n={len(spread_flags)})")
    ax1.axhline(0, color="grey", lw=0.5, ls="--")
    ax1.set_title(f"{pair_label} -- log-space spread (expanding-window hedge ratio, refit every {REFIT_FREQ}d)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("spread (log units)")

    ax2.plot(z.index, z.values, color="#2ca02c", lw=0.7)
    ax2.axhline(0, color="grey", lw=0.5, ls="--")
    for lvl in (2, -2):
        ax2.axhline(lvl, color="orange", lw=0.5, ls=":")
    ax2.set_ylabel(f"z-score ({ZSCORE_WINDOW}d, ex-ante)")
    ax2.set_xlabel("date")

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, f"task1_{pair_label.replace('/', '_').replace('=', '')}_spread.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    leg_flags_all = pd.read_csv(os.path.join(RESULTS_DIR, "task1_roll_discontinuities.csv"), parse_dates=["date"])

    all_fit_rows = []
    all_spread_rows = []
    all_spread_flags = []
    diag_rows = []

    for tick_x, tick_y in ACTIVE_PAIRS:
        pair_label = f"{tick_x}/{tick_y}"
        print("\n" + "=" * 100)
        print(f"  PAIR: {pair_label}")
        print("=" * 100)

        df = load_pair_prices(tick_x, tick_y, end)
        print(f"  Aligned rows (both legs present, gaps<=2d ffilled elsewhere dropped): {len(df)}")
        print(f"  Range: {df.index.min().date()} -> {df.index.max().date()}")

        fit_df, spread_df = rolling_cointegration_and_hedge_ratio(df)
        fit_df.insert(0, "pair", pair_label)
        all_fit_rows.append(fit_df)

        n_coint = fit_df["eg_cointegrated_5pct"].sum() if len(fit_df) else 0
        print(f"\n  Rolling Engle-Granger cointegration test: {len(fit_df)} refits, "
              f"{int(n_coint)}/{len(fit_df)} windows cointegrated at 5% (ADF p<0.05)")
        print(fit_df[["refit_date", "n_train", "alpha", "beta", "eg_adf_stat", "eg_adf_pvalue", "eg_cointegrated_5pct"]]
              .to_string(index=False))

        spread_flags = detect_spread_discontinuities(spread_df, pair_label)
        all_spread_flags.append(spread_flags)
        overlap = leg_flag_overlap(spread_flags, leg_flags_all, tick_x, tick_y)
        print(f"\n  Spread-level discontinuities (5-sigma, {SPREAD_ROLL_WINDOW}d trailing std of spread diff): "
              f"{len(spread_flags)}")
        if len(spread_flags):
            print(spread_flags.to_string(index=False))
        print(f"  Leg-level flags for {tick_x} or {tick_y} (already saved, unchanged): {overlap['n_leg_flags_this_pair']}")
        print(f"  Exact-date overlap (spread flag date == a leg flag date): {overlap['n_exact_date_overlap']}"
              f" / {overlap['n_spread_flags']} spread flags")
        print(f"  Within +/-3 calendar days of any leg flag: {overlap['n_near_overlap_within_3cal_days']}"
              f" / {overlap['n_spread_flags']} spread flags")

        ou = ou_half_life(spread_df["spread"])
        adf_res = adf_full_sample(spread_df["spread"])
        print(f"\n  OU half-life (full-sample descriptive fit): a={ou['ou_a']:.6f}  b={ou['ou_b']:.6f}  "
              f"mean_reverting={ou['mean_reverting']}  half_life_days={ou['ou_half_life_days']}")
        print(f"  ADF on full spread series: stat={adf_res['adf_stat']:.4f}  p={adf_res['adf_pvalue']:.4f}  "
              f"stationary_5pct={adf_res['stationary_5pct']}")

        z = rolling_zscore(spread_df["spread"])
        spread_df = spread_df.copy()
        spread_df["zscore"] = z
        spread_df.insert(0, "pair", pair_label)
        all_spread_rows.append(spread_df.reset_index())

        plot_path = plot_pair(pair_label, spread_df, z, spread_flags)
        print(f"\n  Saved plot: {plot_path}")

        final_params = ({"final_alpha": float(fit_df.iloc[-1]["alpha"]), "final_beta": float(fit_df.iloc[-1]["beta"])}
                        if len(fit_df) else {"final_alpha": np.nan, "final_beta": np.nan})
        diag_rows.append({
            "pair": pair_label, "n_obs": len(df),
            "n_refits": len(fit_df),
            "n_coint_windows_5pct": int(n_coint),
            "pct_coint_windows_5pct": float(n_coint / len(fit_df)) if len(fit_df) else np.nan,
            **final_params,
            **adf_res, **ou,
            **overlap,
        })

    fit_all = pd.concat(all_fit_rows, ignore_index=True)
    spread_all = pd.concat(all_spread_rows, ignore_index=True)
    flags_all = pd.concat(all_spread_flags, ignore_index=True) if all_spread_flags else pd.DataFrame()
    diag_df = pd.DataFrame(diag_rows)

    fit_all.to_csv(os.path.join(RESULTS_DIR, "task1_hedge_ratio_history.csv"), index=False)
    spread_all.to_csv(os.path.join(RESULTS_DIR, "task1_spread_series.csv"), index=False)
    flags_all.to_csv(os.path.join(RESULTS_DIR, "task1_spread_discontinuities.csv"), index=False)
    diag_df.to_csv(os.path.join(RESULTS_DIR, "task1_pair_diagnostics.csv"), index=False)

    print("\n" + "=" * 100)
    print("  SUMMARY -- both active pairs")
    print("=" * 100)
    print(diag_df.to_string(index=False))

    print(f"\nSaved: {os.path.join(RESULTS_DIR, 'task1_hedge_ratio_history.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1_spread_series.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1_spread_discontinuities.csv')}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'task1_pair_diagnostics.csv')}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
