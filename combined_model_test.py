"""
combined_model_test.py
=======================
Strict walk-forward OOS comparison of three volatility models for SPY:

  Model 1 -- HAR+TS  : OLS on (RV1, RV2, RV5, RV22)  [PACF-informed HAR]
  Model 2 -- EGARCH   : EGARCH(1,1,1) 1-step cond. vol from returns
  Model 3 -- Combined : OLS on (RV1, RV2, RV5, RV22, egarch_sigma)

Target: 1-day-ahead Garman-Klass Realized Volatility (daily %).

Walk-forward protocol
  - Expanding window, refit every REFIT_FREQ trading days (default 21).
  - At step t, models are fit only on data strictly up to t-1.
  - EGARCH sigma_t is the 1-step-ahead filtered vol from the end of the
    training window, NOT from the test day itself (no leakage).

Metrics: RMSE, MAE, QLIKE, Diebold-Mariano (HLN-corrected, MSE loss).
Diagnostics (on final-window residuals): Ljung-Box, Ljung-Box sq, ARCH-LM.

Dependencies: yfinance, arch, statsmodels, numpy, pandas, scipy, matplotlib
"""

import sys
import io
import os
import warnings
import logging
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows to avoid encoding errors
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (all defaults overridable via argparse)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    ticker       = "SPY",
    start        = "2011-01-01",  # Long history so walk-forward has depth
    test_months  = 18,            # ~1.5 yr OOS window gives stable DM tests
    refit_freq   = 21,            # Monthly refit (balance between bias & speed)
    min_train    = 500,           # Minimum training obs before first forecast
    sim_paths    = 1000,          # MC paths for EGARCH 1-step forecast
    out_dir      = "results",
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(ticker, start):
    log.info(f"Fetching {ticker} from {start}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        log.error("No data returned. Check ticker and internet connection.")
        sys.exit(1)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    log.info(f"  {len(df)} trading days  |  {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def compute_features(df):
    """
    Garman-Klass RV  (daily %)  +  HAR lags  +  log-returns.
    All lags are formed with .shift(1) so they use information from t-1 only.
    """
    # GK daily variance
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    rv = np.sqrt(np.maximum(gk_var, 1e-10)) * 100.0   # % units

    log_ret = np.log(df["Close"] / df["Close"].shift(1)) * 100.0

    data = pd.DataFrame({"rv": rv, "return": log_ret})

    # HAR + PACF-motivated lags — shift(1) so features are available at t
    data["rv1"]  = data["rv"].shift(1)
    data["rv2"]  = data["rv"].shift(2)                   # PACF ≈ 0.30
    data["rv5"]  = data["rv"].shift(1).rolling(5).mean()
    data["rv22"] = data["rv"].shift(1).rolling(22).mean()

    return data.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def qlike(actual, forecast):
    """QLIKE loss: E[ sigma^2/h^2 - log(sigma^2/h^2) - 1 ]"""
    ratio = (actual ** 2) / (forecast ** 2)
    return float(np.mean(ratio - np.log(ratio) - 1))


def dm_test_hln(actual, f1, f2, loss="mse"):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold finite-sample correction.
    H0: equal predictive accuracy.
    Negative stat => f2 is better (smaller loss).
    """
    a, p1, p2 = np.asarray(actual), np.asarray(f1), np.asarray(f2)
    if loss == "mse":
        d = (a - p1)**2 - (a - p2)**2
    elif loss == "mae":
        d = np.abs(a - p1) - np.abs(a - p2)
    elif loss == "qlike":
        def _ql(x, h): return x**2/h**2 - np.log(x**2/h**2) - 1
        d = _ql(a, p1) - _ql(a, p2)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    n = len(d)
    mean_d = np.mean(d)
    # Newey-West variance with bandwidth 0 (no autocorrelation correction for h=1)
    var_d  = np.var(d, ddof=1)
    # HLN correction factor
    k = np.sqrt((n + 1 - 2 + 1/n) / n)
    dm_stat = (mean_d / np.sqrt(var_d / n)) * k
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)


def residual_diagnostics(resid, label, lags=20):
    """Return dict with LB(resid), LB(resid^2), ARCH-LM, and significant PACF lags."""
    z = np.asarray(resid, dtype=float)
    z = z[np.isfinite(z)]

    lb_res = acorr_ljungbox(z, lags=[lags], return_df=True)
    lb_sq  = acorr_ljungbox(z**2, lags=[lags], return_df=True)
    arch_  = het_arch(z, nlags=min(10, len(z)//4))

    ci = 1.96 / np.sqrt(len(z))
    pacf_vals = pacf(z, nlags=min(30, len(z)//4), method="ywm")
    sig_pacf  = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > ci]

    return {
        "lb_resid_stat": float(lb_res["lb_stat"].iloc[0]),
        "lb_resid_pval": float(lb_res["lb_pvalue"].iloc[0]),
        "lb_sq_stat":    float(lb_sq["lb_stat"].iloc[0]),
        "lb_sq_pval":    float(lb_sq["lb_pvalue"].iloc[0]),
        "arch_stat":     float(arch_[0]),
        "arch_pval":     float(arch_[1]),
        "sig_pacf_lags": sig_pacf,
    }


def _metrics(actual, forecast):
    a, f = np.asarray(actual), np.asarray(forecast)
    return {
        "RMSE":  float(np.sqrt(np.mean((a - f)**2))),
        "MAE":   float(np.mean(np.abs(a - f))),
        "QLIKE": qlike(a, f),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(data, test_months, refit_freq, min_train, sim_paths):
    split_date = data.index[-1] - pd.DateOffset(months=test_months)
    test_data  = data[data.index > split_date]
    test_dates = test_data.index

    log.info(f"Test period: {test_dates[0].date()} -> {test_dates[-1].date()} ({len(test_dates)} obs)")

    HAR_FEATURES = ["rv1", "rv2", "rv5", "rv22"]

    har_fc   = []
    eg_fc    = []
    comb_fc  = []
    actuals  = []

    # Cached EGARCH params (updated at refit boundaries)
    eg_params = None
    har_coef  = None   # (intercept, coef array)
    comb_coef = None   # (intercept, coef array)

    for i, tdate in enumerate(test_dates):
        # Strict: all data BEFORE tdate
        hist = data[data.index < tdate]

        if len(hist) < min_train:
            log.warning(f"Skipping {tdate.date()}: insufficient training data ({len(hist)} < {min_train})")
            continue

        returns_hist = hist["return"]
        rv_hist      = hist["rv"]

        # ── REFIT at boundary ───────────────────────────────────────────────
        do_refit = (i % refit_freq == 0) or (eg_params is None)

        if do_refit:
            log.info(f"Refitting at {tdate.date()} (step {i}/{len(test_dates)})...")

            # --- EGARCH ---
            am      = arch_model(returns_hist, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
            eg_res  = am.fit(starting_values=eg_params, disp="off", show_warning=False)
            eg_params = eg_res.params.values

            # --- HAR+TS ---
            X_train = hist[HAR_FEATURES].values
            y_train = rv_hist.values
            har_mod = LinearRegression().fit(X_train, y_train)
            har_coef = (har_mod.intercept_, har_mod.coef_)

        # ── EGARCH 1-step-ahead sigma from end of training window ──────────
        # Use fixed params to filter up to t-1, then read off the next-step variance.
        # `fix()` re-runs the Kalman filter with the supplied params — no future leakage.
        am_fix      = arch_model(returns_hist, vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)
        eg_fixed    = am_fix.fix(eg_params)
        eg_sigma_t  = float(eg_fixed.conditional_volatility.iloc[-1])   # sigma at last training obs
        # This is the EGARCH estimate for the CURRENT period end, i.e. our 1-step forecast
        # for what volatility looked like at t. We use it as a predictor in the combined model.

        # ── HAR+TS forecast ────────────────────────────────────────────────
        x_test = data.loc[tdate, HAR_FEATURES].values.reshape(1, -1)
        har_pred = float(har_coef[0] + np.dot(har_coef[1], x_test[0]))
        har_pred = max(har_pred, 1e-4)

        # ── Combined forecast (HAR+TS features + EGARCH sigma) ────────────
        if do_refit:
            eg_cond_vol_train = eg_res.conditional_volatility.values
            # Align EGARCH conditional vol with training RV (both length N)
            # egarch conditional vol at time t is sigma_t|t-1 (1-step filter from returns)
            # we pair it with rv (realized for that day) for the training regression
            n_align = min(len(eg_cond_vol_train), len(rv_hist))
            X_comb_train = np.column_stack([
                hist[HAR_FEATURES].values[-n_align:],
                eg_cond_vol_train[-n_align:]
            ])
            y_comb_train = rv_hist.values[-n_align:]
            comb_mod  = LinearRegression().fit(X_comb_train, y_comb_train)
            comb_coef = (comb_mod.intercept_, comb_mod.coef_)

        x_comb_test = np.append(x_test[0], eg_sigma_t).reshape(1, -1)
        comb_pred   = float(comb_coef[0] + np.dot(comb_coef[1], x_comb_test[0]))
        comb_pred   = max(comb_pred, 1e-4)

        # ── EGARCH standalone 1-step forecast vol ──────────────────────────
        # Use simulation to draw 1-step forecast from end of history
        try:
            fc_eg = eg_fixed.forecast(horizon=1, method="simulation",
                                      simulations=sim_paths, reindex=False)
            eg_pred = float(np.sqrt(fc_eg.variance.iloc[-1, 0]))
        except Exception:
            eg_pred = eg_sigma_t   # fallback: use filtered value
        eg_pred = max(eg_pred, 1e-4)

        har_fc.append(har_pred)
        eg_fc.append(eg_pred)
        comb_fc.append(comb_pred)
        actuals.append(float(data.loc[tdate, "rv"]))

    return (
        np.array(actuals),
        np.array(har_fc),
        np.array(eg_fc),
        np.array(comb_fc),
        test_dates[-len(actuals):],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_separator(char="=", width=72):
    print(char * width)

def report(actuals, har_fc, eg_fc, comb_fc, test_dates, cfg):
    SEP  = "=" * 72
    sep2 = "-" * 72
    N    = len(actuals)

    print(SEP)
    print("  COMBINED MODEL COMPARISON — SPY Garman-Klass Realized Volatility")
    print(f"  Test period : {test_dates[0].date()} -> {test_dates[-1].date()}  (N={N})")
    print(f"  Refit freq  : every {cfg['refit_freq']} trading days")
    print(SEP)

    # ── 4a. Accuracy Metrics ─────────────────────────────────────────────────
    mets = {
        "HAR+TS (M1)":   _metrics(actuals, har_fc),
        "EGARCH (M2)":   _metrics(actuals, eg_fc),
        "Combined (M3)": _metrics(actuals, comb_fc),
    }

    print(f"\n  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'QLIKE':>10}")
    print(f"  {sep2}")
    for name, m in mets.items():
        print(f"  {name:<20} {m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['QLIKE']:>10.4f}")

    # ── 4b. Diebold-Mariano Tests ─────────────────────────────────────────────
    print(f"\n  DIEBOLD-MARIANO TESTS  (H0: equal accuracy | negative stat => alt wins)")
    print(f"  {sep2}")

    dm_comparisons = [
        ("M3 vs M1 (Combined vs HAR+TS)", actuals, har_fc, comb_fc),
        ("M3 vs M2 (Combined vs EGARCH)", actuals, eg_fc,  comb_fc),
        ("M2 vs M1 (EGARCH vs HAR+TS)",  actuals, har_fc, eg_fc),
    ]

    for label, a, f1, f2 in dm_comparisons:
        for loss in ("mse", "mae", "qlike"):
            stat, pval = dm_test_hln(a, f1, f2, loss=loss)
            sig   = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            arrow = "-> alt wins" if stat < 0 and pval < 0.10 else ("-> base wins" if stat > 0 and pval < 0.10 else "-> no sig diff")
            print(f"  {label:<40} [{loss.upper():>5}]  stat={stat:+.3f}  p={pval:.4f} {sig:3s}  {arrow}")

    # ── 4c. Residual Diagnostics ──────────────────────────────────────────────
    print(f"\n  RESIDUAL DIAGNOSTICS  (on OOS residuals, lag={20})")
    print(f"  {sep2}")

    for name, fc in [("HAR+TS", har_fc), ("EGARCH", eg_fc), ("Combined", comb_fc)]:
        resid = actuals - fc
        diag  = residual_diagnostics(resid, name, lags=20)
        print(f"\n  [{name}]")
        print(f"    LB(resid  lag 20): stat={diag['lb_resid_stat']:.2f}  p={diag['lb_resid_pval']:.4f}"
              f"  {'REJECT (structure left)' if diag['lb_resid_pval'] < 0.05 else 'OK'}")
        print(f"    LB(resid^2 lag 20): stat={diag['lb_sq_stat']:.2f}  p={diag['lb_sq_pval']:.4f}"
              f"  {'REJECT (ARCH effects left)' if diag['lb_sq_pval'] < 0.05 else 'OK'}")
        print(f"    ARCH-LM (lag 10): stat={diag['arch_stat']:.2f}  p={diag['arch_pval']:.4f}"
              f"  {'REJECT' if diag['arch_pval'] < 0.05 else 'OK'}")
        if diag["sig_pacf_lags"]:
            print(f"    Significant residual PACF lags: {diag['sig_pacf_lags']}")
        else:
            print(f"    No significant residual PACF lags (model is well-specified)")

    # ── 4d. Verdict ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  VERDICT")
    print(f"  {sep2}")
    best_model = min(mets, key=lambda k: mets[k]["RMSE"])
    print(f"  Best OOS RMSE: {best_model} ({mets[best_model]['RMSE']:.4f})")
    best_qlike = min(mets, key=lambda k: mets[k]["QLIKE"])
    print(f"  Best QLIKE:    {best_qlike} ({mets[best_qlike]['QLIKE']:.4f})")

    # DM: does Combined significantly beat HAR+TS?
    stat_m3_m1, pval_m3_m1 = dm_test_hln(actuals, har_fc, comb_fc, loss="mse")
    if pval_m3_m1 < 0.05 and stat_m3_m1 < 0:
        print("  [+] EGARCH ADDS VALUE: Combined significantly beats HAR+TS (p={:.4f})".format(pval_m3_m1))
    elif pval_m3_m1 < 0.10 and stat_m3_m1 < 0:
        print("  [~] WEAK EVIDENCE: Combined marginally better than HAR+TS (p={:.4f})".format(pval_m3_m1))
    else:
        print("  [-] EGARCH ADDS NO VALUE: Combined does NOT significantly beat HAR+TS (p={:.4f})".format(pval_m3_m1))
        if mets["HAR+TS (M1)"]["RMSE"] < mets["EGARCH (M2)"]["RMSE"]:
            print("  [-] HAR+TS alone outperforms standalone EGARCH on RMSE.")

    print(SEP + "\n")
    return mets


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(actuals, har_fc, eg_fc, comb_fc, test_dates, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Combined EGARCH+HAR Model Comparison — SPY GK-RV", fontsize=13, fontweight="bold")

    # ── Panel 1: Forecasts vs Actuals ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(test_dates, actuals, color="grey",   lw=1.2, alpha=0.8, label="Actual GK-RV")
    ax.plot(test_dates, har_fc,  color="#3498db", lw=1.0, alpha=0.85, label="HAR+TS (M1)")
    ax.plot(test_dates, eg_fc,   color="#e74c3c", lw=1.0, alpha=0.85, label="EGARCH (M2)")
    ax.plot(test_dates, comb_fc, color="#2ecc71", lw=1.2, alpha=0.90, label="Combined (M3)")
    ax.set_title("OOS Forecasts vs Actual GK-RV", fontweight="bold")
    ax.set_ylabel("Volatility (% / day)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 2: Absolute Errors ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(test_dates, np.abs(actuals - har_fc),  color="#3498db", lw=0.9, alpha=0.75, label="HAR+TS")
    ax.plot(test_dates, np.abs(actuals - eg_fc),   color="#e74c3c", lw=0.9, alpha=0.75, label="EGARCH")
    ax.plot(test_dates, np.abs(actuals - comb_fc), color="#2ecc71", lw=0.9, alpha=0.75, label="Combined")
    ax.set_title("Absolute Forecast Error", fontweight="bold")
    ax.set_ylabel("|Error| (% / day)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Panel 3: RMSE bar chart ───────────────────────────────────────────────
    ax = axes[1, 0]
    names  = ["HAR+TS\n(M1)", "EGARCH\n(M2)", "Combined\n(M3)"]
    rmses  = [
        float(np.sqrt(np.mean((actuals - har_fc)**2))),
        float(np.sqrt(np.mean((actuals - eg_fc)**2))),
        float(np.sqrt(np.mean((actuals - comb_fc)**2))),
    ]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    bars   = ax.bar(names, rmses, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("OOS RMSE Comparison", fontweight="bold")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)

    # ── Panel 4: QLIKE bar chart ──────────────────────────────────────────────
    ax = axes[1, 1]
    qlikes = [
        qlike(actuals, har_fc),
        qlike(actuals, eg_fc),
        qlike(actuals, comb_fc),
    ]
    bars = ax.bar(names, qlikes, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, qlikes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("OOS QLIKE Comparison", fontweight="bold")
    ax.set_ylabel("QLIKE")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "combined_model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HAR+TS vs EGARCH vs Combined walk-forward test")
    p.add_argument("--ticker",      default=DEFAULTS["ticker"])
    p.add_argument("--start",       default=DEFAULTS["start"])
    p.add_argument("--test-months", type=int, default=DEFAULTS["test_months"])
    p.add_argument("--refit-freq",  type=int, default=DEFAULTS["refit_freq"])
    p.add_argument("--min-train",   type=int, default=DEFAULTS["min_train"])
    p.add_argument("--sim-paths",   type=int, default=DEFAULTS["sim_paths"])
    p.add_argument("--out-dir",     default=DEFAULTS["out_dir"])
    return p.parse_args()


def main():
    cfg  = vars(parse_args())
    df   = fetch_data(cfg["ticker"], cfg["start"])
    data = compute_features(df)

    log.info(f"Total observations after feature construction: {len(data)}")

    actuals, har_fc, eg_fc, comb_fc, test_dates = walk_forward(
        data,
        test_months = cfg["test_months"],
        refit_freq  = cfg["refit_freq"],
        min_train   = cfg["min_train"],
        sim_paths   = cfg["sim_paths"],
    )

    mets = report(actuals, har_fc, eg_fc, comb_fc, test_dates, cfg)
    make_plots(actuals, har_fc, eg_fc, comb_fc, test_dates, cfg["out_dir"])

    log.info("Done.")


if __name__ == "__main__":
    main()
